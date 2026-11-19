import AVFoundation
import Foundation

/// Captures microphone audio for shadowing and streams it to the backend.
///
/// Design constraints (see README):
/// - Native AVFoundation only — no cross-platform audio layer.
/// - AVAudioSession in .playAndRecord so reference playback and capture
///   can overlap, as shadowing requires.
/// - Capture is converted to 16 kHz mono PCM and chunked into 20 ms
///   frames (320 samples) for the /ws/align WebSocket.
@MainActor
final class RecorderViewModel: ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var lastScore: Double?

    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?

    /// Callback for each 20 ms PCM frame; the transport layer forwards
    /// these to the backend WebSocket as Opus frames.
    var onFrame: ((Data) -> Void)?

    /// Day 2 transport: streams frames to /ws/align and receives the
    /// VAD endpoint + score. Defaults to the local backend.
    private let socket = AlignSocket()
    var language = "ja"
    var referenceId: String?
    @Published private(set) var connectionState = "未连接"

    init() {
        socket.onScore = { [weak self] gop in
            Task { @MainActor in self?.lastScore = gop }
        }
        socket.onEndpoint = { [weak self] _ in
            // VAD decided the utterance ended; the score follows.
            Task { @MainActor in self?.stopEngineOnly() }
        }
        socket.onStateChange = { [weak self] state in
            Task { @MainActor in self?.connectionState = state.rawValue }
        }
        // Forward every recorder frame to the backend as it arrives.
        onFrame = { [weak self] data in self?.socket.sendFrame(data) }
    }

    private static let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatInt16,
        sampleRate: 16_000,
        channels: 1,
        interleaved: true
    )!

    func start() {
        socket.connect()
        socket.sendStart(language: language, referenceId: referenceId)
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .spokenAudio,
                                    options: [.duckOthers, .allowBluetooth])
            try session.setActive(true)
        } catch {
            print("Audio session setup failed: \(error)")
            return
        }

        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        converter = AVAudioConverter(from: inputFormat, to: Self.targetFormat)

        // 1024-frame tap buffer; frames are re-chunked to 20 ms below.
        input.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) {
            [weak self] buffer, _ in
            self?.convertAndEmit(buffer)
        }

        engine.prepare()
        do {
            try engine.start()
            isRecording = true
        } catch {
            print("Audio engine start failed: \(error)")
        }
    }

    func stop() {
        socket.sendEnd()
        stopEngineOnly()
    }

    /// Stops capture without sending another "end" (used when the
    /// server already endpointed the utterance via VAD).
    private func stopEngineOnly() {
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        try? AVAudioSession.sharedInstance().setActive(false)
        isRecording = false
    }

    private func convertAndEmit(_ buffer: AVAudioPCMBuffer) {
        guard let converter else { return }
        let ratio = Self.targetFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1
        guard let converted = AVAudioPCMBuffer(
            pcmFormat: Self.targetFormat, frameCapacity: capacity
        ) else { return }

        var error: NSError?
        converter.convert(to: converted, error: &error) { _, status in
            status.pointee = .haveData
            return buffer
        }
        if let error {
            print("Conversion failed: \(error)")
            return
        }

        // Emit 20 ms frames: 320 samples * 2 bytes = 640 bytes of PCM16.
        guard let channel = converted.int16ChannelData?[0] else { return }
        let frameSamples = 320
        let totalSamples = Int(converted.frameLength)
        var offset = 0
        while offset + frameSamples <= totalSamples {
            let data = Data(bytes: channel + offset,
                            count: frameSamples * MemoryLayout<Int16>.size)
            onFrame?(data)
            offset += frameSamples
        }
    }
}
