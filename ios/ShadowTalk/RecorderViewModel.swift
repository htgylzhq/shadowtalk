import AVFoundation
import Combine
import Foundation

struct ScoreReport {
    let payload: ScorePayload
    let source: Source

    enum Source {
        case recording, sample, offlineSample
    }
}

@MainActor
final class RecorderViewModel: ObservableObject {
    enum Phase {
        case idle, preparing, recording, recordingOffline, scoring

        var isRecording: Bool { self == .recording || self == .recordingOffline }
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var error: String?
    @Published var report: ScoreReport?

    private lazy var engine = AVAudioEngine()
    private var frames: AsyncStream<AlignSocket.Frame>.Continuation?
    private var scoringTask: Task<Void, Never>?
    private var timeoutTask: Task<Void, Never>?
    private var isReady = false

    func start() async {
        guard phase == .idle else { return }
        error = nil
        phase = .preparing
        guard await AVAudioApplication.requestRecordPermission() else {
            fail("Microphone access is off. Enable it for ShadowTalk in Settings, or view a sample score.")
            return
        }

        let configuration: BackendConfiguration
        do {
            configuration = try BackendConfiguration()
        } catch let error as URLError where error.code == .appTransportSecurityRequiresSecureConnection {
            fail("Use an HTTPS URL for a remote scoring server.")
            return
        } catch {
            fail("The scoring server URL is invalid.")
            return
        }

        let stream = AsyncStream.makeStream(of: AlignSocket.Frame.self)
        do {
            try startCapture(stream.continuation)
        } catch {
            fail("Could not start recording. Check your microphone and try again.")
            return
        }
        frames = stream.continuation
        phase = .recording
        isReady = false
        setTimeout(.seconds(8)) { [weak self] in
            guard let self, !isReady else { return }
            if phase == .recording {
                recordOffline()
            } else if phase == .scoring {
                finish()
                showSample(source: .offlineSample)
            }
        }
        scoringTask = Task {
            do {
                let payload = try await AlignSocket(configuration: configuration).score(
                    frames: stream.stream,
                    onReady: { [weak self] in
                        guard !Task.isCancelled else { return }
                        guard let self else { return }
                        isReady = true
                        if phase == .scoring {
                            startScoreTimeout()
                        } else {
                            timeoutTask?.cancel()
                        }
                    },
                    onEndpoint: { [weak self] in
                        guard !Task.isCancelled else { return }
                        self?.endRecording(sendEnd: false)
                    }
                )
                guard !Task.isCancelled else { return }
                if payload.emptyRecording {
                    fail("No audio was captured. Tap Record and speak before stopping.")
                    return
                }
                finish()
                report = ScoreReport(payload: payload, source: .recording)
            } catch {
                guard !Task.isCancelled else { return }
                if error is AlignSocket.ProtocolError {
                    fail("The scoring server didn’t return a valid result.")
                } else if isReady {
                    if (error as? URLError)?.code == .timedOut {
                        fail("The scoring server took too long to respond. Try again.")
                    } else {
                        fail("The connection to the scoring server was lost. Record again.")
                    }
                } else if phase.isRecording {
                    recordOffline()
                } else {
                    finish()
                    showSample(source: .offlineSample)
                }
            }
        }
    }

    func stop() {
        endRecording(sendEnd: true)
    }

    private func endRecording(sendEnd: Bool) {
        guard phase.isRecording else { return }
        if phase == .recordingOffline {
            finish()
            showSample(source: .offlineSample)
            return
        }
        stopCapture(sendEnd: sendEnd)
        phase = .scoring
        if isReady { startScoreTimeout() }
    }

    private func startScoreTimeout() {
        setTimeout(.seconds(30)) { [weak self] in
            self?.fail("The scoring server took too long to respond. Try again.")
        }
    }

    private func setTimeout(_ duration: Duration, action: @escaping @MainActor () -> Void) {
        timeoutTask?.cancel()
        timeoutTask = Task {
            do {
                try await Task.sleep(for: duration)
            } catch { return }
            action()
        }
    }

    private func recordOffline() {
        scoringTask?.cancel()
        scoringTask = nil
        timeoutTask?.cancel()
        frames?.finish()
        phase = .recordingOffline
    }

    func showSample(source: ScoreReport.Source = .sample) {
        error = nil
        do {
            report = ScoreReport(payload: try .sample(), source: source)
        } catch {
            fail("The sample score couldn’t be loaded.")
        }
    }

    private func startCapture(_ continuation: AsyncStream<AlignSocket.Frame>.Continuation) throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetoothHFP])
        try session.setActive(true)
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0,
              let encoder = PCMFrameEncoder(inputFormat: format, continuation: continuation) else {
            try? session.setActive(false)
            throw RecorderError.invalidInputFormat
        }
        encoder.installTap(on: input, format: format)
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            try? session.setActive(false)
            throw error
        }
    }

    private func stopCapture(sendEnd: Bool = false) {
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        try? AVAudioSession.sharedInstance().setActive(false)
        if sendEnd { frames?.yield(.end) }
        frames?.finish()
        frames = nil
    }

    private func finish() {
        if phase.isRecording { stopCapture() }
        timeoutTask?.cancel()
        timeoutTask = nil
        scoringTask?.cancel()
        scoringTask = nil
        phase = .idle
    }

    private func fail(_ message: String) {
        finish()
        error = message
    }

    private enum RecorderError: Error {
        case invalidInputFormat
    }
}

private final class PCMFrameEncoder {
    private static let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: true
    )!
    private static let frameBytes = 640

    private let converter: AVAudioConverter
    private let continuation: AsyncStream<AlignSocket.Frame>.Continuation
    private var pending = Data()

    init?(inputFormat: AVAudioFormat, continuation: AsyncStream<AlignSocket.Frame>.Continuation) {
        guard let converter = AVAudioConverter(from: inputFormat, to: Self.targetFormat) else { return nil }
        self.converter = converter
        self.continuation = continuation
    }

    func installTap(on input: AVAudioInputNode, format: AVAudioFormat) {
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [self] buffer, _ in
            encode(buffer)
        }
    }

    private func encode(_ buffer: AVAudioPCMBuffer) {
        let ratio = Self.targetFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(ceil(Double(buffer.frameLength) * ratio))
        guard capacity > 0, let converted = AVAudioPCMBuffer(
            pcmFormat: Self.targetFormat, frameCapacity: capacity
        ) else { return }

        var suppliedInput = false
        let status = converter.convert(to: converted, error: nil) { _, status in
            if suppliedInput {
                status.pointee = .noDataNow
                return nil
            }
            suppliedInput = true
            status.pointee = .haveData
            return buffer
        }
        guard status != .error, let channel = converted.int16ChannelData?[0] else { return }

        pending.append(Data(bytes: channel, count: Int(converted.frameLength) * MemoryLayout<Int16>.size))
        while pending.count >= Self.frameBytes {
            continuation.yield(.audio(Data(pending.prefix(Self.frameBytes))))
            pending.removeFirst(Self.frameBytes)
        }
    }
}
