import Foundation

/// WebSocket transport for the ShadowTalk alignment session.
///
/// Day 2: the recorder's 20 ms PCM frames are streamed live to
/// `ws://<host>/ws/align`. The server replies with:
/// - `{"type": "ready", "session_id": ...}` after `start`
/// - `{"type": "endpoint", "reason": "vad"|"client_end", ...}` when VAD
///   decides the utterance ended (or the client sent `end`)
/// - `{"type": "score", ...}` with the per-unit scores for the utterance
///
/// Default URL is the local backend; point it at a deployed host before
/// a TestFlight build. Not compiled on Linux — verify with an Xcode build.
final class AlignSocket: NSObject, URLSessionWebSocketDelegate {
    enum State: String {
        case disconnected, connecting, connected
    }

    var onScore: ((Double?) -> Void)?
    var onEndpoint: ((String) -> Void)?
    var onStateChange: ((State) -> Void)?

    private(set) var state: State = .disconnected {
        didSet { onStateChange?(state) }
    }

    private let url: URL
    private var task: URLSessionWebSocketTask?
    private lazy var session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)

    init(url: URL = URL(string: "ws://localhost:8000/ws/align")!) {
        self.url = url
    }

    func connect() {
        guard task == nil else { return }
        state = .connecting
        task = session.webSocketTask(with: url)
        task?.resume()
        state = .connected
        receiveLoop()
    }

    func disconnect() {
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
        state = .disconnected
    }

    func sendStart(language: String, referenceId: String?) {
        sendJSON([
            "type": "start",
            "language": language,
            "reference_id": referenceId as Any,
        ])
    }

    func sendFrame(_ data: Data) {
        task?.send(.data(data)) { error in
            if let error { print("AlignSocket frame send failed: \(error)") }
        }
    }

    func sendEnd() {
        sendJSON(["type": "end"])
    }

    private func sendJSON(_ object: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: object),
              let text = String(data: data, encoding: .utf8) else { return }
        task?.send(.string(text)) { error in
            if let error { print("AlignSocket message send failed: \(error)") }
        }
    }

    private func receiveLoop() {
        task?.receive { [weak self] result in
            guard let self, let task = self.task else { return }
            switch result {
            case .success(.string(let text)):
                self.handleMessage(text)
            case .success(.data):
                break // Server never sends binary.
            case .failure(let error):
                print("AlignSocket receive failed: \(error)")
                self.task = nil
                self.state = .disconnected
                return
            case .success:
                break
            }
            // Keep listening until the task is cancelled.
            if task.state == .running { self.receiveLoop() }
        }
    }

    private func handleMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        switch json["type"] as? String {
        case "score":
            onScore?(json["overall_gop"] as? Double)
        case "endpoint":
            onEndpoint?(json["reason"] as? String ?? "vad")
        default:
            break
        }
    }
}
