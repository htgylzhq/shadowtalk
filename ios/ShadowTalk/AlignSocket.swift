import Foundation

struct AlignSocket {
    enum ProtocolError: Error {
        case invalidMessage
        case rejected
    }

    enum Frame {
        case audio(Data)
        case end
    }

    let configuration: BackendConfiguration

    func score(
        frames: AsyncStream<Frame>,
        onReady: @escaping @MainActor @Sendable () -> Void,
        onEndpoint: @escaping @MainActor @Sendable () -> Void
    ) async throws -> ScorePayload {
        let task = URLSession.shared.webSocketTask(with: configuration.alignURL)
        defer { task.cancel(with: .normalClosure, reason: nil) }
        task.resume()

        let start: [String: String?] = [
            "type": "start", "language": configuration.language, "reference_id": configuration.referenceID
        ]
        let startJSON = String(decoding: try JSONEncoder().encode(start), as: UTF8.self)
        return try await withTaskCancellationHandler {
            try await withThrowingTaskGroup(of: ScorePayload?.self) { group in
                group.addTask {
                    try await task.send(.string(startJSON))
                    for await frame in frames {
                        try Task.checkCancellation()
                        switch frame {
                        case .audio(let data): try await task.send(.data(data))
                        case .end: try await task.send(.string(#"{"type":"end"}"#))
                        }
                    }
                    return nil
                }
                group.addTask {
                    let decoder = JSONDecoder()
                    while !Task.isCancelled {
                        guard case .string(let text) = try await task.receive() else { continue }
                        let data = Data(text.utf8)
                        do {
                            switch try decoder.decode(Message.self, from: data).type {
                            case "ready": await onReady()
                            case "endpoint": await onEndpoint()
                            case "score": return try decoder.decode(ScorePayload.self, from: data)
                            case "error": throw ProtocolError.rejected
                            default: break
                            }
                        } catch is DecodingError {
                            throw ProtocolError.invalidMessage
                        }
                    }
                    throw CancellationError()
                }
                do {
                    for try await response in group {
                        if let response {
                            group.cancelAll()
                            task.cancel(with: .normalClosure, reason: nil)
                            return response
                        }
                    }
                } catch {
                    group.cancelAll()
                    task.cancel(with: .normalClosure, reason: nil)
                    throw error
                }
                throw ProtocolError.invalidMessage
            }
        } onCancel: {
            task.cancel(with: .normalClosure, reason: nil)
        }
    }

    private struct Message: Decodable {
        let type: String
    }
}
