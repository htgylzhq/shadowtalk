import Foundation

struct BackendConfiguration {
    let alignURL: URL
    let language: String
    let referenceID: String?

    init(bundle: Bundle = .main, environment: [String: String] = ProcessInfo.processInfo.environment) throws {
        func value(_ key: String) -> String? {
            let value = environment[key] ?? bundle.object(forInfoDictionaryKey: key) as? String
            return value.flatMap { $0.isEmpty ? nil : $0 }
        }

        let base = value("SHADOWTALK_BACKEND_URL") ?? "http://localhost:8000"
        guard var components = URLComponents(string: base),
              let scheme = components.scheme?.lowercased(),
              let host = components.host, !host.isEmpty,
              components.query == nil, components.fragment == nil,
              scheme == "http" || scheme == "https" else {
            throw URLError(.badURL)
        }
        components.scheme = scheme == "https" ? "wss" : "ws"
        if !components.path.hasSuffix("/") { components.path += "/" }
        components.path += "ws/align"
        guard let url = components.url else { throw URLError(.badURL) }
        alignURL = url
        language = value("SHADOWTALK_LANGUAGE") ?? "ja"
        referenceID = value("SHADOWTALK_REFERENCE_ID")
    }
}
