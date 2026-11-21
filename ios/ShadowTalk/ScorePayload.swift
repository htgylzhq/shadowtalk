import Foundation

struct ScorePayload: Decodable {
    let alignment: String?
    let scores: [PhonemeScore]
    let pitchContour: PitchContour?
    let pitchStatus: String?
    let gopSource: String?

    enum CodingKeys: String, CodingKey {
        case alignment, scores
        case pitchContour = "pitch_contour"
        case pitchStatus = "pitch"
        case gopSource = "gop_source"
    }

    var meanGOP: Double? {
        let values = scores.compactMap(\.displayGOP)
        guard !values.isEmpty else { return nil }
        return values.reduce(0, +) / Double(values.count)
    }

    var flaggedCount: Int { scores.filter(\.flagged).count }

    var pronunciationMetric: String {
        switch gopSource {
        case "ctc_confidence": "Alignment confidence"
        case "mixed": "Pronunciation score"
        default: "GOP"
        }
    }

    var missingPitchMessage: String {
        switch pitchStatus {
        case "missing": "The server has no reference audio for pitch comparison."
        case "unavailable": "Pitch extraction is not available on the server."
        case "error": "The server couldn't extract pitch from the audio."
        default: "Pitch comparison needs voiced audio from both recordings."
        }
    }

    static func sample(in bundle: Bundle = .main) throws -> Self {
        guard let url = bundle.url(forResource: "SampleScore", withExtension: "json") else {
            throw CocoaError(.fileNoSuchFile)
        }
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
}

struct PhonemeScore: Decodable {
    let phoneme: String
    let flagged: Bool
    private let gop: Double?
    private let pitchDelta: Double?

    enum CodingKeys: String, CodingKey {
        case phoneme, gop, flagged
        case pitchDelta = "pitch_delta"
    }

    var displayGOP: Double? { flagged ? nil : gop }
    var displayPitchDelta: Double? { flagged ? nil : pitchDelta }
}

struct PitchContour: Decodable {
    let frameSeconds: Double
    let learner: [Double?]
    let reference: [Double?]

    enum CodingKeys: String, CodingKey {
        case learner, reference
        case frameSeconds = "frame_s"
    }

    var isComparable: Bool {
        frameSeconds > 0 && learner.contains { $0 != nil } && reference.contains { $0 != nil }
    }
}
