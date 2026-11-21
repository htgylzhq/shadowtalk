import XCTest
@testable import ShadowTalk

final class ScorePayloadTests: XCTestCase {
    func testV1WithoutContourPreservesNullsAndConfidenceGating() throws {
        let json = """
        {
          "type": "score", "score_version": 1, "overall_gop": 0.75,
          "scores": [
            {"phoneme": "a", "gop": 0.5, "pitch_delta": -1.25, "flagged": false},
            {"phoneme": "t", "gop": 1.0, "pitch_delta": null, "flagged": false},
            {"phoneme": "a", "gop": null, "pitch_delta": null, "flagged": true}
          ]
        }
        """
        let payload = try JSONDecoder().decode(ScorePayload.self, from: Data(json.utf8))

        XCTAssertEqual(payload.scores.map(\.phoneme), ["a", "t", "a"])
        XCTAssertEqual(payload.meanGOP, 0.75)
        XCTAssertEqual(payload.flaggedCount, 1)
        XCTAssertEqual(payload.scores[0].displayPitchDelta, -1.25)
        XCTAssertNil(payload.scores[1].displayPitchDelta)
        XCTAssertNil(payload.scores[2].displayGOP)
        XCTAssertNil(payload.scores[2].displayPitchDelta)
        XCTAssertNil(payload.pitchContour)
    }

    func testAllFlaggedDoesNotDisplayZeroAsMean() throws {
        let json = #"{"scores":[{"phoneme":"a","flagged":true,"gop":null,"pitch_delta":null}]}"#
        let payload = try JSONDecoder().decode(ScorePayload.self, from: Data(json.utf8))

        XCTAssertNil(payload.meanGOP)
        XCTAssertEqual(payload.flaggedCount, 1)
    }

    func testFallbackScoresAndDisabledPitchAreLabeledAccurately() throws {
        let json = #"{"scores":[],"gop_source":"ctc_confidence","pitch":"unavailable"}"#
        let payload = try JSONDecoder().decode(ScorePayload.self, from: Data(json.utf8))

        XCTAssertEqual(payload.pronunciationMetric, "Alignment confidence")
        XCTAssertEqual(payload.missingPitchMessage, "Pitch extraction is not available on the server.")
    }

    func testBundledSampleHasComparableContoursWithUnvoicedGaps() throws {
        let sample = try ScorePayload.sample()
        let contour = try XCTUnwrap(sample.pitchContour)

        XCTAssertTrue(contour.isComparable)
        XCTAssertEqual(contour.frameSeconds, 0.05)
        XCTAssertEqual(contour.learner[0], -1.6)
        XCTAssertNil(contour.learner[9])
        XCTAssertNotEqual(contour.learner, contour.reference)
        XCTAssertEqual(sample.flaggedCount, 1)
    }

    func testBackendBaseURLMapsToWebSocketEndpoint() throws {
        let config = try BackendConfiguration(environment: [
            "SHADOWTALK_BACKEND_URL": "https://speech.example.org/api/",
            "SHADOWTALK_REFERENCE_ID": "lesson-1",
            "SHADOWTALK_LANGUAGE": "ja"
        ])
        XCTAssertEqual(config.alignURL.absoluteString, "wss://speech.example.org/api/ws/align")
        XCTAssertEqual(config.referenceID, "lesson-1")
        XCTAssertEqual(config.language, "ja")
        XCTAssertThrowsError(try BackendConfiguration(environment: ["SHADOWTALK_BACKEND_URL": "missing-host"]))
    }

    func testPlainHTTPOnlyAllowedOnLoopback() throws {
        for host in ["localhost", "127.0.0.1", "[::1]"] {
            let config = try BackendConfiguration(environment: ["SHADOWTALK_BACKEND_URL": "http://\(host):8000"])
            XCTAssertEqual(config.alignURL.absoluteString, "ws://\(host):8000/ws/align")
        }
        for host in ["server.local", "192.168.1.10", "speech.example.org"] {
            XCTAssertThrowsError(try BackendConfiguration(environment: ["SHADOWTALK_BACKEND_URL": "http://\(host)"])) {
                XCTAssertEqual(($0 as? URLError)?.code, .appTransportSecurityRequiresSecureConnection)
            }
        }
    }

    @MainActor
    func testSocketScoresStreamedFramesAgainstFixtureReference() async throws {
        guard let baseURL = ProcessInfo.processInfo.environment["SHADOWTALK_FIXTURE_URL"] else {
            throw XCTSkip("Set SHADOWTALK_FIXTURE_URL to run the live socket test.")
        }
        let config = try BackendConfiguration(environment: [
            "SHADOWTALK_BACKEND_URL": baseURL,
            "SHADOWTALK_REFERENCE_ID": "score-fixture"
        ])
        let (frames, writer) = AsyncStream.makeStream(of: AlignSocket.Frame.self)
        for _ in 0..<10 { writer.yield(.audio(Data(repeating: 0, count: 640))) }
        writer.yield(.end)
        writer.finish()

        let ready = expectation(description: "Server accepted the attempt")
        let endpoint = expectation(description: "Server ended the utterance")
        let score = try await AlignSocket(configuration: config).score(
            frames: frames, onReady: { ready.fulfill() }, onEndpoint: { endpoint.fulfill() }
        )
        await fulfillment(of: [ready, endpoint], timeout: 1, enforceOrder: true)
        XCTAssertEqual(score.scores.map(\.phoneme), ["こん", "にち"])
        XCTAssertEqual(score.flaggedCount, 1)
        XCTAssertEqual(score.pitchContour?.learner, Array(repeating: 0.0, count: 60))
        XCTAssertEqual(score.pitchContour?.reference.first, -3.0)
    }
}
