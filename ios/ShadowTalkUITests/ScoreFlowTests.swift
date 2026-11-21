import XCTest

final class ScoreFlowTests: XCTestCase {
    func testSampleScoreOpensAndReturnsToRecorder() {
        let app = XCUIApplication()
        app.launch()

        for _ in 0..<2 {
            app.buttons["View sample score"].tap()
            XCTAssertTrue(app.navigationBars["Score"].waitForExistence(timeout: 5))
            XCTAssertTrue(app.staticTexts["Sample score"].exists)
            XCTAssertTrue(app.staticTexts["Mean GOP"].exists)
            XCTAssertTrue(app.staticTexts["Pitch contour"].exists)
            app.navigationBars["Score"].buttons.firstMatch.tap()
            XCTAssertTrue(app.buttons["Record"].exists)
        }
    }

    func testRecordingWithoutBackendShowsLabeledSample() {
        let app = XCUIApplication()
        app.launchEnvironment["SHADOWTALK_BACKEND_URL"] = "http://127.0.0.1:1"
        app.launch()

        recordAndStop(app)
        XCTAssertTrue(app.navigationBars["Score"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["Server unavailable · Sample score"].exists)
    }

    func testRecordingWithFixtureBackendShowsLiveScore() throws {
        guard let baseURL = ProcessInfo.processInfo.environment["SHADOWTALK_FIXTURE_URL"] else {
            throw XCTSkip("Set SHADOWTALK_FIXTURE_URL to run the live recorder test.")
        }
        let app = XCUIApplication()
        app.launchEnvironment["SHADOWTALK_BACKEND_URL"] = baseURL
        app.launchEnvironment["SHADOWTALK_REFERENCE_ID"] = "score-fixture"
        app.launch()

        recordAndStop(app)
        XCTAssertTrue(app.navigationBars["Score"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["Mean GOP"].exists)
        XCTAssertFalse(app.staticTexts.matching(NSPredicate(format: "label CONTAINS[c] %@", "sample score")).firstMatch.exists)
        app.swipeUp()
        XCTAssertTrue(app.staticTexts["こん"].exists)
    }

    private func recordAndStop(_ app: XCUIApplication) {
        app.buttons["Record"].tap()
        let system = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        let allow = system.alerts.buttons["Allow"]
        if allow.waitForExistence(timeout: 3) { allow.tap() }
        XCTAssertTrue(app.buttons["Stop"].waitForExistence(timeout: 8))
        app.buttons["Stop"].tap()
    }
}
