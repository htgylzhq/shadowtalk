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

        app.buttons["Record"].tap()
        let system = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        let allow = system.alerts.buttons["Allow"]
        if allow.waitForExistence(timeout: 3) { allow.tap() }
        XCTAssertTrue(app.buttons["Stop"].waitForExistence(timeout: 8))
        app.buttons["Stop"].tap()

        XCTAssertTrue(app.navigationBars["Score"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["Server unavailable · Sample score"].exists)
    }
}
