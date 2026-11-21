import Charts
import SwiftUI

struct ScoreView: View {
    let report: ScoreReport

    var body: some View {
        List {
            summary

            Section("Pitch contour") {
                if let contour = report.payload.pitchContour, contour.isComparable {
                    PitchContourChart(contour: contour)
                } else {
                    Text(report.payload.missingPitchMessage)
                        .foregroundStyle(.secondary)
                }
            }

            Section("Pronunciation") {
                if report.payload.scores.isEmpty {
                    Text(report.payload.alignment == "unavailable"
                         ? "The server’s speech aligner is unavailable."
                         : "No pronunciation units were scored for this recording.")
                        .foregroundStyle(.secondary)
                }
                ForEach(Array(report.payload.scores.enumerated()), id: \.offset) { _, score in
                    PhonemeScoreRow(score: score, metric: report.payload.pronunciationMetric)
                        .listRowBackground(score.flagged ? Color.orange.opacity(0.14) : nil)
                }
            }
        }
        .navigationTitle("Score")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var summary: some View {
        Section("Summary") {
            switch report.source {
            case .recording: EmptyView()
            case .sample: Label("Sample score", systemImage: "doc.text")
            case .offlineSample: Label("Server unavailable · Sample score", systemImage: "wifi.slash")
            }
            LabeledContent("Mean \(report.payload.pronunciationMetric)") {
                if let mean = report.payload.meanGOP {
                    Text(mean, format: .percent.precision(.fractionLength(0))).monospacedDigit()
                } else {
                    Text("Not scored")
                }
            }
            LabeledContent("Flagged", value: report.payload.flaggedCount.formatted())
            if report.payload.gopSource == "mixed" {
                Text("Some units use alignment confidence in place of GOP.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            if report.payload.flaggedCount > 0 {
                Text("Flagged units have low alignment confidence and do not count against you.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

private struct PhonemeScoreRow: View {
    let score: PhonemeScore
    let metric: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            LabeledContent {
                if score.flagged {
                    Label("Low confidence", systemImage: "flag.fill")
                        .foregroundStyle(.secondary)
                } else if let delta = score.displayPitchDelta {
                    Text("Pitch \(delta, format: .number.sign(strategy: .always()).precision(.fractionLength(1))) st")
                        .monospacedDigit()
                } else {
                    Text("Pitch unavailable")
                }
            } label: {
                Text(score.phoneme).font(.headline)
            }
            if let gop = score.displayGOP {
                ProgressView(value: gop, total: 1) {
                    LabeledContent(metric, value: gop.formatted(.percent.precision(.fractionLength(0))))
                        .font(.caption)
                }
            } else {
                Text("Not scored").font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 3)
    }
}

private struct PitchContourChart: View {
    let contour: PitchContour

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Chart {
                trace("Learner", samples: contour.learner, style: StrokeStyle(lineWidth: 2))
                trace("Reference", samples: contour.reference, style: StrokeStyle(lineWidth: 2, dash: [5, 3]))
            }
            .chartForegroundStyleScale(["Learner": Color.indigo, "Reference": Color.orange])
            .chartXAxisLabel("Time (s)")
            .chartYAxisLabel("Relative pitch (st)")
            .frame(minHeight: 175)

            Text("Pitch is in semitones relative to each speaker’s median. Gaps are unvoiced.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    private struct Point {
        let frame: Int
        let segment: Int
        let pitch: Double
    }

    private func points(_ samples: [Double?]) -> [Point] {
        var segment = 0
        return samples.enumerated().compactMap { frame, pitch in
            guard let pitch else {
                segment += 1
                return nil
            }
            return Point(frame: frame, segment: segment, pitch: pitch)
        }
    }

    @ChartContentBuilder
    private func trace(_ name: String, samples: [Double?], style: StrokeStyle) -> some ChartContent {
        ForEach(points(samples), id: \.frame) { point in
            let x = PlottableValue.value("Time", Double(point.frame) * contour.frameSeconds)
            let y = PlottableValue.value("Relative pitch", point.pitch)
            let before = point.frame > 0 ? samples[point.frame - 1] : nil
            let after = point.frame + 1 < samples.count ? samples[point.frame + 1] : nil
            if before == nil && after == nil {
                PointMark(x: x, y: y)
                    .foregroundStyle(by: .value("Speaker", name))
            } else {
                LineMark(x: x, y: y, series: .value("Segment", "\(name)-\(point.segment)"))
                    .foregroundStyle(by: .value("Speaker", name))
                    .lineStyle(style)
            }
        }
    }
}
