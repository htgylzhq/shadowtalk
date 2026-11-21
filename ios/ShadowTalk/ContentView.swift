import SwiftUI

struct ContentView: View {
    @StateObject private var recorder = RecorderViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                Text("ShadowTalk")
                    .font(.largeTitle.bold())

                Text("Listen, then shadow the sentence.")
                    .foregroundStyle(.secondary)

                switch recorder.phase {
                case .idle:
                    Button("Record", systemImage: "mic.fill") {
                        Task { await recorder.start() }
                    }
                    .buttonStyle(.borderedProminent)

                    Button("View sample score") { recorder.showSample() }
                case .recording, .recordingOffline:
                    Button("Stop", systemImage: "stop.fill", action: recorder.stop)
                        .buttonStyle(.borderedProminent)
                        .tint(.red)
                    if recorder.phase == .recordingOffline {
                        Text("Server unavailable. Stop to view a sample score.")
                            .foregroundStyle(.secondary)
                    }
                case .preparing:
                    ProgressView("Preparing microphone…")
                case .scoring:
                    ProgressView("Scoring recording…")
                }

                if let error = recorder.error {
                    Label(error, systemImage: "exclamationmark.circle")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            .multilineTextAlignment(.center)
            .padding()
            .navigationDestination(isPresented: Binding(
                get: { recorder.report != nil },
                set: { if !$0 { recorder.report = nil } }
            )) {
                if let report = recorder.report {
                    ScoreView(report: report)
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
