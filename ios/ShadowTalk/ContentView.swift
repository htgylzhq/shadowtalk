import SwiftUI

struct ContentView: View {
    @StateObject private var recorder = RecorderViewModel()

    var body: some View {
        VStack(spacing: 24) {
            Text("ShadowTalk")
                .font(.largeTitle.bold())

            // Week-1 milestone: one sentence, end to end.
            Text("Listen, then shadow the sentence.")
                .foregroundStyle(.secondary)

            Button(recorder.isRecording ? "Stop" : "Record") {
                recorder.isRecording ? recorder.stop() : recorder.start()
            }
            .buttonStyle(.borderedProminent)
            .tint(recorder.isRecording ? .red : .accentColor)

            if let score = recorder.lastScore {
                Text("Score: \(score, specifier: "%.2f")")
                    .font(.headline)
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
