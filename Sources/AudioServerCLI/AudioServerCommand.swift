import Foundation
import ArgumentParser
import AudioServer

@main
struct AudioServerCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "audio-server",
        abstract: "HTTP API server for speech models on Apple Silicon"
    )

    @Option(name: .long, help: "Host to bind (default: 127.0.0.1)")
    var host: String = "127.0.0.1"

    @Option(name: .long, help: "Port to bind (default: 8080)")
    var port: Int = 8080

    @Flag(name: .long, help: "Load all models on startup (slower start, faster first request)")
    var preload: Bool = false

    @Flag(name: .long, help: "Log all incoming HTTP requests (method, path, status)")
    var logRequests: Bool = false

    func run() async throws {
        let server = AudioServer(host: host, port: port, logRequests: logRequests)

        if preload {
            print("Preloading models...")
            try await server.preloadModels()
            print("All models loaded.")
        }

        print("Starting server on http://\(host):\(port)")
        print("Endpoints:")
        print("  POST /transcribe                      - Speech-to-text (WAV body or JSON with audio_base64)")
        print("  POST /speak                           - Text-to-speech (JSON: {text, engine?, language?})")
        print("  POST /respond                         - Speech-to-speech (WAV body)")
        print("  POST /enhance                         - Speech enhancement (WAV body)")
        print("  GET  /health                          - Health check")
        print("  WS   /v1/realtime                     - OpenAI Realtime API (JSON events, base64 PCM16 audio)")
        print("  POST /registry/sessions               - Diarize WAV + resolve speakers")
        print("  GET  /registry/sessions               - List sessions")
        print("  GET  /registry/sessions/:id           - Session detail")
        print("  GET  /registry/sessions/:id/segments  - Segments for a session")
        print("  GET  /registry/speakers               - List speakers")
        print("  GET  /registry/speakers/:id           - Get speaker")
        print("  PATCH /registry/speakers/:id          - Label or update notes")
        print("  POST /registry/speakers/merge         - Merge speakers ({src, dst})")
        print("  DELETE /registry/speakers/:id         - Delete speaker")
        print("  GET  /registry/speakers/:id/segments  - Segments for a speaker")

        try await server.run()
    }
}
