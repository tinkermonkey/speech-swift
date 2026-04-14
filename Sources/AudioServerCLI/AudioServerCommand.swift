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

    @Option(
        name: .long,
        help: "Comma-separated models to load on startup: asr, diarizer, tts, cosyvoice, personaplex, enhancer, all. Example: --preload asr,diarizer"
    )
    var preload: String?

    @Option(
        name: .long,
        help: "Run a tiny inference every N seconds to keep GPU kernels hot and prevent eviction latency (default: 10). Set to 0 to disable."
    )
    var keepAlive: Double = 10.0

    @Option(
        name: .long,
        help: "Maximum number of /registry/sessions inference calls to run concurrently (default: 1). Values > 1 run multiple inferences in parallel but each will compete for GPU — only increase if you have spare VRAM and have profiled the benefit."
    )
    var concurrency: Int = 1

    @Flag(name: .long, help: "Log all incoming HTTP requests (method, path, status)")
    var logRequests: Bool = false

    func run() async throws {
        let server = AudioServer(host: host, port: port, logRequests: logRequests, concurrency: concurrency)

        if let preload {
            let models = Set(preload.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) })
            let unknown = models.subtracting(["all", "asr", "diarizer", "tts", "cosyvoice", "personaplex", "enhancer"])
            if !unknown.isEmpty {
                print("Warning: unknown model(s) ignored: \(unknown.sorted().joined(separator: ", "))")
            }
            print("Preloading: \(models.sorted().joined(separator: ", "))")
            try await server.preloadModels(models)
            print("Models ready.")
        }

        if keepAlive > 0 {
            server.startKeepAlive(intervalSeconds: keepAlive)
            print("GPU keep-alive enabled (every \(Int(keepAlive))s)")
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
