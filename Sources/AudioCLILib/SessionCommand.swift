import Foundation
import ArgumentParser
import AudioCommon
import SpeechVAD
import SpeakerRegistry

// MARK: - audio session

public struct SessionCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "session",
        abstract: "Process audio files through the speaker registry pipeline",
        subcommands: [ProcessSubcommand.self],
        defaultSubcommand: ProcessSubcommand.self
    )
    public init() {}
}

// MARK: - audio session process <file>

extension SessionCommand {
    public struct ProcessSubcommand: ParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "process",
            abstract: "Diarize an audio file and resolve speakers against the registry"
        )

        @Argument(help: "Audio file to process (WAV, any sample rate)")
        public var audioFile: String

        @Option(name: .long, help: "Cosine similarity threshold for speaker matching (default 0.75)")
        public var threshold: Float = 0.75

        @Option(name: .long, help: "Speaker embedding engine: mlx (default) or coreml")
        public var embeddingEngine: String = "mlx"

        @Option(name: .long, help: "Registry database path (default: ~/Library/Caches/qwen3-speech/speaker-registry.sqlite)")
        public var registryPath: String?

        @Flag(name: .long, help: "Output as JSON")
        public var json: Bool = false

        public init() {}

        public func run() throws {
            try runAsync {
                let url = URL(fileURLWithPath: audioFile)
                print("Loading audio: \(url.lastPathComponent)")
                let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
                let duration = formatDuration(audio.count, sampleRate: 16000)
                print("  Loaded \(audio.count) samples (\(duration)s)")

                guard let embEngine = WeSpeakerEngine(rawValue: embeddingEngine) else {
                    print("Error: unknown embedding engine '\(embeddingEngine)'. Use 'mlx' or 'coreml'.")
                    return
                }

                print("Loading diarization models...")
                let diarizer = try await DiarizationPipeline.fromPretrained(
                    embeddingEngine: embEngine,
                    progressHandler: reportProgress)

                let registryURL = registryPath.map { URL(fileURLWithPath: $0) } ?? .defaultRegistryURL
                print("Opening registry: \(registryURL.path)")
                let registry = try SpeakerRegistry.open(
                    at: registryURL,
                    similarityThreshold: threshold)

                let pipeline = PipelineSession(diarizer: diarizer, registry: registry)

                print("Processing...")
                let start = Date()
                let result = try await pipeline.process(audioURL: url, audio: audio)
                let elapsed = Date().timeIntervalSince(start)

                if json {
                    printJSON(result)
                } else {
                    printText(result, elapsed: elapsed)
                }
            }
        }

        private func printText(_ result: ProcessedSession, elapsed: TimeInterval) {
            if result.segments.isEmpty {
                print("No speech detected.")
            } else {
                for seg in result.segments {
                    let s = String(format: "%.2f", seg.startTime)
                    let e = String(format: "%.2f", seg.endTime)
                    let d = String(format: "%.2f", seg.duration)
                    print("\(seg.speaker.label): [\(s)s - \(e)s] (\(d)s)")
                }
                print("\n--- \(result.numSpeakers) speaker(s) | session id: \(result.session.id ?? -1) ---")
            }
            print("Processed in \(String(format: "%.2f", elapsed))s")
        }

        private func printJSON(_ result: ProcessedSession) {
            var items: [[String: Any]] = []
            for seg in result.segments {
                items.append([
                    "speaker_id": seg.speaker.id ?? -1,
                    "speaker_label": seg.speaker.label,
                    "start": Double(String(format: "%.3f", seg.startTime))!,
                    "end": Double(String(format: "%.3f", seg.endTime))!,
                    "duration": Double(String(format: "%.3f", seg.duration))!,
                ])
            }
            let output: [String: Any] = [
                "session_id": result.session.id ?? -1,
                "num_speakers": result.numSpeakers,
                "segments": items,
            ]
            if let data = try? JSONSerialization.data(withJSONObject: output, options: .prettyPrinted),
               let str = String(data: data, encoding: .utf8) {
                print(str)
            }
        }
    }
}
