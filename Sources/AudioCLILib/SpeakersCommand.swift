import Foundation
import ArgumentParser
import SpeakerRegistry

// MARK: - audio speakers

public struct SpeakersCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "speakers",
        abstract: "Manage speaker identities in the registry",
        subcommands: [
            ListSubcommand.self,
            LabelSubcommand.self,
            MergeSubcommand.self,
            ShowSubcommand.self,
        ]
    )
    public init() {}
}

// MARK: - Shared option

struct RegistryOptions: ParsableArguments {
    @Option(name: .long, help: "Registry database path (default: ~/Library/Caches/qwen3-speech/speaker-registry.sqlite)")
    var registryPath: String?

    var url: URL { registryPath.map { URL(fileURLWithPath: $0) } ?? .defaultRegistryURL }
}

// MARK: - audio speakers list

extension SpeakersCommand {
    public struct ListSubcommand: ParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "list",
            abstract: "List all registered speakers"
        )

        @OptionGroup var registry: RegistryOptions
        @Flag(name: .long, help: "Output as JSON") public var json: Bool = false

        public init() {}

        public func run() throws {
            let reg = try SpeakerRegistry.open(at: registry.url)
            try runAsync {
                let speakers = try await reg.speakers()
                if json {
                    let items = speakers.map { s -> [String: Any] in
                        var d: [String: Any] = ["id": s.id ?? -1, "label": s.label]
                        if let name = s.displayName { d["displayName"] = name }
                        if let notes = s.notes { d["notes"] = notes }
                        return d
                    }
                    if let data = try? JSONSerialization.data(withJSONObject: items, options: .prettyPrinted),
                       let str = String(data: data, encoding: .utf8) {
                        print(str)
                    }
                } else {
                    if speakers.isEmpty {
                        print("No speakers registered.")
                    } else {
                        for s in speakers {
                            let labeled = s.isLabeled ? s.displayName! : "(unlabeled)"
                            print("  id=\(s.id ?? -1)  \(labeled)")
                        }
                        print("\n\(speakers.count) speaker(s)")
                    }
                }
            }
        }
    }
}

// MARK: - audio speakers label <id> <name>

extension SpeakersCommand {
    public struct LabelSubcommand: ParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "label",
            abstract: "Assign a display name to a speaker"
        )

        @Argument(help: "Speaker id") public var speakerId: Int64
        @Argument(help: "Display name to assign") public var displayName: String
        @OptionGroup var registry: RegistryOptions

        public init() {}

        public func run() throws {
            let reg = try SpeakerRegistry.open(at: registry.url)
            try runAsync {
                try await reg.label(speakerId: speakerId, displayName: displayName)
                print("Labeled speaker \(speakerId) as '\(displayName)'")
            }
        }
    }
}

// MARK: - audio speakers merge <src> <dst>

extension SpeakersCommand {
    public struct MergeSubcommand: ParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "merge",
            abstract: "Merge one speaker into another (re-points all segments, blends centroids)"
        )

        @Argument(help: "Source speaker id (will be deleted)") public var src: Int64
        @Argument(help: "Destination speaker id (will be kept)") public var dst: Int64
        @OptionGroup var registry: RegistryOptions

        public init() {}

        public func run() throws {
            let reg = try SpeakerRegistry.open(at: registry.url)
            try runAsync {
                try await reg.merge(src: src, into: dst)
                print("Merged speaker \(src) into \(dst)")
            }
        }
    }
}

// MARK: - audio speakers show <id>

extension SpeakersCommand {
    public struct ShowSubcommand: ParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "show",
            abstract: "Show all segments attributed to a speaker"
        )

        @Argument(help: "Speaker id") public var speakerId: Int64
        @OptionGroup var registry: RegistryOptions
        @Flag(name: .long, help: "Output as JSON") public var json: Bool = false

        public init() {}

        public func run() throws {
            let reg = try SpeakerRegistry.open(at: registry.url)
            try runAsync {
                guard let speaker = try await reg.speaker(id: speakerId) else {
                    print("Speaker \(speakerId) not found.")
                    return
                }
                let segments = try await reg.segments(for: speakerId)

                if json {
                    let items = segments.map { seg -> [String: Any] in
                        var d: [String: Any] = [
                            "id": seg.id ?? -1,
                            "session_id": seg.sessionId,
                            "start": seg.startTime,
                            "end": seg.endTime,
                            "duration": seg.duration,
                        ]
                        if let t = seg.transcriptText { d["transcript"] = t }
                        return d
                    }
                    let output: [String: Any] = [
                        "speaker": ["id": speaker.id ?? -1, "label": speaker.label],
                        "segments": items,
                    ]
                    if let data = try? JSONSerialization.data(withJSONObject: output, options: .prettyPrinted),
                       let str = String(data: data, encoding: .utf8) {
                        print(str)
                    }
                } else {
                    print("Speaker: \(speaker.label) (id=\(speakerId))")
                    if segments.isEmpty {
                        print("  No segments.")
                    } else {
                        for seg in segments {
                            let s = String(format: "%.2f", seg.startTime)
                            let e = String(format: "%.2f", seg.endTime)
                            let d = String(format: "%.2f", seg.duration)
                            let tx = seg.transcriptText.map { " \"\($0)\"" } ?? ""
                            print("  session=\(seg.sessionId) [\(s)s - \(e)s] (\(d)s)\(tx)")
                        }
                        let total = segments.reduce(0.0) { $0 + $1.duration }
                        print("\n\(segments.count) segment(s), \(String(format: "%.2f", total))s total")
                    }
                }
            }
        }
    }
}
