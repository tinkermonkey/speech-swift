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
            ResetSubcommand.self,
        ]
    )
    public init() {}
}

// MARK: - Shared option

struct RegistryOptions: ParsableArguments {
    @Option(name: .long, help: "Registry file path (default: ~/Library/Caches/qwen3-speech/speaker-registry.json)")
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
                let speakers = await reg.speakers()
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
            abstract: "Merge one speaker into another (blends centroids, removes source)"
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

// MARK: - audio speakers reset

extension SpeakersCommand {
    public struct ResetSubcommand: ParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "reset",
            abstract: "Wipe all speakers and centroids, resetting the registry to empty"
        )

        @OptionGroup var registry: RegistryOptions
        @Flag(name: .long, help: "Skip confirmation prompt") public var force: Bool = false

        public init() {}

        public func run() throws {
            if !force {
                print("This will permanently delete all speakers and centroids in the registry.")
                print("Type 'yes' to confirm: ", terminator: "")
                guard readLine()?.lowercased() == "yes" else {
                    print("Aborted.")
                    return
                }
            }
            let reg = try SpeakerRegistry.open(at: registry.url)
            try runAsync {
                try await reg.reset()
                print("Registry reset: all speakers and centroids cleared.")
            }
        }
    }
}

// MARK: - audio speakers show <id>

extension SpeakersCommand {
    public struct ShowSubcommand: ParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "show",
            abstract: "Show details for a registered speaker"
        )

        @Argument(help: "Speaker id") public var speakerId: Int64
        @OptionGroup var registry: RegistryOptions
        @Flag(name: .long, help: "Output as JSON") public var json: Bool = false

        public init() {}

        public func run() throws {
            let reg = try SpeakerRegistry.open(at: registry.url)
            try runAsync {
                guard let speaker = await reg.speaker(id: speakerId) else {
                    print("Speaker \(speakerId) not found.")
                    return
                }
                if json {
                    var d: [String: Any] = [
                        "id": speaker.id ?? -1,
                        "label": speaker.label,
                        "is_labeled": speaker.isLabeled,
                        "created_at": ISO8601DateFormatter().string(from: speaker.createdAt),
                    ]
                    if let name = speaker.displayName { d["display_name"] = name }
                    if let notes = speaker.notes { d["notes"] = notes }
                    if let data = try? JSONSerialization.data(withJSONObject: d, options: .prettyPrinted),
                       let str = String(data: data, encoding: .utf8) {
                        print(str)
                    }
                } else {
                    print("id=\(speaker.id ?? -1)  \(speaker.label)")
                    print("created: \(ISO8601DateFormatter().string(from: speaker.createdAt))")
                    if let notes = speaker.notes { print("notes:   \(notes)") }
                }
            }
        }
    }
}
