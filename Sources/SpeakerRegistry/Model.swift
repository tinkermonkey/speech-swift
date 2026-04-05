import Foundation

// MARK: - Default Storage URL

extension URL {
    public static var defaultRegistryURL: URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        let dir = caches.appendingPathComponent("qwen3-speech", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("speaker-registry.json")
    }
}

// MARK: - Speaker

/// A registered speaker identity. `displayName` is nil until the user labels the placeholder.
public struct Speaker: Codable, Sendable {
    public var id: Int64?
    public var displayName: String?
    public var createdAt: Date
    public var notes: String?

    public var isLabeled: Bool { displayName != nil }
    public var label: String { displayName ?? "Speaker_\(id.map(String.init) ?? "?")" }

    public init(id: Int64? = nil, displayName: String? = nil, createdAt: Date = .now, notes: String? = nil) {
        self.id = id
        self.displayName = displayName
        self.createdAt = createdAt
        self.notes = notes
    }
}

// MARK: - SpeakerCentroid

/// Running mean of all enrolled embeddings for a speaker, normalised to the unit sphere.
public struct SpeakerCentroid: Codable, Sendable {
    public var speakerId: Int64
    public var centroid: [Float]
    public var sampleCount: Int

    public init(speakerId: Int64, centroid: [Float], sampleCount: Int = 1) {
        self.speakerId = speakerId
        self.centroid = centroid
        self.sampleCount = sampleCount
    }
}
