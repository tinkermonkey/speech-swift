import Foundation
import GRDB

// MARK: - Data Helpers

extension Data {
    func toFloats() -> [Float] {
        withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static func fromFloats(_ floats: [Float]) -> Data {
        floats.withUnsafeBytes { Data($0) }
    }
}

// MARK: - Default Storage URL

extension URL {
    public static var defaultRegistryURL: URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        let dir = caches.appendingPathComponent("qwen3-speech", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("speaker-registry.sqlite")
    }
}

// MARK: - Speaker

/// A registered speaker identity. `displayName` is nil until the user labels the placeholder.
public struct Speaker: Codable, FetchableRecord, PersistableRecord, Sendable {
    public var id: Int64?
    public var displayName: String?
    public var createdAt: Date
    public var notes: String?

    public static let databaseTableName = "speaker"

    public var isLabeled: Bool { displayName != nil }
    public var label: String { displayName ?? "Speaker_\(id.map(String.init) ?? "?")" }

    public init(displayName: String? = nil, createdAt: Date = .now, notes: String? = nil) {
        self.displayName = displayName
        self.createdAt = createdAt
        self.notes = notes
    }
}

// MARK: - SpeakerSession

/// A single processed audio file run through the diarization pipeline.
public struct SpeakerSession: Codable, FetchableRecord, PersistableRecord, Sendable {
    public var id: Int64?
    public var audioPath: String
    public var recordedAt: Date
    public var processedAt: Date?
    public var durationSeconds: Double

    public static let databaseTableName = "speaker_session"

    public init(audioPath: String, recordedAt: Date = .now, durationSeconds: Double) {
        self.audioPath = audioPath
        self.recordedAt = recordedAt
        self.durationSeconds = durationSeconds
    }
}

// MARK: - SpeakerSegment

/// A time-stamped segment attributed to a specific speaker within a session.
public struct SpeakerSegment: Codable, FetchableRecord, PersistableRecord, Sendable {
    public var id: Int64?
    public var sessionId: Int64
    public var speakerId: Int64
    public var startTime: Double
    public var endTime: Double
    public var transcriptText: String?

    public static let databaseTableName = "speaker_segment"

    public var duration: Double { endTime - startTime }

    public init(
        sessionId: Int64,
        speakerId: Int64,
        startTime: Double,
        endTime: Double,
        transcriptText: String? = nil
    ) {
        self.sessionId = sessionId
        self.speakerId = speakerId
        self.startTime = startTime
        self.endTime = endTime
        self.transcriptText = transcriptText
    }
}

// MARK: - SpeakerEmbedding

/// Raw 256-dim WeSpeaker embedding for a segment, kept for future centroid recomputation.
public struct SpeakerEmbedding: Codable, FetchableRecord, PersistableRecord, Sendable {
    public var id: Int64?
    public var speakerId: Int64
    public var sessionId: Int64
    public var segmentStart: Double
    public var segmentEnd: Double
    /// Segment duration in seconds — used as quality proxy; segments < 2s skip centroid update.
    public var qualityScore: Double
    /// 256 × Float32, little-endian bytes.
    public var embedding: Data

    public static let databaseTableName = "speaker_embedding"

    public init(
        speakerId: Int64,
        sessionId: Int64,
        segmentStart: Double,
        segmentEnd: Double,
        qualityScore: Double,
        embedding: [Float]
    ) {
        self.speakerId = speakerId
        self.sessionId = sessionId
        self.segmentStart = segmentStart
        self.segmentEnd = segmentEnd
        self.qualityScore = qualityScore
        self.embedding = Data.fromFloats(embedding)
    }
}

// MARK: - SpeakerCentroid

/// Running mean of all enrolled embeddings for a speaker, normalised to the unit sphere.
public struct SpeakerCentroid: Codable, FetchableRecord, PersistableRecord, Sendable {
    public var speakerId: Int64
    /// 256 × Float32, little-endian bytes.
    public var centroid: Data
    public var sampleCount: Int

    public static let databaseTableName = "speaker_centroid"

    public init(speakerId: Int64, centroid: [Float], sampleCount: Int = 1) {
        self.speakerId = speakerId
        self.centroid = Data.fromFloats(centroid)
        self.sampleCount = sampleCount
    }
}
