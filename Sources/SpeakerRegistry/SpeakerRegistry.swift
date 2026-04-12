import Foundation
import AudioCommon
import os

// MARK: - Registry Store

private struct RegistryStore: Codable {
    var speakers: [Speaker] = []
    var centroids: [SpeakerCentroid] = []
    var nextId: Int64 = 1
}

// MARK: - SpeakerRegistry

/// Persistent speaker identity store backed by a JSON file.
///
/// Resolves raw WeSpeaker embeddings to named (or placeholder) speaker identities
/// using cosine-similarity matching against running per-speaker centroids.
/// All state is kept in memory and flushed atomically on every mutation.
/// Thread-safe via Swift actor isolation.
public actor SpeakerRegistry {

    private var store: RegistryStore
    private let url: URL
    public let similarityThreshold: Float

    // MARK: - Factory

    /// Open (or create) a registry at `url`.
    ///
    /// - Parameters:
    ///   - url: Path to the JSON file. Defaults to `~/Library/Caches/qwen3-speech/speaker-registry.json`.
    ///   - similarityThreshold: Minimum cosine similarity to count as a match (default 0.75).
    public static func open(
        at url: URL = .defaultRegistryURL,
        similarityThreshold: Float = 0.75  // empirically optimal on VoxConverse (sweep 2026-04-12)
    ) throws -> SpeakerRegistry {
        let store: RegistryStore
        if FileManager.default.fileExists(atPath: url.path) {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            store = try decoder.decode(RegistryStore.self, from: Data(contentsOf: url))
        } else {
            store = RegistryStore()
        }
        return SpeakerRegistry(store: store, url: url, threshold: similarityThreshold)
    }

    private init(store: RegistryStore, url: URL, threshold: Float) {
        self.store = store
        self.url = url
        self.similarityThreshold = threshold
    }

    // MARK: - Resolve

    /// Resolve a fresh embedding to a speaker identity.
    ///
    /// If `qualityScore` (segment duration in seconds) is below 2 s the segment is too short to
    /// enroll or update a centroid; a match-only lookup is performed and a new placeholder is
    /// created only if nothing matches.
    ///
    /// - Parameters:
    ///   - embedding: 256-dim L2-normalised WeSpeaker embedding.
    ///   - qualityScore: Segment duration in seconds.
    /// - Returns: The matched or newly created `Speaker`.
    public func resolve(embedding: [Float], qualityScore: Double, threshold: Float? = nil) throws -> Speaker {
        let isHighQuality = qualityScore >= 2.0
        let effectiveThreshold = threshold ?? similarityThreshold

        if let (speaker, similarity) = bestMatch(embedding: embedding, threshold: effectiveThreshold) {
            AudioLog.pipeline.debug("Matched \(speaker.label) (cosine=\(similarity, format: .fixed(precision: 3)))")
            if isHighQuality {
                updateCentroid(speakerId: speaker.id!, with: embedding)
                try save()
            }
            return speaker
        } else {
            let speaker = mintPlaceholder()
            if isHighQuality {
                store.centroids.append(SpeakerCentroid(speakerId: speaker.id!, centroid: normalizeL2(embedding)))
                AudioLog.pipeline.debug("Enrolled new \(speaker.label)")
            } else {
                AudioLog.pipeline.debug("Created placeholder \(speaker.label) (short segment, no match)")
            }
            try save()
            return speaker
        }
    }

    // MARK: - Label Management

    /// Assign a human-readable name to a speaker.
    public func label(speakerId: Int64, displayName: String) throws {
        guard let idx = store.speakers.firstIndex(where: { $0.id == speakerId }) else { return }
        store.speakers[idx].displayName = displayName
        try save()
        AudioLog.pipeline.info("Labeled speaker \(speakerId) as '\(displayName)'")
    }

    /// Merge `src` into `dst`: blends centroids, removes `src`.
    public func merge(src: Int64, into dst: Int64) throws {
        if let srcCentroid = store.centroids.first(where: { $0.speakerId == src }),
           let dstIdx = store.centroids.firstIndex(where: { $0.speakerId == dst }) {
            let dstCentroid = store.centroids[dstIdx]
            let blended = blendCentroids(
                centroidA: dstCentroid.centroid, countA: dstCentroid.sampleCount,
                centroidB: srcCentroid.centroid, countB: srcCentroid.sampleCount)
            store.centroids[dstIdx].centroid = blended
            store.centroids[dstIdx].sampleCount = dstCentroid.sampleCount + srcCentroid.sampleCount
        }
        store.centroids.removeAll { $0.speakerId == src }
        store.speakers.removeAll { $0.id == src }
        try save()
        AudioLog.pipeline.info("Merged speaker \(src) into \(dst)")
    }

    // MARK: - Queries

    public func speakers() -> [Speaker] { store.speakers }

    public func speaker(id: Int64) -> Speaker? {
        store.speakers.first(where: { $0.id == id })
    }

    // MARK: - Mutations

    public func updateNotes(speakerId: Int64, notes: String) throws {
        guard let idx = store.speakers.firstIndex(where: { $0.id == speakerId }) else { return }
        store.speakers[idx].notes = notes
        try save()
    }

    public func deleteSpeaker(id: Int64) throws {
        store.speakers.removeAll { $0.id == id }
        store.centroids.removeAll { $0.speakerId == id }
        try save()
        AudioLog.pipeline.info("Deleted speaker \(id)")
    }

    /// Wipe all speakers, centroids, and reset the ID counter to 1.
    public func reset() throws {
        store = RegistryStore()
        try save()
        AudioLog.pipeline.info("Registry reset: all speakers and centroids cleared")
    }

    // MARK: - Private Helpers

    private func bestMatch(embedding: [Float], threshold: Float) -> (Speaker, Float)? {
        var best: (Int64, Float)?
        for c in store.centroids {
            let sim = cosineSimilarity(embedding, c.centroid)
            if sim > threshold, sim > (best?.1 ?? 0) {
                best = (c.speakerId, sim)
            }
        }
        guard let (speakerId, sim) = best else { return nil }
        return store.speakers.first(where: { $0.id == speakerId }).map { ($0, sim) }
    }

    private func mintPlaceholder() -> Speaker {
        let speaker = Speaker(id: store.nextId)
        store.nextId += 1
        store.speakers.append(speaker)
        return speaker
    }

    private func updateCentroid(speakerId: Int64, with embedding: [Float]) {
        guard let idx = store.centroids.firstIndex(where: { $0.speakerId == speakerId }) else { return }
        let updated = incrementalCentroid(
            old: store.centroids[idx].centroid,
            sampleCount: store.centroids[idx].sampleCount,
            new: embedding)
        store.centroids[idx].centroid = updated
        store.centroids[idx].sampleCount += 1
    }

    private func save() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(store).write(to: url, options: .atomic)
    }
}
