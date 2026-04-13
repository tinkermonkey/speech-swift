import Foundation
import AudioCommon
import os

// MARK: - In-Memory Store

private struct RegistryStore {
    var speakers: [Speaker] = []
    var centroids: [SpeakerCentroid] = []
    var nextId: Int64 = 1
}

// MARK: - JSON Metadata (speakers + nextId only)

private struct MetadataStore: Codable {
    var speakers: [Speaker] = []
    var nextId: Int64 = 1
}

// MARK: - Errors

public enum RegistryError: Error, CustomStringConvertible {
    case speakerNotFound(Int64)

    public var description: String {
        switch self {
        case .speakerNotFound(let id): return "Speaker \(id) not found in registry"
        }
    }
}

// MARK: - SpeakerRegistry

/// Persistent speaker identity store.
///
/// Speaker metadata (id, displayName, notes, createdAt, nextId) is stored in a JSON file.
/// Each speaker's embedding centroid is stored as a compact binary file:
///   `centroids/centroid-{id}.bin` → 4-byte UInt32 sampleCount + Float32[256] (1028 bytes)
///
/// This keeps JSON human-readable and avoids Float encoding issues (NaN/Inf) in the registry.
/// Thread-safe via Swift actor isolation.
public actor SpeakerRegistry {

    private var store: RegistryStore
    private let metadataURL: URL
    private let centroidsDir: URL
    public let similarityThreshold: Float

    // MARK: - Factory

    /// Open (or create) a registry at `url`.
    ///
    /// - Parameters:
    ///   - url: Path to the JSON metadata file. Defaults to
    ///     `~/Library/Caches/qwen3-speech/speaker-registry.json`.
    ///     Centroid binaries are stored in a `centroids/` subdirectory alongside it.
    ///   - similarityThreshold: Minimum cosine similarity to count as a match (default 0.75).
    public static func open(
        at url: URL = .defaultRegistryURL,
        similarityThreshold: Float = 0.75  // empirically optimal on VoxConverse (sweep 2026-04-12)
    ) throws -> SpeakerRegistry {
        let centroidsDir = url.deletingLastPathComponent().appendingPathComponent("centroids")
        try FileManager.default.createDirectory(at: centroidsDir, withIntermediateDirectories: true)

        // Load metadata JSON. Old files that embed centroids decode cleanly — MetadataStore
        // ignores unknown keys, so legacy "centroids" arrays are silently dropped.
        var speakers: [Speaker] = []
        var nextId: Int64 = 1
        if FileManager.default.fileExists(atPath: url.path) {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let metadata = try decoder.decode(MetadataStore.self, from: Data(contentsOf: url))
            speakers = metadata.speakers
            nextId = metadata.nextId
        }

        // Load centroid binaries for each known speaker.
        var centroids: [SpeakerCentroid] = []
        for speaker in speakers {
            guard let id = speaker.id else {
                AudioLog.pipeline.warning("Registry: skipping speaker with nil id (data may be corrupt)")
                continue
            }
            let binURL = centroidsDir.appendingPathComponent("centroid-\(id).bin")
            do {
                let centroid = try loadCentroid(from: binURL, speakerId: id)
                centroids.append(centroid)
            } catch {
                AudioLog.pipeline.warning("Registry: failed to load centroid for speaker \(id), speaker will not match: \(error)")
            }
        }

        AudioLog.pipeline.info(
            "Registry opened: \(speakers.count) speaker(s), \(centroids.count) centroid(s)")

        let store = RegistryStore(speakers: speakers, centroids: centroids, nextId: nextId)
        return SpeakerRegistry(
            store: store,
            metadataURL: url,
            centroidsDir: centroidsDir,
            threshold: similarityThreshold)
    }

    private init(
        store: RegistryStore,
        metadataURL: URL,
        centroidsDir: URL,
        threshold: Float
    ) {
        self.store = store
        self.metadataURL = metadataURL
        self.centroidsDir = centroidsDir
        self.similarityThreshold = threshold
    }

    // MARK: - Resolve / Match

    /// Match an embedding against registered speakers without creating new entries.
    ///
    /// Use this on the fast path (short clips) where you want to identify a known
    /// speaker but must not pollute the registry with unconfirmed identities.
    ///
    /// - Returns: The best-matching `Speaker`, or `nil` if no speaker meets the threshold.
    public func match(embedding: [Float], threshold: Float? = nil) -> Speaker? {
        let effectiveThreshold = threshold ?? similarityThreshold
        guard let (speaker, similarity) = bestMatch(embedding: embedding, threshold: effectiveThreshold) else {
            return nil
        }
        AudioLog.pipeline.debug("Matched \(speaker.label) (cosine=\(similarity, format: .fixed(precision: 3)))")
        return speaker
    }

    /// Resolve a fresh embedding to a speaker identity, enrolling a new speaker if needed.
    ///
    /// Use this on the slow path (long accumulated clips) where enrollment is intentional.
    /// A new speaker is created only when the segment is high-quality (≥ 2 s of speech) and
    /// no existing speaker meets the similarity threshold. Low-quality segments that don't
    /// match any registered speaker return `nil` rather than creating a placeholder.
    ///
    /// - Parameters:
    ///   - embedding: 256-dim L2-normalised WeSpeaker embedding.
    ///   - qualityScore: Longest segment duration in seconds for this speaker in the clip.
    /// - Returns: The matched or newly enrolled `Speaker`, or `nil` if quality is too low to enroll.
    public func resolve(embedding: [Float], qualityScore: Double, threshold: Float? = nil) throws -> Speaker? {
        let isHighQuality = qualityScore >= 2.0
        let effectiveThreshold = threshold ?? similarityThreshold

        if let (speaker, similarity) = bestMatch(embedding: embedding, threshold: effectiveThreshold) {
            AudioLog.pipeline.debug("Matched \(speaker.label) (cosine=\(similarity, format: .fixed(precision: 3)))")
            if isHighQuality {
                guard embedding.allSatisfy({ $0.isFinite }) else {
                    AudioLog.pipeline.warning("Skipped centroid update for \(speaker.label): embedding contains NaN/Inf")
                    return speaker
                }
                updateCentroid(speakerId: speaker.id!, with: embedding)
                if let updated = store.centroids.first(where: { $0.speakerId == speaker.id! }) {
                    try saveCentroid(updated)
                } else {
                    AudioLog.pipeline.warning("Registry: centroid missing after update for speaker \(speaker.id!); not saved")
                }
            }
            return speaker
        } else if isHighQuality {
            // High-quality segment with no match → enroll as new speaker.
            let speaker = mintPlaceholder()
            guard embedding.allSatisfy({ $0.isFinite }) else {
                AudioLog.pipeline.warning("Skipped centroid enrolment for \(speaker.label): embedding contains NaN/Inf")
                try saveMetadata()
                return speaker
            }
            let centroid = SpeakerCentroid(speakerId: speaker.id!, centroid: normalizeL2(embedding))
            store.centroids.append(centroid)
            try saveCentroid(centroid)  // centroid first — if this fails, speaker is not persisted
            try saveMetadata()
            AudioLog.pipeline.debug("Enrolled new \(speaker.label)")
            return speaker
        } else {
            // Low-quality segment with no match → do not create a placeholder.
            AudioLog.pipeline.debug("No match for low-quality segment (\(String(format: "%.1f", qualityScore))s), skipping enrollment")
            return nil
        }
    }

    // MARK: - Label Management

    /// Assign a human-readable name to a speaker.
    public func label(speakerId: Int64, displayName: String) throws {
        guard let idx = store.speakers.firstIndex(where: { $0.id == speakerId }) else {
            throw RegistryError.speakerNotFound(speakerId)
        }
        store.speakers[idx].displayName = displayName
        try saveMetadata()
        AudioLog.pipeline.info("Labeled speaker \(speakerId) as '\(displayName)'")
    }

    /// Merge `src` into `dst`: blends centroids, removes `src`.
    public func merge(src: Int64, into dst: Int64) throws {
        guard store.speakers.contains(where: { $0.id == src }) else {
            throw RegistryError.speakerNotFound(src)
        }
        guard store.speakers.contains(where: { $0.id == dst }) else {
            throw RegistryError.speakerNotFound(dst)
        }
        if let srcCentroid = store.centroids.first(where: { $0.speakerId == src }),
           let dstIdx = store.centroids.firstIndex(where: { $0.speakerId == dst }) {
            let dstCentroid = store.centroids[dstIdx]
            let blended = blendCentroids(
                centroidA: dstCentroid.centroid, countA: dstCentroid.sampleCount,
                centroidB: srcCentroid.centroid, countB: srcCentroid.sampleCount)
            store.centroids[dstIdx].centroid = blended
            store.centroids[dstIdx].sampleCount = dstCentroid.sampleCount + srcCentroid.sampleCount
            try saveCentroid(store.centroids[dstIdx])
        }
        store.centroids.removeAll { $0.speakerId == src }
        store.speakers.removeAll { $0.id == src }
        deleteCentroidFile(speakerId: src)
        try saveMetadata()
        AudioLog.pipeline.info("Merged speaker \(src) into \(dst)")
    }

    // MARK: - Queries

    public func speakers() -> [Speaker] { store.speakers }

    public func speaker(id: Int64) -> Speaker? {
        store.speakers.first(where: { $0.id == id })
    }

    // MARK: - Mutations

    public func updateNotes(speakerId: Int64, notes: String) throws {
        guard let idx = store.speakers.firstIndex(where: { $0.id == speakerId }) else {
            throw RegistryError.speakerNotFound(speakerId)
        }
        store.speakers[idx].notes = notes
        try saveMetadata()
    }

    public func deleteSpeaker(id: Int64) throws {
        store.speakers.removeAll { $0.id == id }
        store.centroids.removeAll { $0.speakerId == id }
        deleteCentroidFile(speakerId: id)
        try saveMetadata()
        AudioLog.pipeline.info("Deleted speaker \(id)")
    }

    /// Wipe all speakers and centroids, reset the ID counter to 1.
    public func reset() throws {
        for speaker in store.speakers {
            if let id = speaker.id { deleteCentroidFile(speakerId: id) }
        }
        store = RegistryStore()
        try saveMetadata()
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
        if let speaker = store.speakers.first(where: { $0.id == speakerId }) {
            return (speaker, sim)
        }
        AudioLog.pipeline.warning("Registry: centroid references speaker \(speakerId) not found in store — data may be inconsistent")
        return nil
    }

    private func mintPlaceholder() -> Speaker {
        let speaker = Speaker(id: store.nextId)
        store.nextId += 1
        store.speakers.append(speaker)
        return speaker
    }

    private func updateCentroid(speakerId: Int64, with embedding: [Float]) {
        guard let idx = store.centroids.firstIndex(where: { $0.speakerId == speakerId }) else {
            AudioLog.pipeline.warning("Registry: centroid not found in memory for speaker \(speakerId); update skipped")
            return
        }
        let updated = incrementalCentroid(
            old: store.centroids[idx].centroid,
            sampleCount: store.centroids[idx].sampleCount,
            new: embedding)
        store.centroids[idx].centroid = updated
        store.centroids[idx].sampleCount += 1
    }

    // MARK: - Persistence

    private func saveMetadata() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let metadata = MetadataStore(speakers: store.speakers, nextId: store.nextId)
        do {
            let data = try encoder.encode(metadata)
            try data.write(to: metadataURL, options: .atomic)
            AudioLog.pipeline.debug("Metadata saved: \(self.store.speakers.count) speaker(s)")
        } catch {
            AudioLog.pipeline.error("Metadata save failed: \(error)")
            throw error
        }
    }

    private func saveCentroid(_ centroid: SpeakerCentroid) throws {
        let url = centroidsDir.appendingPathComponent("centroid-\(centroid.speakerId).bin")
        var data = Data(capacity: 4 + centroid.centroid.count * MemoryLayout<Float>.size)
        var sampleCount = UInt32(centroid.sampleCount)
        withUnsafeBytes(of: &sampleCount) { data.append(contentsOf: $0) }
        centroid.centroid.withUnsafeBytes { data.append(contentsOf: $0) }
        do {
            try data.write(to: url, options: .atomic)
            AudioLog.pipeline.debug("Centroid saved: speaker \(centroid.speakerId), \(centroid.sampleCount) sample(s)")
        } catch {
            AudioLog.pipeline.error("Centroid save failed for speaker \(centroid.speakerId): \(error)")
            throw error
        }
    }

    private func deleteCentroidFile(speakerId: Int64) {
        let url = centroidsDir.appendingPathComponent("centroid-\(speakerId).bin")
        do {
            try FileManager.default.removeItem(at: url)
        } catch let error as CocoaError where error.code == .fileNoSuchFile {
            // File doesn't exist — speaker had no centroid, nothing to clean up.
        } catch {
            AudioLog.pipeline.warning("Registry: failed to delete centroid file for speaker \(speakerId): \(error)")
        }
    }
}

// MARK: - Centroid Binary I/O

/// Load a centroid from a binary file: UInt32 sampleCount + Float32[].
private func loadCentroid(from url: URL, speakerId: Int64) throws -> SpeakerCentroid {
    let data = try Data(contentsOf: url)
    // Expected layout: 4-byte UInt32 sampleCount + 256 × Float32 = 1028 bytes
    let expectedFloatCount = 256
    let expectedSize = 4 + expectedFloatCount * MemoryLayout<Float>.size
    guard data.count == expectedSize else {
        throw CocoaError(.fileReadCorruptFile)
    }
    let sampleCount = data.withUnsafeBytes { $0.load(as: UInt32.self) }
    let centroid: [Float] = data.withUnsafeBytes { raw in
        let floatPtr = raw.baseAddress!.advanced(by: 4).assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: floatPtr, count: expectedFloatCount))
    }
    return SpeakerCentroid(speakerId: speakerId, centroid: centroid, sampleCount: Int(sampleCount))
}
