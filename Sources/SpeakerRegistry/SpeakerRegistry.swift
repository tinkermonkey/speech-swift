import Foundation
import AudioCommon

private func registryElapsedMs(from start: ContinuousClock.Instant) -> Int {
    let d = ContinuousClock.now - start
    return Int(Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15)
}

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

    // MARK: - Observability

    /// Number of centroids currently held in memory. Grows with each enrollment.
    /// Linear scan in `bestMatch()` is O(centroidCount) per embedding.
    public var centroidCount: Int { store.centroids.count }

    /// Number of registered speakers.
    public var speakerCount: Int { store.speakers.count }

    // MARK: - Resolve / Match

    /// Match an embedding against registered speakers without creating new entries.
    ///
    /// Use this on the fast path (short clips) where you want to identify a known
    /// speaker but must not pollute the registry with unconfirmed identities.
    ///
    /// - Returns: The best-matching `Speaker` (nil if below threshold) and the top candidate's
    ///   cosine similarity score (nil only if the registry is empty).
    public func match(embedding: [Float], threshold: Float? = nil) -> (speaker: Speaker?, score: Float?) {
        let effectiveThreshold = threshold ?? similarityThreshold
        let score = topSimilarity(embedding: embedding)
        guard let (speaker, similarity) = bestMatch(embedding: embedding, threshold: effectiveThreshold) else {
            return (nil, score)
        }
        AudioLog.pipeline.debug("Matched \(speaker.label) (cosine=\(String(format: "%.3f", similarity)))")
        return (speaker, similarity)
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
    /// - Returns: The matched or newly enrolled `Speaker` and the top candidate's cosine similarity
    ///   score (nil only if the registry is empty or the speaker was newly enrolled with no prior match).
    public func resolve(embedding: [Float], qualityScore: Double, threshold: Float? = nil) throws -> (speaker: Speaker?, score: Float?) {
        let isHighQuality = qualityScore >= 2.0
        let effectiveThreshold = threshold ?? similarityThreshold
        let topScore = topSimilarity(embedding: embedding)

        if let (speaker, similarity) = bestMatch(embedding: embedding, threshold: effectiveThreshold) {
            AudioLog.pipeline.debug("Matched \(speaker.label) (cosine=\(String(format: "%.3f", similarity)))")
            if isHighQuality {
                guard embedding.allSatisfy({ $0.isFinite }) else {
                    AudioLog.pipeline.warning("Skipped centroid update for \(speaker.label): embedding contains NaN/Inf")
                    return (speaker, similarity)
                }
                updateCentroid(speakerId: speaker.id!, with: embedding)
                if let updated = store.centroids.first(where: { $0.speakerId == speaker.id! }) {
                    try saveCentroid(updated)
                } else {
                    AudioLog.pipeline.warning("Registry: centroid missing after update for speaker \(speaker.id!); not saved")
                }
            }
            return (speaker, similarity)
        } else if isHighQuality {
            // High-quality segment with no match → enroll as new speaker.
            let speaker = mintPlaceholder()
            guard embedding.allSatisfy({ $0.isFinite }) else {
                AudioLog.pipeline.warning("Skipped centroid enrolment for \(speaker.label): embedding contains NaN/Inf")
                try saveMetadata()
                return (speaker, topScore)
            }
            let centroid = SpeakerCentroid(speakerId: speaker.id!, centroid: normalizeL2(embedding))
            store.centroids.append(centroid)
            try saveCentroid(centroid)  // centroid first — if this fails, speaker is not persisted
            try saveMetadata()
            AudioLog.pipeline.debug("Enrolled new \(speaker.label)")
            return (speaker, topScore)
        } else {
            // Low-quality segment with no match → do not create a placeholder.
            AudioLog.pipeline.debug("No match for low-quality segment (\(String(format: "%.1f", qualityScore))s), skipping enrollment")
            return (nil, topScore)
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

    /// Returns other speakers ranked by centroid similarity to `id`, for merge candidate discovery.
    ///
    /// - Parameters:
    ///   - id: The reference speaker.
    ///   - limit: Maximum number of candidates to return.
    ///   - minSimilarity: Floor similarity — candidates below this are excluded.
    ///     Defaults to 0.5, intentionally below the enrollment threshold (0.75) to surface
    ///     near-duplicate speakers that didn't auto-merge.
    /// - Returns: Candidates sorted by descending similarity, or throws if `id` has no centroid.
    public func similarSpeakers(
        to id: Int64,
        limit: Int = 10,
        minSimilarity: Float = 0.5
    ) throws -> [(speaker: Speaker, similarity: Float, sampleCount: Int)] {
        guard let target = store.centroids.first(where: { $0.speakerId == id }) else {
            throw RegistryError.speakerNotFound(id)
        }
        var candidates: [(speaker: Speaker, similarity: Float, sampleCount: Int)] = []
        for centroid in store.centroids where centroid.speakerId != id {
            let sim = cosineSimilarity(target.centroid, centroid.centroid)
            guard sim >= minSimilarity else { continue }
            guard let speaker = store.speakers.first(where: { $0.id == centroid.speakerId }) else { continue }
            candidates.append((speaker, sim, centroid.sampleCount))
        }
        return candidates
            .sorted { $0.similarity > $1.similarity }
            .prefix(limit)
            .map { $0 }
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

    /// Returns the raw top cosine similarity against all centroids, ignoring threshold.
    /// `nil` only when the registry is empty.
    private func topSimilarity(embedding: [Float]) -> Float? {
        store.centroids.map { cosineSimilarity(embedding, $0.centroid) }.max()
    }

    private func bestMatch(embedding: [Float], threshold: Float) -> (Speaker, Float)? {
        var best: (speakerId: Int64, sim: Float)?
        var secondBestSim: Float = 0

        for c in store.centroids {
            let sim = cosineSimilarity(embedding, c.centroid)
            if sim > (best?.sim ?? 0) {
                if let prev = best { secondBestSim = prev.sim }
                best = (c.speakerId, sim)
            } else if sim > secondBestSim {
                secondBestSim = sim
            }
        }

        let regSize = store.centroids.count
        guard let (speakerId, sim) = best, sim > threshold else {
            AudioLog.pipeline.debug("Registry: no match (best=\(String(format: "%.3f", best?.sim ?? 0)) threshold=\(String(format: "%.3f", threshold)) reg_size=\(regSize))")
            return nil
        }

        guard let speaker = store.speakers.first(where: { $0.id == speakerId }) else {
            AudioLog.pipeline.warning("Registry: centroid references speaker \(speakerId) not found in store — data may be inconsistent")
            return nil
        }

        let margin = sim - secondBestSim
        AudioLog.pipeline.debug("Registry: matched \(speaker.label) sim=\(String(format: "%.3f", sim)) runner_up=\(String(format: "%.3f", secondBestSim)) margin=\(String(format: "%.3f", margin)) reg_size=\(regSize)")
        return (speaker, sim)
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
            let tWrite = ContinuousClock.now
            try data.write(to: metadataURL, options: .atomic)
            let writeMs = registryElapsedMs(from: tWrite)
            AudioLog.pipeline.debug("Metadata saved: \(self.store.speakers.count) speaker(s) io=\(writeMs)ms")
            if writeMs > 50 {
                AudioLog.pipeline.warning("Registry: metadata write slow (\(writeMs)ms) — filesystem contention?")
            }
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
            let tWrite = ContinuousClock.now
            try data.write(to: url, options: .atomic)
            let writeMs = registryElapsedMs(from: tWrite)
            AudioLog.pipeline.debug("Centroid saved: speaker \(centroid.speakerId), \(centroid.sampleCount) sample(s) io=\(writeMs)ms")
            if writeMs > 50 {
                AudioLog.pipeline.warning("Registry: centroid write slow (\(writeMs)ms) for speaker \(centroid.speakerId) — filesystem contention?")
            }
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
