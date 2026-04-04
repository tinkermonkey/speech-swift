import Foundation
import GRDB
import AudioCommon
import os

// MARK: - SpeakerRegistry

/// Persistent speaker identity store backed by SQLite.
///
/// Resolves raw WeSpeaker embeddings to named (or placeholder) speaker identities
/// using cosine-similarity matching against running per-speaker centroids.
/// Thread-safe via Swift actor isolation; all SQLite I/O is synchronous but fast.
public actor SpeakerRegistry {

    private let db: DatabaseQueue
    public let similarityThreshold: Float

    // MARK: - Factory

    /// Open (or create) a registry database at `url`.
    ///
    /// - Parameters:
    ///   - url: Path to the SQLite file. Defaults to `~/Library/Caches/qwen3-speech/speaker-registry.sqlite`.
    ///   - similarityThreshold: Minimum cosine similarity to count as a match (default 0.75).
    public static func open(
        at url: URL = .defaultRegistryURL,
        similarityThreshold: Float = 0.75
    ) throws -> SpeakerRegistry {
        let db = try DatabaseQueue(path: url.path)
        try SpeakerRegistry.migrate(db: db)
        return SpeakerRegistry(db: db, threshold: similarityThreshold)
    }

    private init(db: DatabaseQueue, threshold: Float) {
        self.db = db
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
    ///   - sessionId: Session this embedding belongs to (for storage).
    ///   - start: Segment start time (seconds).
    ///   - end: Segment end time (seconds).
    /// - Returns: The matched or newly created `Speaker`.
    public func resolve(
        embedding: [Float],
        qualityScore: Double,
        sessionId: Int64,
        start: Double,
        end: Double
    ) throws -> Speaker {
        let isHighQuality = qualityScore >= 2.0

        if let (speaker, similarity) = try bestMatch(embedding: embedding) {
            AudioLog.pipeline.debug("Matched \(speaker.label) (cosine=\(similarity, format: .fixed(precision: 3)))")
            if isHighQuality {
                try updateCentroid(speakerId: speaker.id!, with: embedding)
                try storeEmbedding(
                    speakerId: speaker.id!, sessionId: sessionId,
                    start: start, end: end, quality: qualityScore, embedding: embedding)
            }
            return speaker
        } else {
            if isHighQuality {
                var speaker = try mintPlaceholder()
                try enrollNew(speakerId: speaker.id!, embedding: embedding,
                              sessionId: sessionId, start: start, end: end, quality: qualityScore)
                AudioLog.pipeline.debug("Enrolled new \(speaker.label)")
                return speaker
            } else {
                // Short segment, no match — create a placeholder but don't enroll
                let speaker = try mintPlaceholder()
                AudioLog.pipeline.debug("Created placeholder \(speaker.label) (short segment, no match)")
                return speaker
            }
        }
    }

    // MARK: - Label Management

    /// Assign a human-readable name to a speaker.
    public func label(speakerId: Int64, displayName: String) throws {
        try db.write { db in
            try db.execute(
                sql: "UPDATE speaker SET displayName = ? WHERE id = ?",
                arguments: [displayName, speakerId])
        }
        AudioLog.pipeline.info("Labeled speaker \(speakerId) as '\(displayName)'")
    }

    /// Merge `src` into `dst`: re-points all segments and embeddings, blends centroids, removes `src`.
    /// All mutations are committed in a single transaction — safe against partial failures.
    public func merge(src: Int64, into dst: Int64) throws {
        try db.write { db in
            // Read both centroids inside the same transaction so the fetch and
            // the subsequent mutations are consistent.
            let srcCentroid = try SpeakerCentroid.fetchOne(db, key: src)
            let dstCentroid = try SpeakerCentroid.fetchOne(db, key: dst)

            // Re-point all references from src to dst
            try db.execute(
                sql: "UPDATE speaker_segment SET speakerId = ? WHERE speakerId = ?",
                arguments: [dst, src])
            try db.execute(
                sql: "UPDATE speaker_embedding SET speakerId = ? WHERE speakerId = ?",
                arguments: [dst, src])

            // Blend centroids before deleting src
            if let sc = srcCentroid, let dc = dstCentroid {
                let blended = blendCentroids(
                    centroidA: dc.centroid.toFloats(), countA: dc.sampleCount,
                    centroidB: sc.centroid.toFloats(), countB: sc.sampleCount)
                try db.execute(
                    sql: "UPDATE speaker_centroid SET centroid = ?, sampleCount = ? WHERE speakerId = ?",
                    arguments: [Data.fromFloats(blended), dc.sampleCount + sc.sampleCount, dst])
            }

            // Remove src (centroid first to satisfy any FK ordering, then speaker row)
            try db.execute(sql: "DELETE FROM speaker_centroid WHERE speakerId = ?", arguments: [src])
            try db.execute(sql: "DELETE FROM speaker WHERE id = ?", arguments: [src])
        }

        AudioLog.pipeline.info("Merged speaker \(src) into \(dst)")
    }

    // MARK: - Queries

    public func speakers() throws -> [Speaker] {
        try db.read { db in try Speaker.fetchAll(db) }
    }

    public func segments(for speakerId: Int64) throws -> [SpeakerSegment] {
        try db.read { db in
            try SpeakerSegment
                .filter(Column("speakerId") == speakerId)
                .order(Column("startTime"))
                .fetchAll(db)
        }
    }

    public func speaker(id: Int64) throws -> Speaker? {
        try db.read { db in try Speaker.fetchOne(db, key: id) }
    }

    // MARK: - Private Helpers

    private func bestMatch(embedding: [Float]) throws -> (Speaker, Float)? {
        let centroids = try db.read { db in try SpeakerCentroid.fetchAll(db) }
        var best: (Int64, Float)?
        for c in centroids {
            let sim = cosineSimilarity(embedding, c.centroid.toFloats())
            if sim > similarityThreshold, sim > (best?.1 ?? 0) {
                best = (c.speakerId, sim)
            }
        }
        guard let (speakerId, sim) = best else { return nil }
        let speaker = try db.read { db in try Speaker.fetchOne(db, key: speakerId) }
        return speaker.map { ($0, sim) }
    }

    private func mintPlaceholder() throws -> Speaker {
        var speaker = Speaker()
        try db.write { db in try speaker.insert(db) }
        return speaker
    }

    private func enrollNew(
        speakerId: Int64,
        embedding: [Float],
        sessionId: Int64,
        start: Double,
        end: Double,
        quality: Double
    ) throws {
        var centroid = SpeakerCentroid(speakerId: speakerId, centroid: normalizeL2(embedding))
        try db.write { db in try centroid.insert(db) }
        try storeEmbedding(
            speakerId: speakerId, sessionId: sessionId,
            start: start, end: end, quality: quality, embedding: embedding)
    }

    private func updateCentroid(speakerId: Int64, with embedding: [Float]) throws {
        try db.write { db in
            guard var c = try SpeakerCentroid.fetchOne(db, key: speakerId) else { return }
            let updated = incrementalCentroid(
                old: c.centroid.toFloats(), sampleCount: c.sampleCount, new: embedding)
            c.centroid = Data.fromFloats(updated)
            c.sampleCount += 1
            try c.save(db)
        }
    }

    private func storeEmbedding(
        speakerId: Int64,
        sessionId: Int64,
        start: Double,
        end: Double,
        quality: Double,
        embedding: [Float]
    ) throws {
        var emb = SpeakerEmbedding(
            speakerId: speakerId, sessionId: sessionId,
            segmentStart: start, segmentEnd: end,
            qualityScore: quality, embedding: embedding)
        try db.write { db in try emb.insert(db) }
    }

    // MARK: - Schema Migration

    private static func migrate(db: DatabaseQueue) throws {
        var migrator = DatabaseMigrator()

        migrator.registerMigration("v1_initial") { db in
            try db.create(table: "speaker", ifNotExists: true) { t in
                t.autoIncrementedPrimaryKey("id")
                t.column("displayName", .text)
                t.column("createdAt", .datetime).notNull().defaults(sql: "CURRENT_TIMESTAMP")
                t.column("notes", .text)
            }

            try db.create(table: "speaker_session", ifNotExists: true) { t in
                t.autoIncrementedPrimaryKey("id")
                t.column("audioPath", .text).notNull()
                t.column("recordedAt", .datetime).notNull()
                t.column("processedAt", .datetime)
                t.column("durationSeconds", .double).notNull()
            }

            try db.create(table: "speaker_segment", ifNotExists: true) { t in
                t.autoIncrementedPrimaryKey("id")
                t.column("sessionId", .integer).notNull().references("speaker_session", onDelete: .cascade)
                t.column("speakerId", .integer).notNull().references("speaker", onDelete: .cascade)
                t.column("startTime", .double).notNull()
                t.column("endTime", .double).notNull()
                t.column("transcriptText", .text)
            }

            try db.create(table: "speaker_embedding", ifNotExists: true) { t in
                t.autoIncrementedPrimaryKey("id")
                t.column("speakerId", .integer).notNull().references("speaker", onDelete: .cascade)
                t.column("sessionId", .integer).notNull().references("speaker_session", onDelete: .cascade)
                t.column("segmentStart", .double).notNull()
                t.column("segmentEnd", .double).notNull()
                t.column("qualityScore", .double).notNull()
                t.column("embedding", .blob).notNull()
            }

            try db.create(table: "speaker_centroid", ifNotExists: true) { t in
                t.primaryKey(["speakerId"])
                t.column("speakerId", .integer).notNull().references("speaker", onDelete: .cascade)
                t.column("centroid", .blob).notNull()
                t.column("sampleCount", .integer).notNull()
            }
        }

        try migrator.migrate(db)
    }

    // MARK: - Session Persistence (used by PipelineSession)

    /// Insert a new session record and return its auto-assigned id.
    func insertSession(_ session: inout SpeakerSession) throws -> Int64 {
        try db.write { db in
            try session.insert(db)
            return db.lastInsertedRowID
        }
    }

    /// Insert a segment record.
    func insertSegment(_ segment: SpeakerSegment) throws {
        var s = segment
        try db.write { db in try s.insert(db) }
    }

    /// Mark a session's `processedAt` timestamp.
    func markSessionProcessed(id: Int64) throws {
        try db.write { db in
            try db.execute(
                sql: "UPDATE speaker_session SET processedAt = ? WHERE id = ?",
                arguments: [Date(), id])
        }
    }
}
