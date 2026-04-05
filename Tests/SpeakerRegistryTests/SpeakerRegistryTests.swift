import XCTest
import Foundation
@testable import SpeakerRegistry

final class SpeakerRegistryTests: XCTestCase {

    // MARK: - Centroid Math

    func testCosineSimilaritySameVector() {
        let v: [Float] = [1, 0, 0, 0]
        XCTAssertEqual(cosineSimilarity(v, v), 1.0, accuracy: 1e-6)
    }

    func testCosineSimilarityOrthogonal() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [0, 1, 0]
        XCTAssertEqual(cosineSimilarity(a, b), 0.0, accuracy: 1e-6)
    }

    func testCosineSimilarityOpposite() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [-1, 0, 0]
        XCTAssertEqual(cosineSimilarity(a, b), -1.0, accuracy: 1e-6)
    }

    func testNormalizeL2UnitVector() {
        let v: [Float] = [3, 4]
        let n = normalizeL2(v)
        let norm = sqrt(n.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: 1e-6)
        XCTAssertEqual(n[0], 0.6, accuracy: 1e-5)
        XCTAssertEqual(n[1], 0.8, accuracy: 1e-5)
    }

    func testNormalizeL2ZeroVector() {
        let v: [Float] = [0, 0, 0]
        let n = normalizeL2(v)
        XCTAssertEqual(n, v)
    }

    func testIncrementalCentroidSingleSample() {
        let old: [Float] = [1, 0, 0]
        let new: [Float] = [0, 1, 0]
        let result = incrementalCentroid(old: old, sampleCount: 1, new: new)
        // Expected: mean([1,0,0], [0,1,0]) = [0.5, 0.5, 0], then normalised
        let norm = sqrt(result.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: 1e-5)
    }

    func testIncrementalCentroidStaysNormalised() {
        var centroid: [Float] = normalizeL2([1, 2, 3])
        var count = 1
        for _ in 0..<20 {
            let newEmb = normalizeL2([Float.random(in: -1...1),
                                      Float.random(in: -1...1),
                                      Float.random(in: -1...1)])
            centroid = incrementalCentroid(old: centroid, sampleCount: count, new: newEmb)
            count += 1
        }
        let norm = sqrt(centroid.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: 1e-4)
    }

    func testBlendCentroids() {
        let a: [Float] = [1, 0, 0]
        let b: [Float] = [0, 1, 0]
        let blended = blendCentroids(centroidA: a, countA: 1, centroidB: b, countB: 1)
        let norm = sqrt(blended.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: 1e-5)
        XCTAssertEqual(blended[0], blended[1], accuracy: 1e-5)
    }

    // MARK: - In-Memory Registry

    func testOpenRegistry() throws {
        let registry = try makeRegistry()
        let speakers = waitFor { await registry.speakers() }
        XCTAssertEqual(speakers.count, 0)
    }

    func testResolveCreatesPlaceholder() throws {
        let registry = try makeRegistry()
        let embedding = normalizeL2((0..<256).map { Float($0) })
        let speaker = try waitFor { try await registry.resolve(embedding: embedding, qualityScore: 3.0) }
        XCTAssertNotNil(speaker.id)
        XCTAssertFalse(speaker.isLabeled)
    }

    func testResolveMatchesSameSpeaker() throws {
        let registry = try makeRegistry()
        let base = normalizeL2((0..<256).map { Float($0) })

        let first = try waitFor { try await registry.resolve(embedding: base, qualityScore: 3.0) }

        let slightly = normalizeL2(base.map { $0 + 0.001 })
        let second = try waitFor { try await registry.resolve(embedding: slightly, qualityScore: 3.0) }
        XCTAssertEqual(first.id, second.id, "Same speaker should be matched")
    }

    func testResolveLowQualityDoesNotEnroll() throws {
        let registry = try makeRegistry()
        let embedding = normalizeL2((0..<256).map { Float($0) })

        // qualityScore < 2.0 — should create a placeholder but not enroll a centroid
        let speaker = try waitFor { try await registry.resolve(embedding: embedding, qualityScore: 1.5) }
        XCTAssertNotNil(speaker.id)

        // A second resolve with a high-quality, very-similar embedding should NOT match the
        // placeholder from the first call because no centroid was enrolled yet.
        let slightly = normalizeL2(embedding.map { $0 + 0.001 })
        let second = try waitFor { try await registry.resolve(embedding: slightly, qualityScore: 3.0) }
        XCTAssertNotEqual(speaker.id, second.id,
            "Low-quality segment must not enroll a centroid, so the follow-up should create a new speaker")
    }

    func testLabelSpeaker() throws {
        let registry = try makeRegistry()
        let embedding = normalizeL2((0..<256).map { _ in Float.random(in: -1...1) })
        let speaker = try waitFor { try await registry.resolve(embedding: embedding, qualityScore: 3.0) }

        try waitFor { try await registry.label(speakerId: speaker.id!, displayName: "Alice") }

        let updated = waitFor { await registry.speaker(id: speaker.id!) }
        XCTAssertEqual(updated?.displayName, "Alice")
        XCTAssertTrue(updated?.isLabeled ?? false)
    }

    func testMergeSpeakers() throws {
        let registry = try makeRegistry()

        let embA = normalizeL2([Float](repeating: 1, count: 256))
        let embB = normalizeL2([Float](repeating: -1, count: 256))

        let spkA = try waitFor { try await registry.resolve(embedding: embA, qualityScore: 3.0) }
        let spkB = try waitFor { try await registry.resolve(embedding: embB, qualityScore: 3.0) }

        try waitFor { try await registry.merge(src: spkA.id!, into: spkB.id!) }

        let remaining = waitFor { await registry.speakers() }
        XCTAssertEqual(remaining.count, 1)
        XCTAssertEqual(remaining.first?.id, spkB.id)

        let srcGone = waitFor { await registry.speaker(id: spkA.id!) }
        XCTAssertNil(srcGone)
    }

    func testPersistsAcrossOpen() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("test-registry-\(UUID().uuidString).json")
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }

        let embedding = normalizeL2((0..<256).map { Float($0) })

        // First open — enroll a speaker
        let reg1 = try SpeakerRegistry.open(at: url, similarityThreshold: 0.75)
        let spk = try waitFor { try await reg1.resolve(embedding: embedding, qualityScore: 3.0) }
        try waitFor { try await reg1.label(speakerId: spk.id!, displayName: "Bob") }

        // Second open — should find the same speaker
        let reg2 = try SpeakerRegistry.open(at: url, similarityThreshold: 0.75)
        let speakers = waitFor { await reg2.speakers() }
        XCTAssertEqual(speakers.count, 1)
        XCTAssertEqual(speakers.first?.displayName, "Bob")
    }

    // MARK: - Helpers

    private func makeRegistry() throws -> SpeakerRegistry {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("test-registry-\(UUID().uuidString).json")
        addTeardownBlock { try? FileManager.default.removeItem(at: tmp) }
        return try SpeakerRegistry.open(at: tmp, similarityThreshold: 0.75)
    }

    /// Bridge async actor calls into synchronous XCTest.
    @discardableResult
    private func waitFor<T>(_ block: @escaping () async throws -> T) throws -> T {
        var result: Result<T, Error>?
        let sem = DispatchSemaphore(value: 0)
        Task {
            do { result = .success(try await block()) }
            catch { result = .failure(error) }
            sem.signal()
        }
        sem.wait()
        return try result!.get()
    }

    /// Non-throwing variant for actor queries that don't throw.
    @discardableResult
    private func waitFor<T>(_ block: @escaping () async -> T) -> T {
        var result: T?
        let sem = DispatchSemaphore(value: 0)
        Task {
            result = await block()
            sem.signal()
        }
        sem.wait()
        return result!
    }
}
