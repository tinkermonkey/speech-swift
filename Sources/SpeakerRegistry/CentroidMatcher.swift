import Foundation

// MARK: - Cosine Similarity

/// Cosine similarity between two L2-normalised vectors (equals dot product).
/// WeSpeaker embeddings are already unit-normalised, so no normalisation step is needed here.
func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    zip(a, b).reduce(0, { $0 + $1.0 * $1.1 })
}

// MARK: - L2 Normalisation

/// Re-normalise a vector to the unit sphere.
func normalizeL2(_ v: [Float]) -> [Float] {
    let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
    guard norm > 1e-8 else { return v }
    return v.map { $0 / norm }
}

// MARK: - Online Centroid Update

/// Incremental mean update: `centroid = (old * n + new) / (n + 1)`, re-normalised.
func incrementalCentroid(old: [Float], sampleCount: Int, new embedding: [Float]) -> [Float] {
    let n = Float(sampleCount)
    let updated = zip(old, embedding).map { ($0 * n + $1) / (n + 1) }
    return normalizeL2(updated)
}

// MARK: - Centroid Blend (used during merge)

/// Weighted average of two centroids by their sample counts, re-normalised.
func blendCentroids(
    centroidA: [Float], countA: Int,
    centroidB: [Float], countB: Int
) -> [Float] {
    let wA = Float(countA)
    let wB = Float(countB)
    let total = wA + wB
    let blended = zip(centroidA, centroidB).map { (wA * $0 + wB * $1) / total }
    return normalizeL2(blended)
}
