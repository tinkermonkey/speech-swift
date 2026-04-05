import Foundation
import AudioCommon
import SpeechVAD

// MARK: - Output Types

/// A diarized audio file fully resolved against the speaker registry.
public struct ProcessedSession: Sendable {
    public let segments: [AnnotatedSegment]

    /// Number of distinct registry speakers seen.
    public var numSpeakers: Int {
        Set(segments.map(\.speaker.id)).count
    }
}

/// A single speaker turn with its registry-resolved identity.
public struct AnnotatedSegment: Sendable {
    public let speaker: Speaker
    public let startTime: Double
    public let endTime: Double
    public let transcriptText: String?

    public var duration: Double { endTime - startTime }
}

// MARK: - PipelineSession

/// Ties together `DiarizationPipeline` and `SpeakerRegistry` for a single audio file.
///
/// Typical usage:
/// ```swift
/// let registry = try SpeakerRegistry.open()
/// let diarizer = try await DiarizationPipeline.fromPretrained()
/// let ps = PipelineSession(diarizer: diarizer, registry: registry)
/// let result = try await ps.process(audioURL: url, audio: samples)
/// ```
public struct PipelineSession: Sendable {

    public let diarizer: DiarizationPipeline
    public let registry: SpeakerRegistry

    public init(diarizer: DiarizationPipeline, registry: SpeakerRegistry) {
        self.diarizer = diarizer
        self.registry = registry
    }

    // MARK: - Process

    /// Diarize `audio` and resolve each speaker cluster against the registry.
    ///
    /// - Parameters:
    ///   - audioURL: Source file path (used for log messages).
    ///   - audio: Float32 PCM at 16 kHz.
    ///   - config: Diarization hyper-parameters.
    /// - Returns: A `ProcessedSession` with registry-resolved speaker identities.
    public func process(
        audioURL: URL,
        audio: [Float],
        config: DiarizationConfig = .default
    ) async throws -> ProcessedSession {
        let sampleRate = 16000
        let durationSeconds = Double(audio.count) / Double(sampleRate)

        AudioLog.pipeline.info("Diarizing \(audioURL.lastPathComponent) (\(String(format: "%.1f", durationSeconds))s)")
        let diarResult = diarizer.diarize(audio: audio, sampleRate: sampleRate, config: config)

        // Map local diarization speaker index → registry Speaker
        var localToRegistry: [Int: Speaker] = [:]
        for (localId, embedding) in diarResult.speakerEmbeddings.enumerated() {
            let speakerSegs = diarResult.segments.filter { $0.speakerId == localId }
            let bestQuality = speakerSegs.map(\.duration).max().map(Double.init) ?? 0.0

            let speaker = try await registry.resolve(embedding: embedding, qualityScore: bestQuality)
            localToRegistry[localId] = speaker
        }

        // Build annotated output
        var annotated: [AnnotatedSegment] = []
        for seg in diarResult.segments {
            guard let speaker = localToRegistry[seg.speakerId] else { continue }
            annotated.append(AnnotatedSegment(
                speaker: speaker,
                startTime: Double(seg.startTime),
                endTime: Double(seg.endTime),
                transcriptText: nil))
        }

        AudioLog.pipeline.info("Resolved \(localToRegistry.count) speaker(s) from \(audioURL.lastPathComponent)")
        return ProcessedSession(segments: annotated)
    }
}
