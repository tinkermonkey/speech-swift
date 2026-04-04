import Foundation
import AudioCommon
import SpeechVAD

// MARK: - Output Types

/// A diarized audio file fully resolved against the speaker registry.
public struct ProcessedSession: Sendable {
    public let session: SpeakerSession
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

    /// Diarize `audio`, resolve each speaker cluster against the registry, and persist results.
    ///
    /// - Parameters:
    ///   - audioURL: Source file path (stored in the session record).
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

        // Persist session record
        var session = SpeakerSession(
            audioPath: audioURL.path,
            recordedAt: .now,
            durationSeconds: durationSeconds)
        let sessionId = try await registry.insertSession(&session)

        // Map local diarization speaker index → registry Speaker
        var localToRegistry: [Int: Speaker] = [:]
        for (localId, embedding) in diarResult.speakerEmbeddings.enumerated() {
            // Find the longest segment for this local speaker as quality proxy
            let speakerSegs = diarResult.segments.filter { $0.speakerId == localId }
            let bestQuality = speakerSegs.map(\.duration).max().map(Double.init) ?? 0.0

            // Use the centroid embedding from diarization (already averaged over all segments)
            let speaker = try await registry.resolve(
                embedding: embedding,
                qualityScore: bestQuality,
                sessionId: sessionId,
                start: Double(speakerSegs.first?.startTime ?? 0),
                end: Double(speakerSegs.last?.endTime ?? 0))
            localToRegistry[localId] = speaker
        }

        // Persist segments and build annotated output
        var annotated: [AnnotatedSegment] = []
        for seg in diarResult.segments {
            guard let speaker = localToRegistry[seg.speakerId] else { continue }
            let start = Double(seg.startTime)
            let end = Double(seg.endTime)

            try await registry.insertSegment(SpeakerSegment(
                sessionId: sessionId,
                speakerId: speaker.id!,
                startTime: start,
                endTime: end))

            annotated.append(AnnotatedSegment(
                speaker: speaker,
                startTime: start,
                endTime: end,
                transcriptText: nil))
        }

        // Mark session as processed
        try await registry.markSessionProcessed(id: sessionId)

        AudioLog.pipeline.info("Resolved \(localToRegistry.count) speaker(s) for session \(sessionId)")
        return ProcessedSession(session: session, segments: annotated)
    }
}

