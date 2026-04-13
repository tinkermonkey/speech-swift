import Foundation
import AudioCommon
import SpeechVAD

// MARK: - Output Types

/// A diarized audio file fully resolved against the speaker registry.
public struct ProcessedSession: Sendable {
    public let segments: [AnnotatedSegment]

    /// Number of distinct identified registry speakers seen. Unidentified segments are excluded.
    public var numSpeakers: Int {
        Set(segments.compactMap(\.speaker?.id)).count
    }
}

/// A single speaker turn with its registry-resolved identity.
///
/// `speaker` is `nil` when the clip was too short for recognition or no registered
/// speaker met the similarity threshold. The transcript is still populated if an
/// ASR model was provided.
public struct AnnotatedSegment: Sendable {
    public let speaker: Speaker?
    public let startTime: Double
    public let endTime: Double
    public let transcriptText: String?

    public var duration: Double { endTime - startTime }
}

// MARK: - PipelineSession

/// Ties together `DiarizationPipeline`, `SpeakerRegistry`, and an optional ASR model
/// for a single audio file.
///
/// Typical usage:
/// ```swift
/// let registry = try SpeakerRegistry.open()
/// let diarizer = try await DiarizationPipeline.fromPretrained()
/// let asr = try await Qwen3ASRModel.fromPretrained()
/// let ps = PipelineSession(diarizer: diarizer, registry: registry, asr: asr)
/// let result = try await ps.process(audioURL: url, audio: samples)
/// ```
public struct PipelineSession: Sendable {

    public let diarizer: DiarizationPipeline
    public let registry: SpeakerRegistry
    /// Optional ASR model. When provided, each segment is transcribed and
    /// `AnnotatedSegment.transcriptText` is populated.
    public let asr: (any SpeechRecognitionModel)?

    public init(
        diarizer: DiarizationPipeline,
        registry: SpeakerRegistry,
        asr: (any SpeechRecognitionModel)? = nil
    ) {
        self.diarizer = diarizer
        self.registry = registry
        self.asr = asr
    }

    // MARK: - Process

    /// Diarize `audio`, resolve each speaker cluster against the registry, and
    /// optionally transcribe each segment with the injected ASR model.
    ///
    /// - Parameters:
    ///   - audioURL: Source file path (used for log messages).
    ///   - audio: Float32 PCM at 16 kHz.
    ///   - config: Diarization hyper-parameters.
    ///   - threshold: Cosine similarity override for this request.
    ///   - minimumDurationForRecognition: Clips shorter than this (seconds) skip diarization
    ///     and registry entirely. ASR still runs if a model is available. Defaults to 10 s.
    /// - Returns: A `ProcessedSession` with registry-resolved speaker identities
    ///   and, if an ASR model was provided, per-segment transcripts.
    ///   `AnnotatedSegment.speaker` is `nil` when the clip was too short or no
    ///   registered speaker matched.
    public func process(
        audioURL: URL,
        audio: [Float],
        config: DiarizationConfig = .default,
        threshold: Float? = nil,
        minimumDurationForRecognition: Double = 10.0
    ) async throws -> ProcessedSession {
        let sampleRate = 16000
        let durationSeconds = Double(audio.count) / Double(sampleRate)

        // ── Short-clip guard ─────────────────────────────────────────────────
        // Clips below the minimum are not suitable for speaker recognition.
        // Skip diarization and registry entirely to avoid polluting the registry
        // with low-confidence identities. ASR still runs if available.
        if durationSeconds < minimumDurationForRecognition {
            AudioLog.pipeline.info("Clip too short for speaker recognition (\(String(format: "%.1f", durationSeconds))s < \(minimumDurationForRecognition)s) — ASR only: \(audioURL.lastPathComponent)")
            return try await asrOnlySession(audio: audio, sampleRate: sampleRate, durationSeconds: durationSeconds)
        }

        // ── Full pipeline ────────────────────────────────────────────────────
        AudioLog.pipeline.info("Diarizing \(audioURL.lastPathComponent) (\(String(format: "%.1f", durationSeconds))s)")
        let diarResult = diarizer.diarize(audio: audio, sampleRate: sampleRate, config: config)

        // Map local diarization speaker index → registry Speaker (nil if unidentifiable)
        var localToRegistry: [Int: Speaker?] = [:]
        for (localId, embedding) in diarResult.speakerEmbeddings.enumerated() {
            let speakerSegs = diarResult.segments.filter { $0.speakerId == localId }
            let bestQuality = speakerSegs.map(\.duration).max().map(Double.init) ?? 0.0

            // resolve() creates new speakers only for high-quality (≥ 2s) segments.
            // Low-quality segments with no match return nil rather than a placeholder.
            let speaker = try await registry.resolve(
                embedding: embedding,
                qualityScore: bestQuality,
                threshold: threshold)
            localToRegistry[localId] = speaker
        }

        // Build annotated output
        var annotated: [AnnotatedSegment] = []
        for seg in diarResult.segments {
            guard let speakerEntry = localToRegistry[seg.speakerId] else { continue }
            let start = Double(seg.startTime)
            let end = Double(seg.endTime)

            let transcript = asr.map { model in
                let startSample = Int(start * Double(sampleRate))
                let endSample = min(Int(end * Double(sampleRate)), audio.count)
                let slice = Array(audio[startSample..<endSample])
                return model.transcribe(audio: slice, sampleRate: sampleRate, language: nil)
            }

            annotated.append(AnnotatedSegment(
                speaker: speakerEntry,
                startTime: start,
                endTime: end,
                transcriptText: transcript))
        }

        let asrNote = asr != nil ? " with transcription" : ""
        let identified = localToRegistry.values.compactMap { $0 }.count
        AudioLog.pipeline.info("Resolved \(identified)/\(localToRegistry.count) speaker(s) from \(audioURL.lastPathComponent)\(asrNote)")
        return ProcessedSession(segments: annotated)
    }

    // MARK: - ASR-Only Path

    /// Run ASR across the entire clip as a single segment. Used when the clip is
    /// too short for speaker recognition.
    private func asrOnlySession(
        audio: [Float],
        sampleRate: Int,
        durationSeconds: Double
    ) async throws -> ProcessedSession {
        guard let asr else {
            return ProcessedSession(segments: [])
        }
        let transcript = asr.transcribe(audio: audio, sampleRate: sampleRate, language: nil)
        let segment = AnnotatedSegment(
            speaker: nil,
            startTime: 0,
            endTime: durationSeconds,
            transcriptText: transcript.isEmpty ? nil : transcript)
        return ProcessedSession(segments: [segment])
    }
}
