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
    /// - Returns: A `ProcessedSession` with registry-resolved speaker identities
    ///   and, if an ASR model was provided, per-segment transcripts.
    public func process(
        audioURL: URL,
        audio: [Float],
        config: DiarizationConfig = .default,
        threshold: Float? = nil
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

            let speaker = try await registry.resolve(embedding: embedding, qualityScore: bestQuality, threshold: threshold)
            localToRegistry[localId] = speaker
        }

        // Build annotated output, slicing audio per segment for ASR if available
        var annotated: [AnnotatedSegment] = []
        for seg in diarResult.segments {
            guard let speaker = localToRegistry[seg.speakerId] else { continue }
            let start = Double(seg.startTime)
            let end = Double(seg.endTime)

            let transcript = asr.map { model in
                let startSample = Int(start * Double(sampleRate))
                let endSample = min(Int(end * Double(sampleRate)), audio.count)
                let slice = Array(audio[startSample..<endSample])
                return model.transcribe(audio: slice, sampleRate: sampleRate, language: nil)
            }

            annotated.append(AnnotatedSegment(
                speaker: speaker,
                startTime: start,
                endTime: end,
                transcriptText: transcript))
        }

        let asrNote = asr != nil ? " with transcription" : ""
        AudioLog.pipeline.info("Resolved \(localToRegistry.count) speaker(s) from \(audioURL.lastPathComponent)\(asrNote)")
        return ProcessedSession(segments: annotated)
    }
}
