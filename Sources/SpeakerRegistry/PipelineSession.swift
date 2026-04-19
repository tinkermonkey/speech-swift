import Foundation
import AudioCommon
import SpeechVAD

/// Returns elapsed milliseconds from `start` to now using the continuous (monotonic) clock.
private func elapsedMs(from start: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - start
    return Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15
}

// MARK: - Output Types

/// Per-phase wall-clock timings for a processed session (milliseconds).
public struct PipelineTimings: Sendable {
    /// Sub-phase breakdown inside diarization (segmentation, embedding, clustering).
    public let diarize: DiarizationTimings
    /// Time spent resolving speaker embeddings against the registry.
    public let resolveMs: Int
    /// Number of centroids in the registry at resolve time (linear scan cost).
    public let regSize: Int
    /// Number of new speakers enrolled during this session (each triggers disk I/O).
    public let enrolled: Int
    /// Time spent on per-segment ASR transcription (0 if no ASR model).
    public let asrMs: Int
    /// Number of diarized segments passed to ASR.
    public let segCount: Int
    /// Total inference time inside the semaphore.
    public let totalMs: Int

    /// Convenience: total diarization wall time (seg + embed + cluster).
    public var diarizeMs: Int { diarize.segmentMs + diarize.embedMs + diarize.clusterMs }
}

/// A diarized audio file fully resolved against the speaker registry.
public struct ProcessedSession: Sendable {
    public let segments: [AnnotatedSegment]
    /// Per-phase timing breakdown. Use this for structured performance logging.
    public let timings: PipelineTimings

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
    /// Top cosine similarity score against the registry (0–1), regardless of whether it
    /// crossed the match threshold. `nil` only when the registry was empty or the clip
    /// was too short for speaker recognition.
    public let bestScore: Float?
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
///
/// ## Sendable note
/// `DiarizationPipeline` and `SpeechRecognitionModel` are not thread-safe. This type is
/// marked `@unchecked Sendable` because each instance is created per-request and used
/// exclusively within a single `InferenceSemaphore.withPermit` scope — it is never shared
/// across concurrent tasks.
public struct PipelineSession: @unchecked Sendable {

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
    ///   - minimumDurationForDiarization: Clips shorter than this (seconds) skip diarization
    ///     and registry entirely — the diarizer needs a minimum of audio to produce useful
    ///     embeddings. ASR still runs if a model is available. Defaults to 1 s.
    ///   - minimumDurationForEnrollment: Clips shorter than this (seconds) run diarization
    ///     and match against existing registry speakers, but will not create new speaker
    ///     entries. Use this to identify known speakers in short clips without polluting
    ///     the registry with low-confidence new identities. Defaults to 10 s.
    ///   - language: Optional language hint passed to Qwen3-ASR (e.g. `"english"`, `"chinese"`).
    ///     When provided, skips in-model language detection — improves accuracy and reduces
    ///     compute on short or noisy segments. `nil` lets the model auto-detect.
    /// - Returns: A `ProcessedSession` with registry-resolved speaker identities
    ///   and, if an ASR model was provided, per-segment transcripts.
    ///   `AnnotatedSegment.speaker` is `nil` when no registered speaker matched.
    public func process(
        audioURL: URL,
        audio: [Float],
        config: DiarizationConfig = .default,
        threshold: Float? = nil,
        minimumDurationForDiarization: Double = 1.0,
        minimumDurationForEnrollment: Double = 10.0,
        language: String? = nil,
        requestID: String? = nil
    ) async throws -> ProcessedSession {
        let sampleRate = 16000
        let durationSeconds = Double(audio.count) / Double(sampleRate)
        let tag = "[\(currentThreadTag)]\(requestID.map { "[\($0)]" } ?? "") "
        let t0 = ContinuousClock.now

        // ── Threshold sanity ─────────────────────────────────────────────────
        // If the caller inverts the thresholds, clips that pass the diarization
        // floor would be immediately eligible for enrollment while clips between
        // the two values get ASR-only — the opposite of the intended semantics.
        // Clamp silently so the invariant always holds; the HTTP layer validates
        // inputs before reaching here, so this guards against direct library use.
        let effectiveEnrollmentMin = max(minimumDurationForEnrollment, minimumDurationForDiarization)
        if effectiveEnrollmentMin != minimumDurationForEnrollment {
            AudioLog.pipeline.warning("\(tag)minimumDurationForEnrollment (\(minimumDurationForEnrollment)s) < minimumDurationForDiarization (\(minimumDurationForDiarization)s) — clamped to \(effectiveEnrollmentMin)s")
        }

        // ── Short-clip guard ─────────────────────────────────────────────────
        // Clips below the diarization minimum cannot produce reliable speaker
        // embeddings. Skip diarization and registry entirely; ASR still runs.
        if durationSeconds < minimumDurationForDiarization {
            AudioLog.pipeline.info("\(tag)Clip too short for diarization (\(String(format: "%.1f", durationSeconds))s < \(minimumDurationForDiarization)s) — ASR only")
            let regSize = await registry.centroidCount
            return try await asrOnlySession(audio: audio, sampleRate: sampleRate, durationSeconds: durationSeconds, language: language, regSize: regSize, t0: t0)
        }

        // ── Enrollment mode ───────────────────────────────────────────────────
        // Clips below the enrollment threshold are diarized and matched against
        // existing speakers but will not create new registry entries.
        let canEnroll = durationSeconds >= effectiveEnrollmentMin
        let enrollNote = canEnroll ? "" : " (match-only, clip < \(String(format: "%.0f", effectiveEnrollmentMin))s enrollment threshold)"

        // ── Full pipeline ────────────────────────────────────────────────────
        AudioLog.pipeline.info("\(tag)Diarizing \(String(format: "%.2f", durationSeconds))s (\(audio.count) samples)\(enrollNote)")
        let diarResult = diarizer.diarize(audio: audio, sampleRate: sampleRate, config: config)
        let diarMs = Int(elapsedMs(from: t0))
        AudioLog.pipeline.info("\(tag)Diarization done: \(diarResult.segments.count) segs, \(diarResult.speakerEmbeddings.count) speakers (\(diarMs)ms)")

        // Map local diarization speaker index → registry Speaker + match score (nil if unidentifiable)
        let tResolve = ContinuousClock.now
        let regSizeBefore = await registry.centroidCount
        let speakerCountBefore = await registry.speakerCount
        var localToRegistry: [Int: (speaker: Speaker?, score: Float?)] = [:]
        for (localId, embedding) in diarResult.speakerEmbeddings.enumerated() {
            if canEnroll {
                let speakerSegs = diarResult.segments.filter { $0.speakerId == localId }
                let bestQuality = speakerSegs.map(\.duration).max().map(Double.init) ?? 0.0
                // resolve() creates new speakers only for high-quality (≥ 2s) segments.
                // Low-quality segments with no match return nil rather than a placeholder.
                localToRegistry[localId] = try await registry.resolve(
                    embedding: embedding,
                    qualityScore: bestQuality,
                    threshold: threshold)
            } else {
                // Match-only: identify known speakers without touching the registry.
                localToRegistry[localId] = await registry.match(embedding: embedding, threshold: threshold)
            }
        }
        let resolveMs = Int(elapsedMs(from: tResolve))
        // max(0,...) guards against an unlikely underflow if a speaker was concurrently
        // deleted via the DELETE /registry/speakers/:id endpoint during inference.
        let enrolled = canEnroll ? max(0, await registry.speakerCount - speakerCountBefore) : 0
        AudioLog.pipeline.info("\(tag)Registry resolve done: reg_size=\(regSizeBefore) enrolled=\(enrolled) (\(resolveMs)ms)")

        // Build annotated output
        let tASR = ContinuousClock.now
        var annotated: [AnnotatedSegment] = []
        for seg in diarResult.segments {
            guard let entry = localToRegistry[seg.speakerId] else { continue }
            let start = Double(seg.startTime)
            let end = Double(seg.endTime)

            let transcript = asr.map { model in
                let startSample = Int(start * Double(sampleRate))
                let endSample = min(Int(end * Double(sampleRate)), audio.count)
                let slice = Array(audio[startSample..<endSample])
                return model.transcribe(audio: slice, sampleRate: sampleRate, language: language)
            }

            annotated.append(AnnotatedSegment(
                speaker: entry.speaker,
                bestScore: entry.score,
                startTime: start,
                endTime: end,
                transcriptText: transcript))
        }
        let asrMs = asr != nil ? Int(elapsedMs(from: tASR)) : 0
        let segCount = annotated.count
        if asr != nil {
            AudioLog.pipeline.info("\(tag)ASR done: \(segCount) segments (\(asrMs)ms)")
        }

        let totalMs = Int(elapsedMs(from: t0))
        let dt = diarResult.timings
        let asrNote = asr != nil ? " with transcription" : ""
        let identified = localToRegistry.values.filter { $0.speaker != nil }.count
        AudioLog.pipeline.info("\(tag)Resolved \(identified)/\(localToRegistry.count) speaker(s)\(asrNote) [seg=\(dt.segmentMs)ms emb=\(dt.embedMs)ms cluster=\(dt.clusterMs)ms wins=\(dt.windowCount) embeds=\(dt.embedCount) resolve=\(resolveMs)ms reg_size=\(regSizeBefore) enrolled=\(enrolled) asr=\(asrMs)ms segs=\(segCount) total=\(totalMs)ms]")
        return ProcessedSession(
            segments: annotated,
            timings: PipelineTimings(
                diarize: diarResult.timings,
                resolveMs: resolveMs,
                regSize: regSizeBefore,
                enrolled: enrolled,
                asrMs: asrMs,
                segCount: segCount,
                totalMs: totalMs))
    }

    // MARK: - ASR-Only Path

    /// Run ASR across the entire clip as a single segment. Used when the clip is
    /// too short for speaker recognition.
    private func asrOnlySession(
        audio: [Float],
        sampleRate: Int,
        durationSeconds: Double,
        language: String? = nil,
        regSize: Int,
        t0: ContinuousClock.Instant
    ) async throws -> ProcessedSession {
        guard let asr else {
            return ProcessedSession(
                segments: [],
                timings: PipelineTimings(
                    diarize: .zero, resolveMs: 0, regSize: regSize, enrolled: 0,
                    asrMs: 0, segCount: 0, totalMs: Int(elapsedMs(from: t0))))
        }
        let tASR = ContinuousClock.now
        let transcript = asr.transcribe(audio: audio, sampleRate: sampleRate, language: language)
        let asrMs = Int(elapsedMs(from: tASR))
        let totalMs = Int(elapsedMs(from: t0))
        let segment = AnnotatedSegment(
            speaker: nil,
            bestScore: nil,
            startTime: 0,
            endTime: durationSeconds,
            transcriptText: transcript.isEmpty ? nil : transcript)
        return ProcessedSession(
            segments: [segment],
            timings: PipelineTimings(
                diarize: .zero, resolveMs: 0, regSize: regSize, enrolled: 0,
                asrMs: asrMs, segCount: 1, totalMs: totalMs))
    }
}
