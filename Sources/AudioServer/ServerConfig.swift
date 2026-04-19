import Foundation
import SpeechVAD

/// Centralised server configuration for model selection and pipeline tuning.
///
/// All fields default to the values previously hardcoded in the server. Override them
/// by passing a JSON file path to `--config` on the CLI, or by setting the
/// `SERVER_CONFIG` environment variable to the file path.
///
/// Example config file:
/// ```json
/// {
///   "asrModelId": "aufklarer/Qwen3-ASR-1.7B-MLX-8bit",
///   "embeddingEngine": "coreml",
///   "similarityThreshold": 0.72,
///   "minimumDurationForDiarization": 1.0,
///   "minimumDurationForEnrollment": 10.0
/// }
/// ```
public struct ServerConfig: Codable, Sendable {

    // MARK: - ASR

    /// HuggingFace model ID for the ASR model used in `/registry/sessions` and `/transcribe`.
    public var asrModelId: String = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"

    // MARK: - Diarization

    /// HuggingFace model ID for the Pyannote segmentation model.
    public var diarizationSegModelId: String = PyannoteVADModel.defaultModelId

    /// HuggingFace model ID for the WeSpeaker embedding model.
    /// `nil` picks the engine-appropriate default (`WeSpeakerModel.defaultModelId` for MLX,
    /// `WeSpeakerModel.defaultCoreMLModelId` for CoreML).
    public var embeddingModelId: String? = nil

    /// Backend for WeSpeaker speaker embeddings: `"mlx"` (GPU) or `"coreml"` (Neural Engine).
    public var embeddingEngine: String = "mlx"

    /// Apply an additional Silero VAD pre-filter during diarization to suppress non-speech frames.
    public var useVADFilter: Bool = false

    // MARK: - Speaker matching

    /// Minimum cosine similarity to count as a speaker match (0–1). Increase for stricter
    /// matching; decrease if known speakers are being missed.
    public var similarityThreshold: Float = 0.75

    /// Clips shorter than this (seconds) skip diarization entirely — ASR still runs.
    public var minimumDurationForDiarization: Double = 1.0

    /// Clips shorter than this (seconds) run diarization and matching but will not enroll
    /// new speakers into the registry.
    public var minimumDurationForEnrollment: Double = 10.0

    // MARK: - Init

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = ServerConfig()
        asrModelId                   = try c.decodeIfPresent(String.self,  forKey: .asrModelId)                   ?? defaults.asrModelId
        diarizationSegModelId        = try c.decodeIfPresent(String.self,  forKey: .diarizationSegModelId)        ?? defaults.diarizationSegModelId
        embeddingModelId             = try c.decodeIfPresent(String?.self, forKey: .embeddingModelId)             ?? defaults.embeddingModelId
        embeddingEngine              = try c.decodeIfPresent(String.self,  forKey: .embeddingEngine)              ?? defaults.embeddingEngine
        useVADFilter                 = try c.decodeIfPresent(Bool.self,    forKey: .useVADFilter)                 ?? defaults.useVADFilter
        similarityThreshold          = try c.decodeIfPresent(Float.self,   forKey: .similarityThreshold)          ?? defaults.similarityThreshold
        minimumDurationForDiarization = try c.decodeIfPresent(Double.self, forKey: .minimumDurationForDiarization) ?? defaults.minimumDurationForDiarization
        minimumDurationForEnrollment  = try c.decodeIfPresent(Double.self, forKey: .minimumDurationForEnrollment)  ?? defaults.minimumDurationForEnrollment
    }

    // MARK: - Helpers

    /// Resolve the `embeddingEngine` string to a `WeSpeakerEngine`.
    /// Unrecognised values fall back to `.mlx` with a warning printed to stdout.
    public var resolvedEmbeddingEngine: WeSpeakerEngine {
        if let engine = WeSpeakerEngine(rawValue: embeddingEngine) { return engine }
        print("[ServerConfig] Warning: unknown embeddingEngine '\(embeddingEngine)', falling back to 'mlx'")
        return .mlx
    }

    // MARK: - Loading

    /// Load a `ServerConfig` from a JSON file.
    ///
    /// Resolution order:
    /// 1. `path` argument (e.g. from `--config`)
    /// 2. `SERVER_CONFIG` environment variable
    /// 3. Default `ServerConfig()` if neither is set
    ///
    /// - Parameter path: explicit file path, or `nil` to fall through to the env var.
    /// - Throws: file read or JSON decoding errors if a path is resolved but the file is invalid.
    public static func load(from path: String? = nil) throws -> ServerConfig {
        let resolvedPath = path ?? ProcessInfo.processInfo.environment["SERVER_CONFIG"]
        guard let resolvedPath else { return ServerConfig() }
        let url = URL(fileURLWithPath: resolvedPath)
        let data = try Data(contentsOf: url)
        let config = try JSONDecoder().decode(ServerConfig.self, from: data)
        print("[ServerConfig] Loaded from \(resolvedPath)")
        return config
    }
}
