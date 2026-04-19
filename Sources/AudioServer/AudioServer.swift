import Foundation
import Hummingbird
import HummingbirdCore
import HummingbirdWebSocket
import Logging
import NIOCore
import Qwen3ASR
import Qwen3TTS
import CosyVoiceTTS
import PersonaPlex
import SpeechEnhancement
import SpeechVAD
import AudioCommon

// MARK: - Server

public struct AudioServer {
    let state: ModelState
    let config: ServerConfig
    let host: String
    let port: Int
    let logRequests: Bool
    /// Gates concurrent GPU inference on /registry/sessions.
    let inferenceSemaphore: InferenceSemaphore

    public init(
        host: String = "127.0.0.1",
        port: Int = 8080,
        logRequests: Bool = false,
        concurrency: Int = 1,
        config: ServerConfig = ServerConfig()
    ) {
        self.config = config
        self.state = ModelState(config: config)
        self.host = host
        self.port = port
        self.logRequests = logRequests
        self.inferenceSemaphore = InferenceSemaphore(permits: concurrency)
    }

    public func run() async throws {
        let router = buildRouter()
        let state = self.state
        let wsConfig = WebSocketServerConfiguration(maxFrameSize: 1 << 24)  // 16 MB max frame
        let wsServer: HTTPServerBuilder = .http1WebSocketUpgrade(configuration: wsConfig) { head, _, _ in
            let path = head.path ?? ""
            guard path == "/v1/realtime" else { return .dontUpgrade }
            return .upgrade([:]) { inbound, outbound, _ in
                try await handleRealtimeWS(inbound: inbound, outbound: outbound, state: state)
            }
        }
        let app = Application(
            router: router,
            server: wsServer,
            configuration: .init(address: .hostname(host, port: port)))
        try await app.run()
    }

    /// Start a background keep-alive loop that runs a tiny inference every `intervalSeconds`
    /// to prevent macOS from evicting MLX Metal kernels from GPU memory.
    ///
    /// Without this, the first request after a period of GPU idle (typically ~15–30s)
    /// re-uploads and re-dispatches the compute graph, adding several seconds of latency.
    /// A minimal inference (0.1s of silence) is enough to keep the kernels hot.
    ///
    /// Call this after `preloadModels()`. Returns immediately; the loop runs in the background
    /// until the task is cancelled.
    @discardableResult
    public func startKeepAlive(intervalSeconds: Double = 10.0) -> Task<Void, Never> {
        let state = self.state
        let semaphore = self.inferenceSemaphore
        return Task {
            // 0.1 s of silence at 16 kHz — enough to dispatch the compute graph
            let silence = [Float](repeating: 0, count: 1600)
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: .seconds(intervalSeconds))
                } catch {
                    break  // cancelled
                }
                guard !Task.isCancelled else { break }
                // Only warm up models that are already loaded; don't trigger a load.
                // Runs inside the inference semaphore to avoid racing with real requests
                // on the shared GPU command queue — concurrent MLX calls run 4-7× slower.
                // Skip keep-alive when callers are already waiting to avoid joining the queue
                // behind a backlog. Note: there is a TOCTOU window between checking waitingCount
                // and acquiring the permit — a request arriving in that window will wait behind
                // the keep-alive for at most ~100ms (0.1s silence inference), which is acceptable.
                let waiting = await semaphore.waitingCount
                guard waiting == 0 else { continue }
                try? await semaphore.withPermit {
                    guard !Task.isCancelled else { return }
                    if let asr = try? await state.loadedASR() {
                        guard !Task.isCancelled else { return }
                        _ = asr.transcribe(audio: silence, sampleRate: 16000, language: nil)
                    }
                    if let diarizer = try? await state.loadedDiarizer() {
                        guard !Task.isCancelled else { return }
                        _ = diarizer.diarize(audio: silence, sampleRate: 16000)
                    }
                }
            }
        }
    }

    /// Preload the specified models concurrently.
    ///
    /// `models` is a set of names from: `asr`, `tts`, `cosyvoice`, `diarizer`,
    /// `personaplex`, `enhancer`. Pass `["all"]` to load everything.
    public func preloadModels(_ models: Set<String> = ["all"]) async throws {
        let all = models.contains("all")
        try await withThrowingTaskGroup(of: Void.self) { group in
            if all || models.contains("asr") {
                group.addTask { _ = try await self.state.loadASR() }
            }
            if all || models.contains("diarizer") {
                group.addTask { _ = try await self.state.loadDiarizer() }
            }
            if all || models.contains("tts") {
                group.addTask { _ = try await self.state.loadTTS() }
            }
            if all || models.contains("cosyvoice") {
                group.addTask { _ = try await self.state.loadCosyVoice() }
            }
            if all || models.contains("personaplex") {
                group.addTask { _ = try await self.state.loadPersonaPlex() }
            }
            if all || models.contains("enhancer") {
                group.addTask { _ = try await self.state.loadEnhancer() }
            }
            try await group.waitForAll()
        }
    }

    // MARK: - HTTP Routes

    func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()
        let state = self.state

        if logRequests {
            router.add(middleware: TimedRequestLogger<BasicRequestContext>(logLevel: .info))
        }

        addRegistryRoutes(to: router)

        router.get("/health") { _, _ in
            Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(string: "{\"status\":\"ok\"}")))
        }

        // GET /status — per-model readiness. Poll this after startup to know when
        // preloaded models are ready. `ready` is true when all requested models have
        // finished loading. Models that were never requested are omitted.
        router.get("/status") { _, _ in
            let statuses = await state.statuses()
            let ready = await state.isReady()
            let models = statuses.mapValues { $0.rawValue }
            return jsonResponse(["ready": ready, "models": models] as [String: Any])
        }

        router.post("/transcribe") { request, _ in
            let body = try await request.body.collect(upTo: 50 * 1024 * 1024)
            let params = try RequestParams.parse(body, contentType: request.headers[.contentType])

            guard let audioData = params.audioData else {
                return errorResponse("Missing audio data", status: .badRequest)
            }

            let sampleRate = params.int("sample_rate") ?? 16000
            let model = try await state.loadASR()
            let audio = try decodeWAVData(audioData, targetSampleRate: sampleRate)
            let text = model.transcribe(audio: audio, sampleRate: sampleRate)

            return jsonResponse([
                "text": text,
                "duration": round(Double(audio.count) / Double(sampleRate) * 100) / 100
            ] as [String: Any])
        }

        router.post("/speak") { request, _ in
            let body = try await request.body.collect(upTo: 1024 * 1024)
            let params = try RequestParams.parse(body, contentType: request.headers[.contentType])

            guard let text = params.text else {
                return errorResponse("Missing 'text' field", status: .badRequest)
            }

            let engine = params.string("engine") ?? "cosyvoice"
            let language = params.string("language") ?? "english"

            let samples: [Float]

            if engine == "qwen3" {
                let model = try await state.loadTTS()
                samples = model.synthesize(text: text, language: language)
            } else {
                let model = try await state.loadCosyVoice()
                samples = model.synthesize(text: text, language: language)
            }

            let wavData = try encodeWAV(samples: samples, sampleRate: 24000)
            return Response(
                status: .ok,
                headers: [.contentType: "audio/wav"],
                body: .init(byteBuffer: .init(data: wavData)))
        }

        router.post("/respond") { request, _ in
            let body = try await request.body.collect(upTo: 50 * 1024 * 1024)
            let params = try RequestParams.parse(body, contentType: request.headers[.contentType])

            guard let audioData = params.audioData else {
                return errorResponse("Missing audio data", status: .badRequest)
            }

            let voiceName = params.string("voice") ?? "NATM0"
            let maxSteps = params.int("max_steps") ?? 200

            guard let voice = PersonaPlexVoice(rawValue: voiceName) else {
                return errorResponse("Unknown voice: \(voiceName)", status: .badRequest)
            }

            let model = try await state.loadPersonaPlex()
            let audio = try decodeWAVData(audioData, targetSampleRate: 24000)
            let result = model.respond(
                userAudio: audio,
                voice: voice,
                maxSteps: maxSteps)

            var transcript: String?
            if let dec = await state.getSpmDecoder(), !result.textTokens.isEmpty {
                transcript = dec.decode(result.textTokens)
            }

            let wavData = try encodeWAV(samples: result.audio, sampleRate: 24000)
            let duration = Double(result.audio.count) / 24000.0

            if params.string("format") == "json" {
                var json: [String: Any] = [
                    "duration": round(duration * 100) / 100,
                    "text_tokens": result.textTokens.count
                ]
                if let t = transcript { json["transcript"] = t }
                json["audio_base64"] = wavData.base64EncodedString()
                return jsonResponse(json)
            }

            return Response(
                status: .ok,
                headers: [.contentType: "audio/wav"],
                body: .init(byteBuffer: .init(data: wavData)))
        }

        router.post("/enhance") { request, _ in
            let body = try await request.body.collect(upTo: 50 * 1024 * 1024)
            let params = try RequestParams.parse(body, contentType: request.headers[.contentType])

            guard let audioData = params.audioData else {
                return errorResponse("Missing audio data", status: .badRequest)
            }

            let enhancer = try await state.loadEnhancer()
            let audio = try decodeWAVData(audioData, targetSampleRate: 48000)
            let enhanced = try enhancer.enhance(audio: audio, sampleRate: 48000)

            let wavData = try encodeWAV(samples: enhanced, sampleRate: 48000)
            return Response(
                status: .ok,
                headers: [.contentType: "audio/wav"],
                body: .init(byteBuffer: .init(data: wavData)))
        }

        return router
    }
}

// MARK: - Lazy Model State

/// Actor-isolated lazy model loader with per-model status tracking.
///
/// Each `load*()` method creates a `Task` on the first call and stores it. Concurrent
/// callers that arrive before loading finishes await the same task rather than starting
/// independent loads. Actor isolation guarantees the task-reference check and set are
/// atomic (no suspension point between them), so no explicit locking is needed.
actor ModelState {

    // MARK: Config

    private let config: ServerConfig

    init(config: ServerConfig = ServerConfig()) {
        self.config = config
    }

    // MARK: Status

    enum ModelStatus: String {
        case idle       // never requested
        case loading    // task in flight
        case ready      // loaded successfully
        case error      // load failed
    }

    private var status: [String: ModelStatus] = [:]

    /// Snapshot of all model statuses. Only models that have been requested appear.
    func statuses() -> [String: ModelStatus] { status }

    /// `true` when every requested model has finished loading (ready or error).
    func isReady() -> Bool {
        !status.isEmpty && status.values.allSatisfy { $0 == .ready || $0 == .error }
    }

    // MARK: Tasks

    private var asrTask: Task<Qwen3ASRModel, Error>?
    private var ttsTask: Task<Qwen3TTSModel, Error>?
    private var cosyvoiceTask: Task<CosyVoiceTTSModel, Error>?
    private var personaplexTask: Task<PersonaPlexModel, Error>?
    private var enhancerTask: Task<SpeechEnhancer, Error>?
    private var diarizerTask: Task<DiarizationPipeline, Error>?
    private var spmDecoder: SentencePieceDecoder?

    func getSpmDecoder() -> SentencePieceDecoder? { spmDecoder }

    // MARK: Already-loaded accessors (for keep-alive — never trigger a load)

    func loadedASR() async throws -> Qwen3ASRModel? {
        guard let task = asrTask else { return nil }
        return try await task.value
    }

    func loadedDiarizer() async throws -> DiarizationPipeline? {
        guard let task = diarizerTask else { return nil }
        return try await task.value
    }

    // MARK: Loaders

    func loadASR() async throws -> Qwen3ASRModel {
        if let task = asrTask { return try await task.value }
        status["asr"] = .loading
        let modelId = config.asrModelId
        let task = Task {
            do {
                let m = try await Qwen3ASRModel.fromPretrained(modelId: modelId, progressHandler: logProgress)
                self.setStatus("asr", .ready)
                return m
            } catch {
                self.setStatus("asr", .error)
                throw error
            }
        }
        asrTask = task
        print("[server] Loading Qwen3-ASR (\(modelId))...")
        return try await task.value
    }

    func loadTTS() async throws -> Qwen3TTSModel {
        if let task = ttsTask { return try await task.value }
        status["tts"] = .loading
        let task = Task {
            do {
                let m = try await Qwen3TTSModel.fromPretrained(progressHandler: logProgress)
                self.setStatus("tts", .ready)
                return m
            } catch {
                self.setStatus("tts", .error)
                throw error
            }
        }
        ttsTask = task
        print("[server] Loading Qwen3-TTS...")
        return try await task.value
    }

    func loadCosyVoice() async throws -> CosyVoiceTTSModel {
        if let task = cosyvoiceTask { return try await task.value }
        status["cosyvoice"] = .loading
        let task = Task {
            do {
                let m = try await CosyVoiceTTSModel.fromPretrained(progressHandler: logProgress)
                self.setStatus("cosyvoice", .ready)
                return m
            } catch {
                self.setStatus("cosyvoice", .error)
                throw error
            }
        }
        cosyvoiceTask = task
        print("[server] Loading CosyVoice...")
        return try await task.value
    }

    func loadPersonaPlex() async throws -> PersonaPlexModel {
        if let task = personaplexTask { return try await task.value }
        status["personaplex"] = .loading
        let task = Task {
            do {
                let m = try await PersonaPlexModel.fromPretrained(progressHandler: logProgress)
                self.setStatus("personaplex", .ready)
                return m
            } catch {
                self.setStatus("personaplex", .error)
                throw error
            }
        }
        personaplexTask = task
        print("[server] Loading PersonaPlex 7B...")
        let model = try await task.value
        if spmDecoder == nil {
            do {
                let cacheDir = try HuggingFaceDownloader.getCacheDirectory(
                    for: "aufklarer/PersonaPlex-7B-MLX-4bit")
                let spmPath = cacheDir.appendingPathComponent("tokenizer_spm_32k_3.model").path
                if FileManager.default.fileExists(atPath: spmPath) {
                    spmDecoder = try SentencePieceDecoder(modelPath: spmPath)
                }
            } catch {}
        }
        return model
    }

    func loadEnhancer() async throws -> SpeechEnhancer {
        if let task = enhancerTask { return try await task.value }
        status["enhancer"] = .loading
        let task = Task {
            do {
                let m = try await SpeechEnhancer.fromPretrained(progressHandler: logProgress)
                self.setStatus("enhancer", .ready)
                return m
            } catch {
                self.setStatus("enhancer", .error)
                throw error
            }
        }
        enhancerTask = task
        print("[server] Loading DeepFilterNet3...")
        return try await task.value
    }

    func loadDiarizer() async throws -> DiarizationPipeline {
        if let task = diarizerTask { return try await task.value }
        status["diarizer"] = .loading
        let segModelId = config.diarizationSegModelId
        let embModelId = config.embeddingModelId
        let embEngine = config.resolvedEmbeddingEngine
        let useVADFilter = config.useVADFilter
        let task = Task {
            do {
                let m = try await DiarizationPipeline.fromPretrained(
                    segModelId: segModelId,
                    embModelId: embModelId,
                    embeddingEngine: embEngine,
                    useVADFilter: useVADFilter,
                    progressHandler: logProgress)
                self.setStatus("diarizer", .ready)
                return m
            } catch {
                self.setStatus("diarizer", .error)
                throw error
            }
        }
        diarizerTask = task
        print("[server] Loading diarization pipeline (seg=\(segModelId) emb=\(embEngine.rawValue))...")
        return try await task.value
    }

    private func setStatus(_ name: String, _ s: ModelStatus) { status[name] = s }
}

private func logProgress(_ progress: Double, _ status: String) {
    print("  [\(Int(progress * 100))%] \(status)")
}

// MARK: - Request Logging Middleware

/// Logs method, path, status code, and elapsed time after each response.
struct TimedRequestLogger<Context: RequestContext>: RouterMiddleware {
    let logLevel: Logger.Level

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        let start = ContinuousClock.now
        let response = try await next(request, context)
        let elapsed = ContinuousClock.now - start
        let ms = Int(Double(elapsed.components.seconds) * 1000
                     + Double(elapsed.components.attoseconds) / 1e15)
        context.logger.log(
            level: logLevel,
            "\(request.method.rawValue) \(request.uri.path) \(response.status.code) (\(ms)ms)"
        )
        return response
    }
}

// MARK: - OpenAI Realtime API Handler

/// Per-connection session state for the OpenAI Realtime protocol.
private final class RealtimeSession {
    var engine: String = "cosyvoice"
    var language: String = "english"
    var inputAudioBuffer = Data()
    var inputSampleRate: Int = 24000
}

/// Handle /v1/realtime: OpenAI Realtime API compatible protocol.
/// All messages are JSON with a "type" field. Audio is base64-encoded PCM16 24kHz.
func handleRealtimeWS(
    inbound: WebSocketInboundStream,
    outbound: WebSocketOutboundWriter,
    state: ModelState
) async throws {
    let session = RealtimeSession()
    let sessionId = UUID().uuidString

    // Send session.created
    try await outbound.write(.text(formatJSON([
        "type": "session.created",
        "session": [
            "id": sessionId,
            "model": "qwen3-speech",
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16"
        ]
    ] as [String: Any])))

    for try await message in inbound.messages(maxSize: 50 * 1024 * 1024) {
        guard case .text(let string) = message else { continue }
        guard let jsonData = string.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
              let eventType = json["type"] as? String else {
            try await sendRealtimeError(outbound: outbound, message: "Invalid message format")
            continue
        }

        switch eventType {

        case "session.update":
            if let sessionConfig = json["session"] as? [String: Any] {
                if let engine = sessionConfig["engine"] as? String {
                    session.engine = engine
                }
                if let lang = sessionConfig["language"] as? String {
                    session.language = lang
                }
                if let fmt = sessionConfig["input_audio_format"] as? String, fmt == "pcm16" {
                    session.inputSampleRate = 24000
                }
            }
            try await outbound.write(.text(formatJSON([
                "type": "session.updated",
                "session": [
                    "id": sessionId,
                    "engine": session.engine,
                    "language": session.language,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16"
                ]
            ] as [String: Any])))

        case "input_audio_buffer.append":
            guard let audioB64 = json["audio"] as? String,
                  let audioData = Data(base64Encoded: audioB64) else {
                try await sendRealtimeError(outbound: outbound, message: "Missing or invalid 'audio' field")
                continue
            }
            session.inputAudioBuffer.append(audioData)

        case "input_audio_buffer.clear":
            session.inputAudioBuffer.removeAll()
            try await outbound.write(.text(formatJSON([
                "type": "input_audio_buffer.cleared"
            ])))

        case "input_audio_buffer.commit":
            let audioData = session.inputAudioBuffer
            session.inputAudioBuffer.removeAll()

            guard !audioData.isEmpty else {
                try await sendRealtimeError(outbound: outbound, message: "Audio buffer is empty")
                continue
            }

            let itemId = UUID().uuidString
            try await outbound.write(.text(formatJSON([
                "type": "input_audio_buffer.committed",
                "item_id": itemId
            ])))

            // Transcribe: PCM16 24kHz → resample to 16kHz for ASR
            let floats = pcm16LEToFloat(audioData)
            let audio16k = resample(floats, from: session.inputSampleRate, to: 16000)
            let model = try await state.loadASR()
            let text = model.transcribe(audio: audio16k, sampleRate: 16000)

            let responseId = UUID().uuidString
            try await outbound.write(.text(formatJSON([
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": itemId,
                "transcript": text
            ])))

            // Also emit as a response for clients expecting response.* events
            try await outbound.write(.text(formatJSON([
                "type": "response.created",
                "response": ["id": responseId, "status": "in_progress"]
            ] as [String: Any])))
            try await outbound.write(.text(formatJSON([
                "type": "response.audio_transcript.delta",
                "response_id": responseId,
                "delta": text
            ])))
            try await outbound.write(.text(formatJSON([
                "type": "response.audio_transcript.done",
                "response_id": responseId,
                "transcript": text
            ])))
            try await outbound.write(.text(formatJSON([
                "type": "response.done",
                "response": ["id": responseId, "status": "completed"]
            ] as [String: Any])))

        case "response.create":
            let input = json["response"] as? [String: Any]
            let instructions = input?["instructions"] as? String
            let modalities = input?["modalities"] as? [String] ?? ["audio", "text"]
            let responseId = UUID().uuidString

            // If there's text to speak (from instructions or input items)
            var textToSpeak: String?

            if let instructions = instructions, !instructions.isEmpty {
                textToSpeak = instructions
            }

            // Check for input items with text content
            if textToSpeak == nil, let inputItems = input?["input"] as? [[String: Any]] {
                for item in inputItems {
                    if let content = item["content"] as? [[String: Any]] {
                        for part in content {
                            if part["type"] as? String == "input_text",
                               let text = part["text"] as? String {
                                textToSpeak = text
                            }
                        }
                    }
                }
            }

            // Also check conversation.item.create pattern — text in input
            if textToSpeak == nil, let text = input?["text"] as? String {
                textToSpeak = text
            }

            guard let text = textToSpeak else {
                try await sendRealtimeError(outbound: outbound, message: "No text to synthesize")
                continue
            }

            try await outbound.write(.text(formatJSON([
                "type": "response.created",
                "response": ["id": responseId, "status": "in_progress"]
            ] as [String: Any])))

            // Stream TTS audio as base64 PCM16 24kHz chunks
            let engine = (input?["engine"] as? String) ?? session.engine
            let language = (input?["language"] as? String) ?? session.language
            var totalSamples = 0

            if engine == "qwen3" {
                let model = try await state.loadTTS()
                let stream = model.synthesizeStream(text: text, language: language)
                for try await chunk in stream {
                    if !chunk.samples.isEmpty {
                        totalSamples += chunk.samples.count
                        let pcm = floatToPCM16LE(chunk.samples)
                        try await outbound.write(.text(formatJSON([
                            "type": "response.audio.delta",
                            "response_id": responseId,
                            "delta": pcm.base64EncodedString()
                        ])))
                    }
                }
            } else {
                let model = try await state.loadCosyVoice()
                let stream = model.synthesizeStream(text: text, language: language)
                for try await chunk in stream {
                    if !chunk.samples.isEmpty {
                        totalSamples += chunk.samples.count
                        let pcm = floatToPCM16LE(chunk.samples)
                        try await outbound.write(.text(formatJSON([
                            "type": "response.audio.delta",
                            "response_id": responseId,
                            "delta": pcm.base64EncodedString()
                        ])))
                    }
                }
            }

            if modalities.contains("text") {
                try await outbound.write(.text(formatJSON([
                    "type": "response.audio_transcript.done",
                    "response_id": responseId,
                    "transcript": text
                ])))
            }

            try await outbound.write(.text(formatJSON([
                "type": "response.audio.done",
                "response_id": responseId
            ])))

            let duration = Double(totalSamples) / 24000.0
            try await outbound.write(.text(formatJSON([
                "type": "response.done",
                "response": [
                    "id": responseId,
                    "status": "completed",
                    "usage": [
                        "total_tokens": 0,
                        "output_tokens": 0
                    ],
                    "output": [
                        ["type": "audio", "duration": round(duration * 100) / 100, "sample_rate": 24000]
                    ]
                ]
            ] as [String: Any])))

        case "conversation.item.create":
            // Accept text items for TTS via response.create flow
            if let item = json["item"] as? [String: Any],
               let content = item["content"] as? [[String: Any]] {
                for part in content {
                    if part["type"] as? String == "input_text" || part["type"] as? String == "text",
                       let _ = part["text"] as? String {
                        try await outbound.write(.text(formatJSON([
                            "type": "conversation.item.created",
                            "item": item
                        ] as [String: Any])))
                    }
                }
            }

        default:
            try await sendRealtimeError(outbound: outbound,
                message: "Unknown event type: \(eventType)")
        }
    }
}

private func sendRealtimeError(outbound: WebSocketOutboundWriter, message: String) async throws {
    try await outbound.write(.text(formatJSON([
        "type": "error",
        "error": ["type": "invalid_request_error", "message": message]
    ] as [String: Any])))
}

/// Resample audio via AVAudioConverter (delegates to AudioFileLoader).
func resample(_ samples: [Float], from sourceSR: Int, to targetSR: Int) -> [Float] {
    AudioFileLoader.resample(samples, from: sourceSR, to: targetSR)
}

// MARK: - Request Parsing

struct RequestParams {
    var audioData: Data?
    var text: String?
    var fields: [String: String] = [:]

    func string(_ key: String) -> String? { fields[key] }
    func int(_ key: String) -> Int? { fields[key].flatMap(Int.init) }

    static func parse(_ body: ByteBuffer, contentType: String?) throws -> RequestParams {
        var params = RequestParams()

        if let ct = contentType, ct.contains("application/json") {
            let data = Data(buffer: body)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                if let text = json["text"] as? String { params.text = text }
                if let b64 = json["audio_base64"] as? String {
                    params.audioData = Data(base64Encoded: b64)
                }
                for (k, v) in json {
                    if let s = v as? String { params.fields[k] = s }
                    else if let n = v as? Int { params.fields[k] = String(n) }
                    else if let n = v as? Double { params.fields[k] = String(n) }
                }
            }
            return params
        }

        // Raw audio body (WAV)
        let data = Data(buffer: body)
        if data.count > 44 {
            params.audioData = data
        }
        return params
    }
}

// MARK: - Audio Encoding/Decoding

func decodeWAVData(_ data: Data, targetSampleRate: Int) throws -> [Float] {
    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString + ".wav")
    try data.write(to: tmpURL)
    defer { try? FileManager.default.removeItem(at: tmpURL) }
    return try AudioFileLoader.load(url: tmpURL, targetSampleRate: targetSampleRate)
}

func encodeWAV(samples: [Float], sampleRate: Int) throws -> Data {
    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString + ".wav")
    try WAVWriter.write(samples: samples, sampleRate: sampleRate, to: tmpURL)
    defer { try? FileManager.default.removeItem(at: tmpURL) }
    return try Data(contentsOf: tmpURL)
}

// MARK: - Response Helpers

func jsonResponse(_ dict: [String: Any]) -> Response {
    let data = (try? JSONSerialization.data(
        withJSONObject: dict, options: [.prettyPrinted, .sortedKeys])) ?? Data()
    return Response(
        status: .ok,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: .init(data: data)))
}

func errorResponse(_ message: String, status: HTTPResponse.Status) -> Response {
    let data = (try? JSONSerialization.data(
        withJSONObject: ["error": message], options: [])) ?? Data()
    return Response(
        status: status,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: .init(data: data)))
}

// MARK: - PCM Conversion

func pcm16LEToFloat(_ data: Data) -> [Float] {
    let sampleCount = data.count / 2
    var result = [Float](repeating: 0, count: sampleCount)
    data.withUnsafeBytes { raw in
        let int16s = raw.bindMemory(to: Int16.self)
        for i in 0..<sampleCount {
            result[i] = Float(Int16(littleEndian: int16s[i])) / 32768.0
        }
    }
    return result
}

func floatToPCM16LE(_ samples: [Float]) -> Data {
    var data = Data(count: samples.count * 2)
    data.withUnsafeMutableBytes { raw in
        let int16s = raw.bindMemory(to: Int16.self)
        for i in 0..<samples.count {
            let clamped = max(-1.0, min(1.0, samples[i]))
            int16s[i] = Int16(clamped * 32767.0).littleEndian
        }
    }
    return data
}

func formatJSON(_ dict: [String: Any]) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys]) else {
        return "{}"
    }
    return String(data: data, encoding: .utf8) ?? "{}"
}
