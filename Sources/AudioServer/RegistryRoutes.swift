import Foundation
import Hummingbird
import NIOCore
import AudioCommon
import SpeakerRegistry

// MARK: - Registry Route Registration

extension AudioServer {
    /// Attach all /registry/* routes to the router.
    func addRegistryRoutes(to router: Router<BasicRequestContext>) {
        let registry = openOrCreateRegistry()

        let group = router.group("registry")

        // MARK: Sessions

        // POST /registry/sessions[?threshold=0.65]
        // Body: raw WAV bytes or multipart/form-data with a "file" field.
        // Diarizes the audio, resolves speakers against the registry, and returns the result.
        // Optional query param `threshold` overrides the registry's default similarity threshold.
        group.post("sessions") { request, _ in
            let threshold = request.uri.queryParameters.get("threshold").flatMap(Float.init)

            let body = try await request.body.collect(upTo: 100 * 1024 * 1024)
            let audioData = try extractAudioData(from: body, contentType: request.headers[.contentType])

            let tmpURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + ".wav")
            defer { try? FileManager.default.removeItem(at: tmpURL) }
            try Data(buffer: audioData).write(to: tmpURL)

            let audio = try AudioFileLoader.load(url: tmpURL, targetSampleRate: 16000)
            let diarizer = try await self.state.loadDiarizer()
            let asr = try await self.state.loadASR()
            let pipeline = PipelineSession(diarizer: diarizer, registry: registry, asr: asr)
            let result = try await pipeline.process(audioURL: tmpURL, audio: audio, threshold: threshold)

            return jsonResponse(ProcessedSessionResponse(result).json)
        }

        // MARK: Speakers

        // GET /registry/speakers
        group.get("speakers") { _, _ in
            let speakers = await registry.speakers()
            return jsonResponse(["speakers": speakers.map(SpeakerResponse.init).map(\.json)])
        }

        // GET /registry/speakers/:id
        group.get("speakers/:id") { _, context in
            let id = try requireInt64(context.parameters.get("id"))
            guard let speaker = await registry.speaker(id: id) else {
                return errorResponse("Speaker \(id) not found", status: .notFound)
            }
            return jsonResponse(SpeakerResponse(speaker).json)
        }

        // PATCH /registry/speakers/:id
        // Body: { "displayName": "Alice", "notes": "..." }
        group.patch("speakers/:id") { request, context in
            let id = try requireInt64(context.parameters.get("id"))
            let body = try await request.body.collect(upTo: 64 * 1024)
            let json = try requireJSON(body)

            if let name = json["displayName"] as? String {
                try await registry.label(speakerId: id, displayName: name)
            }
            if let notes = json["notes"] as? String {
                try await registry.updateNotes(speakerId: id, notes: notes)
            }

            guard let speaker = await registry.speaker(id: id) else {
                return errorResponse("Speaker \(id) not found", status: .notFound)
            }
            return jsonResponse(SpeakerResponse(speaker).json)
        }

        // POST /registry/speakers/merge
        // Body: { "src": 3, "dst": 7 }
        group.post("speakers/merge") { request, _ in
            let body = try await request.body.collect(upTo: 64 * 1024)
            let json = try requireJSON(body)
            guard let src = (json["src"] as? Int).map(Int64.init),
                  let dst = (json["dst"] as? Int).map(Int64.init) else {
                return errorResponse("Body must include integer 'src' and 'dst'", status: .badRequest)
            }
            try await registry.merge(src: src, into: dst)
            guard let speaker = await registry.speaker(id: dst) else {
                return errorResponse("Speaker \(dst) not found after merge", status: .internalServerError)
            }
            return jsonResponse(SpeakerResponse(speaker).json)
        }

        // DELETE /registry/speakers
        // Wipes all speakers and centroids, resetting the registry to empty.
        group.delete("speakers") { _, _ in
            try await registry.reset()
            return Response(status: .noContent)
        }

        // DELETE /registry/speakers/:id
        group.delete("speakers/:id") { _, context in
            let id = try requireInt64(context.parameters.get("id"))
            try await registry.deleteSpeaker(id: id)
            return Response(status: .noContent)
        }
    }

    private func openOrCreateRegistry() -> SpeakerRegistry {
        guard let reg = try? SpeakerRegistry.open(similarityThreshold: 0.75) else {
            fatalError("Failed to open speaker registry at default path")
        }
        return reg
    }
}

// MARK: - Response Types

private struct ProcessedSessionResponse {
    let result: ProcessedSession

    init(_ result: ProcessedSession) { self.result = result }

    var json: [String: Any] {
        [
            "num_speakers": result.numSpeakers,
            "segments": result.segments.map { seg -> [String: Any] in
                var d: [String: Any] = [
                    "speaker_id": seg.speaker.id ?? -1,
                    "speaker_label": seg.speaker.label,
                    "start": seg.startTime,
                    "end": seg.endTime,
                    "duration": seg.duration,
                ]
                if let t = seg.transcriptText { d["transcript"] = t }
                return d
            },
        ]
    }
}

private struct SpeakerResponse {
    let speaker: Speaker

    init(_ speaker: Speaker) { self.speaker = speaker }

    var json: [String: Any] {
        var d: [String: Any] = [
            "id": speaker.id ?? -1,
            "label": speaker.label,
            "is_labeled": speaker.isLabeled,
        ]
        if let name = speaker.displayName { d["display_name"] = name }
        if let notes = speaker.notes { d["notes"] = notes }
        return d
    }
}

// MARK: - Request Helpers

private func requireInt64(_ string: String?) throws -> Int64 {
    guard let s = string, let id = Int64(s) else {
        throw HTTPError(.badRequest, message: "Invalid or missing id parameter")
    }
    return id
}

private func requireJSON(_ buffer: ByteBuffer) throws -> [String: Any] {
    let data = Data(buffer: buffer)
    guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw HTTPError(.badRequest, message: "Request body must be valid JSON")
    }
    return json
}

/// Extract audio bytes from either a raw WAV body or a multipart/form-data "file" field.
private func extractAudioData(from buffer: ByteBuffer, contentType: String?) throws -> ByteBuffer {
    let ct = contentType ?? ""
    if ct.contains("multipart/form-data") {
        guard let boundary = ct.components(separatedBy: "boundary=").last else {
            throw HTTPError(.badRequest, message: "Missing multipart boundary")
        }
        let data = Data(buffer: buffer)
        if let extracted = extractMultipartField(named: "file", from: data, boundary: boundary) {
            return ByteBuffer(data: extracted)
        }
        throw HTTPError(.badRequest, message: "No 'file' field found in multipart body")
    }
    // Assume raw WAV bytes
    return buffer
}

/// Minimal multipart parser — extracts the body bytes of the named field.
private func extractMultipartField(named name: String, from data: Data, boundary: String) -> Data? {
    guard let boundaryData = "--\(boundary)".data(using: .utf8),
          let crlf = "\r\n".data(using: .utf8),
          let doubleCRLF = "\r\n\r\n".data(using: .utf8) else { return nil }

    var searchRange = data.startIndex..<data.endIndex
    while let boundaryRange = data.range(of: boundaryData, in: searchRange) {
        let headerStart = boundaryRange.upperBound
        guard let headerEnd = data.range(of: doubleCRLF, in: headerStart..<data.endIndex) else { break }
        let headerData = data[headerStart..<headerEnd.lowerBound]
        let headers = String(data: headerData, encoding: .utf8) ?? ""

        if headers.contains("name=\"\(name)\"") || headers.contains("name=\"\(name)\"") {
            let bodyStart = headerEnd.upperBound
            if let nextBoundary = data.range(of: boundaryData, in: bodyStart..<data.endIndex) {
                let bodyEnd = nextBoundary.lowerBound
                let trimEnd = data[bodyStart..<bodyEnd].hasSuffix(crlf)
                    ? data.index(bodyEnd, offsetBy: -crlf.count)
                    : bodyEnd
                return data[bodyStart..<trimEnd]
            }
        }
        searchRange = boundaryRange.upperBound..<data.endIndex
    }
    return nil
}

private extension Data {
    func hasSuffix(_ suffix: Data) -> Bool {
        count >= suffix.count && self[(count - suffix.count)...] == suffix
    }
}
