# Speaker Audio Samples API

## Problem

The registry currently stores only centroids — there is no way for a client to retrieve audio of an identified speaker. Without playable voice clips, end users have no means of matching a placeholder like `Speaker_3` to a real person. This makes the speaker naming and merge workflows nearly unusable in practice.

## Proposed Solution

Capture a small set of representative audio clips per speaker during `POST /registry/sessions` processing and expose them via two new read-only endpoints. Clips are stored on disk alongside the existing registry JSON; metadata is persisted in `RegistryStore`.

---

## Storage

### Directory layout

```
~/Library/Caches/qwen3-speech/
├── speaker-registry.json          # existing
└── speaker-samples/
    ├── 1/
    │   ├── a3f2c1d0.wav
    │   └── b7e4a209.wav
    └── 4/
        └── f1c38d72.wav
```

WAV files are 16 kHz mono (matching the pipeline's native sample rate). File names are UUID strings without extension prefix. The directory for a speaker is named by its `Int64` ID.

### New model: `SpeakerSample`

Add to `Sources/SpeakerRegistry/Model.swift`:

```swift
public struct SpeakerSample: Codable, Sendable {
    public var id: String               // UUID string, also the WAV filename stem
    public var speakerId: Int64
    public var durationSeconds: Double
    public var createdAt: Date
}
```

### Updated `RegistryStore`

Add a `samples` array to the existing store so metadata is flushed atomically with the rest of the registry state:

```swift
private struct RegistryStore: Codable {
    var speakers: [Speaker] = []
    var centroids: [SpeakerCentroid] = []
    var samples: [SpeakerSample] = []   // new
    var nextId: Int64 = 1
}
```

**Cap**: keep at most **5 samples per speaker**, retaining the longest by duration. When a new clip would exceed the cap, discard the shortest existing one and delete its WAV file.

---

## Registry Actor Changes

Add to `Sources/SpeakerRegistry/SpeakerRegistry.swift`:

```swift
/// Save a raw audio slice as a representative sample for `speakerId`.
/// Only persists if `durationSeconds >= 2.0` (matches centroid quality threshold).
/// Enforces a per-speaker cap of 5, dropping the shortest clip if exceeded.
public func addSample(
    speakerId: Int64,
    audio: [Float],
    durationSeconds: Double,
    sampleRate: Int = 16000
) throws {
    guard durationSeconds >= 2.0 else { return }

    let sampleId = UUID().uuidString
    let dir = samplesDirectory(for: speakerId)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let fileURL = dir.appendingPathComponent(sampleId + ".wav")

    try WAVWriter.write(audio: audio, sampleRate: sampleRate, to: fileURL)

    var newSample = SpeakerSample(
        id: sampleId,
        speakerId: speakerId,
        durationSeconds: durationSeconds,
        createdAt: .now)

    store.samples.append(newSample)

    // Enforce cap: keep the 5 longest, delete the rest
    let speakerSamples = store.samples
        .filter { $0.speakerId == speakerId }
        .sorted { $0.durationSeconds > $1.durationSeconds }
    if speakerSamples.count > 5 {
        let toRemove = speakerSamples.dropFirst(5)
        for s in toRemove {
            try? FileManager.default.removeItem(at: sampleFileURL(s))
        }
        let removeIds = Set(toRemove.map(\.id))
        store.samples.removeAll { removeIds.contains($0.id) }
    }

    try save()
}

/// Return sample metadata for a speaker, sorted longest-first.
public func samples(forSpeaker id: Int64) -> [SpeakerSample] {
    store.samples
        .filter { $0.speakerId == id }
        .sorted { $0.durationSeconds > $1.durationSeconds }
}

/// Return the file URL for a sample, or nil if not found.
public func sampleFileURL(id: String, speakerId: Int64) -> URL? {
    let url = samplesDirectory(for: speakerId).appendingPathComponent(id + ".wav")
    return FileManager.default.fileExists(atPath: url.path) ? url : nil
}

private func samplesDirectory(for speakerId: Int64) -> URL {
    url.deletingLastPathComponent()
        .appendingPathComponent("speaker-samples/\(speakerId)", isDirectory: true)
}

private func sampleFileURL(_ sample: SpeakerSample) -> URL {
    samplesDirectory(for: sample.speakerId)
        .appendingPathComponent(sample.id + ".wav")
}
```

Also update `deleteSpeaker` to remove the speaker's sample directory:

```swift
public func deleteSpeaker(id: Int64) throws {
    store.speakers.removeAll { $0.id == id }
    store.centroids.removeAll { $0.speakerId == id }
    store.samples.removeAll { $0.speakerId == id }           // new
    try? FileManager.default.removeItem(at: samplesDirectory(for: id))  // new
    try save()
}
```

And update `merge` to migrate samples from `src` to `dst` before removing `src`:

```swift
// Inside merge(src:into:), before store.speakers.removeAll:
let srcSamples = store.samples.filter { $0.speakerId == src }
for var s in srcSamples {
    let oldURL = sampleFileURL(s)
    s.speakerId = dst
    let newDir = samplesDirectory(for: dst)
    try? FileManager.default.createDirectory(at: newDir, withIntermediateDirectories: true)
    let newURL = newDir.appendingPathComponent(s.id + ".wav")
    try? FileManager.default.moveItem(at: oldURL, to: newURL)
    store.samples.removeAll { $0.id == s.id }
    store.samples.append(s)
}
// Cap combined samples to 5 after migration
let combined = store.samples.filter { $0.speakerId == dst }
    .sorted { $0.durationSeconds > $1.durationSeconds }
if combined.count > 5 {
    let toRemove = combined.dropFirst(5)
    for s in toRemove {
        try? FileManager.default.removeItem(at: sampleFileURL(s))
    }
    let removeIds = Set(toRemove.map(\.id))
    store.samples.removeAll { removeIds.contains($0.id) }
}
store.samples.removeAll { $0.speakerId == src }
```

---

## Pipeline Integration

In `Sources/SpeakerRegistry/PipelineSession.swift`, the audio slice per segment is already extracted for ASR at lines 98–101. Capture the same slice for sample storage immediately after:

```swift
for seg in diarResult.segments {
    guard let speaker = localToRegistry[seg.speakerId] else { continue }
    let start = Double(seg.startTime)
    let end   = Double(seg.endTime)
    let startSample = Int(start * Double(sampleRate))
    let endSample   = min(Int(end * Double(sampleRate)), audio.count)
    let slice = Array(audio[startSample..<endSample])

    // Transcribe (existing)
    let transcript = asr.map { $0.transcribe(audio: slice, sampleRate: sampleRate, language: nil) }

    // Capture sample (new) — registry enforces the quality threshold and cap internally
    if let speakerId = speaker.id {
        try await registry.addSample(
            speakerId: speakerId,
            audio: slice,
            durationSeconds: end - start,
            sampleRate: sampleRate)
    }

    annotated.append(AnnotatedSegment(
        speaker: speaker,
        startTime: start,
        endTime: end,
        transcriptText: transcript))
}
```

---

## New HTTP Endpoints

Add to `Sources/AudioServer/RegistryRoutes.swift` within `addRegistryRoutes`:

```swift
// GET /registry/speakers/:id/samples
// Returns metadata for all stored clips for this speaker.
group.get("speakers/:id/samples") { _, context in
    let id = try requireInt64(context.parameters.get("id"))
    let samples = await registry.samples(forSpeaker: id)
    return jsonResponse([
        "samples": samples.map { s -> [String: Any] in
            [
                "id": s.id,
                "duration_seconds": s.durationSeconds,
                "created_at": ISO8601DateFormatter().string(from: s.createdAt),
            ]
        }
    ])
}

// GET /registry/speakers/:id/samples/:sample_id
// Returns the WAV clip as audio/wav.
group.get("speakers/:id/samples/:sample_id") { _, context in
    let speakerId  = try requireInt64(context.parameters.get("id"))
    let sampleId   = context.parameters.get("sample_id") ?? ""
    guard let fileURL = await registry.sampleFileURL(id: sampleId, speakerId: speakerId) else {
        return errorResponse("Sample not found", status: .notFound)
    }
    let data = try Data(contentsOf: fileURL)
    return Response(
        status: .ok,
        headers: [.contentType: "audio/wav"],
        body: .init(byteBuffer: ByteBuffer(data: data)))
}
```

---

## Full API Surface (additions only)

| Endpoint | Method | Response | Notes |
|---|---|---|---|
| `/registry/speakers/:id/samples` | GET | `{ samples: [{ id, duration_seconds, created_at }] }` | Sorted longest-first |
| `/registry/speakers/:id/samples/:sample_id` | GET | WAV binary (`audio/wav`) | 404 if not found |

Existing endpoints are unchanged.

---

## Behaviour Summary

| Event | Effect on samples |
|---|---|
| `POST /registry/sessions` | Clips ≥ 2 s captured per segment; capped at 5 per speaker (longest kept) |
| `DELETE /registry/speakers/:id` | All clips for that speaker deleted from disk and metadata |
| `POST /registry/speakers/merge` | `src` clips migrated to `dst`, combined set re-capped at 5 |
| Speaker cap exceeded | Shortest clip deleted from disk and metadata |
