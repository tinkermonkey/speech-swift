# Speaker Recognition Tuning Plan

## Context

The speech-swift speaker registry uses WeSpeaker ResNet34-LM embeddings and cosine-similarity
matching to maintain speaker identity across audio chunks. Commit `1648e9c` (Apr 5) simplified
the registry from SQLite to a JSON-backed centroid store and inadvertently broke cross-chunk
speaker continuity.

**Root cause (confirmed by existing benchmark data):**
WeSpeaker's mean intra-speaker cosine similarity on VoxConverse is **0.726**, but the registry
threshold was **0.75** — meaning the same speaker's embedding from a different chunk was routinely
*below* the match threshold and assigned a new identity. The threshold has been corrected to
**0.65** as an immediate fix (see `RegistryRoutes.swift`).

**Intra vs inter-speaker similarity (WeSpeaker MLX, VoxConverse):**

| | Mean |
|---|---|
| Intra-speaker (same person, different chunks) | 0.726 |
| Inter-speaker (different people) | 0.142 |
| Separation | 0.584 |

The 0.65 threshold sits well above inter-speaker (0.142) and below the intra-speaker mean
(0.726), giving ≈ 3× margin on the inter side and reasonable tolerance for natural embedding
variance. The goal of this tuning plan is to determine the *optimal* threshold empirically and
to validate end-to-end diarization quality.

---

## Phase 1 — Environment Setup

### Requirements

```bash
pip install pyannote.metrics pyannote.audio huggingface_hub \
            soundfile librosa tqdm rich tabulate requests
```

`pyannote.audio` is used for its RTTM I/O utilities and DER computation only — the speech-swift
model stack is **not** replaced.

### Dataset Storage

All datasets land under `~/workspace/speech-swift/benchmarks/data/` (git-ignored).

```
benchmarks/
  data/
    ami/          # AMI Meeting Corpus
    voxconverse/  # VoxConverse v0.3
  results/        # JSON + RTTM outputs from benchmark runs
  scripts/        # Python scripts described below
```

---

## Phase 2 — Dataset Download Scripts

### `benchmarks/scripts/download_ami.py`

Downloads the **AMI Meeting Corpus** headset-mix audio and RTTM ground-truth annotations
from the official Hugging Face mirror (argmaxinc collection).

**What it fetches:**
- ~18.7 h of 4-speaker meeting audio (headset mix, 16 kHz mono WAV)
- Forced-alignment RTTM files with speaker turn boundaries
- Meeting IDs: EN2001–EN2010, ES2002–ES2015, IS1000–IS1008, TS3003–TS3012

**Implementation outline:**

```python
#!/usr/bin/env python3
"""Download AMI Meeting Corpus via Hugging Face Datasets."""

from pathlib import Path
from huggingface_hub import snapshot_download
import soundfile as sf, shutil, sys

DATA_DIR = Path(__file__).parent.parent / "data" / "ami"
HF_REPO  = "argmaxinc/ami-diarization"   # public mirror

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading AMI corpus to {DATA_DIR} …")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        ignore_patterns=["*.zip"],    # prefer pre-extracted WAV
    )
    wavs = list(DATA_DIR.rglob("*.wav"))
    print(f"Done — {len(wavs)} WAV files, "
          f"{sum(sf.info(w).duration for w in wavs)/3600:.1f} h total")

if __name__ == "__main__":
    main()
```

**Fallback (direct HTTP):** If the HF mirror is unavailable, fall back to the official RTTM
annotations from `https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/` and audio from the
AMI IHM (Individual Headset Mix) links documented at `https://groups.inf.ed.ac.uk/ami/corpus/`.

---

### `benchmarks/scripts/download_voxconverse.py`

Downloads **VoxConverse v0.3** (~43.6 h, 1–21 speakers per recording).

**What it fetches:**
- Audio: YouTube-derived MP4/WAV files via the official GitHub release
- Annotations: RTTM files from `https://github.com/joonson/voxconverse`

**Implementation outline:**

```python
#!/usr/bin/env python3
"""Download VoxConverse v0.3 audio and RTTM annotations."""

import subprocess, zipfile, urllib.request
from pathlib import Path

DATA_DIR     = Path(__file__).parent.parent / "data" / "voxconverse"
RTTM_URL     = "https://github.com/joonson/voxconverse/archive/refs/heads/master.zip"
AUDIO_GDRIVE = "https://huggingface.co/datasets/argmaxinc/voxconverse/resolve/main"

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Annotations
    rttm_zip = DATA_DIR / "annotations.zip"
    print("Downloading RTTM annotations …")
    urllib.request.urlretrieve(RTTM_URL, rttm_zip)
    with zipfile.ZipFile(rttm_zip) as z:
        z.extractall(DATA_DIR / "annotations_raw")
    # Flatten: copy *.rttm into data/voxconverse/rttm/
    rttm_dir = DATA_DIR / "rttm"
    rttm_dir.mkdir(exist_ok=True)
    for rttm in (DATA_DIR / "annotations_raw").rglob("*.rttm"):
        shutil.copy(rttm, rttm_dir / rttm.name)

    # 2. Audio (HF mirror — dev split ~5 h, good for initial benchmarking)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="argmaxinc/voxconverse",
        repo_type="dataset",
        local_dir=str(DATA_DIR / "audio"),
        ignore_patterns=["test/*"],   # dev split only for speed
    )
    print("Done.")

if __name__ == "__main__":
    main()
```

---

## Phase 3 — Diarization Benchmark Script

### `benchmarks/scripts/run_diarization_benchmark.py`

Submits each benchmark recording to the live speech-swift audio-server at
`http://localhost:8080` and computes DER / JER against ground-truth RTTM files
using `pyannote.metrics`.

**What it does per recording:**
1. POST the WAV to `/registry/sessions`
2. Convert the JSON response (`start`/`end` in seconds, `speaker_label`) → RTTM
3. Load ground-truth RTTM
4. Compute DER (with 0.25 s collar) and JER using `pyannote.metrics`
5. Accumulate per-file and aggregate results

**Key outputs:**
- Per-file DER breakdown (missed speech / false alarm / speaker confusion)
- Aggregate DER and JER across the full dataset
- Confusion matrix: which speaker pairs get merged or split most often
- JSON report written to `benchmarks/results/<dataset>_<timestamp>.json`

**Implementation outline:**

```python
#!/usr/bin/env python3
"""
Benchmark speech-swift diarization against AMI or VoxConverse.

Usage:
    python run_diarization_benchmark.py --dataset ami [--threshold 0.65] [--limit 10]
    python run_diarization_benchmark.py --dataset voxconverse [--threshold 0.65]
"""

import argparse, json, requests, tempfile
from pathlib import Path
from datetime import datetime
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from rich.console import Console
from rich.table import Table
import soundfile as sf

SERVER   = "http://localhost:8080"
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS  = Path(__file__).parent.parent / "results"
console  = Console()

def wav_to_annotation(response_json: dict) -> Annotation:
    """Convert speech-swift JSON response to pyannote Annotation."""
    ann = Annotation()
    for seg in response_json["segments"]:
        ann[Segment(seg["start"], seg["end"])] = seg["speaker_label"]
    return ann

def rttm_to_annotation(rttm_path: Path) -> Annotation:
    """Parse an RTTM file into a pyannote Annotation."""
    ann = Annotation()
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] != "SPEAKER":
                continue
            start    = float(parts[3])
            duration = float(parts[4])
            speaker  = parts[7]
            ann[Segment(start, start + duration)] = speaker
    return ann

def submit_chunk(wav_path: Path) -> dict:
    with open(wav_path, "rb") as f:
        r = requests.post(
            f"{SERVER}/registry/sessions",
            files={"file": (wav_path.name, f, "audio/wav")},
            timeout=300,
        )
    r.raise_for_status()
    return r.json()

def benchmark(dataset: str, limit: int | None):
    audio_dir = DATA_DIR / dataset / "audio"
    rttm_dir  = DATA_DIR / dataset / "rttm"

    wav_files = sorted(audio_dir.rglob("*.wav"))
    if limit:
        wav_files = wav_files[:limit]

    der_metric = DiarizationErrorRate(collar=0.25)
    jer_metric = JaccardErrorRate(collar=0.25)

    rows = []
    for wav in wav_files:
        rttm = rttm_dir / (wav.stem + ".rttm")
        if not rttm.exists():
            console.print(f"[yellow]skip {wav.name} — no RTTM[/yellow]")
            continue

        try:
            result    = submit_chunk(wav)
            hypothesis = wav_to_annotation(result)
            reference  = rttm_to_annotation(rttm)
            dur        = sf.info(wav).duration

            der = abs(der_metric(reference, hypothesis)) * 100
            jer = abs(jer_metric(reference, hypothesis)) * 100
            rows.append({"file": wav.stem, "duration_s": dur, "DER": der, "JER": jer})
            console.print(f"[green]✓[/green] {wav.stem:30s}  DER={der:5.1f}%  JER={jer:5.1f}%")
        except Exception as e:
            console.print(f"[red]✗[/red] {wav.name}: {e}")

    # Summary table
    table = Table(title=f"{dataset.upper()} — Aggregate Results")
    table.add_column("Metric"); table.add_column("Value", justify="right")
    agg_der = sum(r["DER"] for r in rows) / len(rows) if rows else 0
    agg_jer = sum(r["JER"] for r in rows) / len(rows) if rows else 0
    table.add_row("Files evaluated", str(len(rows)))
    table.add_row("Mean DER",  f"{agg_der:.1f}%")
    table.add_row("Mean JER",  f"{agg_jer:.1f}%")
    console.print(table)

    # Persist results
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS / f"{dataset}_{ts}.json"
    out.write_text(json.dumps({"dataset": dataset, "rows": rows,
                               "agg_DER": agg_der, "agg_JER": agg_jer}, indent=2))
    console.print(f"\nResults written to {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ami", "voxconverse"], required=True)
    ap.add_argument("--limit",   type=int, default=None,
                    help="Max recordings to evaluate (omit for full dataset)")
    args = ap.parse_args()
    benchmark(args.dataset, args.limit)
```

---

## Phase 4 — Threshold Sweep Script

### `benchmarks/scripts/sweep_threshold.py`

Iterates over candidate thresholds and reports DER/JER for each, using a small
held-out subset (10 AMI meetings) to find the optimal value without overfitting
to the full corpus.

```python
#!/usr/bin/env python3
"""
Sweep the similarity threshold by temporarily patching RegistryRoutes.swift,
rebuilding, restarting the server, and running the benchmark subset.

Thresholds tested: 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
"""

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# For each threshold:
#   1. sed-replace threshold in RegistryRoutes.swift
#   2. make build (or swift build)
#   3. restart audio-server
#   4. DELETE /registry/speakers (reset registry between runs)
#   5. run run_diarization_benchmark.py --dataset ami --limit 10
#   6. collect DER/JER
#   7. restore previous threshold

# Results written to benchmarks/results/threshold_sweep_<timestamp>.json
# and printed as a ranked table.
```

The sweep script will also test **per-chunk registry reset** vs **persistent registry** to
measure how much identity continuity across chunks contributes to overall DER.

---

## Phase 5 — Cross-Session Identity Test

### `benchmarks/scripts/test_cross_session.py`

Validates that the same physical speaker is re-identified across separate POST
`/registry/sessions` calls — the specific failure mode that motivated this plan.

**Method:**
1. Slice VoxCeleb speaker clips into pairs: two 15 s clips of the same speaker from different
   source videos, treated as separate "sessions."
2. Submit clip A → record `speaker_id` returned.
3. Submit clip B → assert same `speaker_id`.
4. Repeat for 50 speaker pairs (25 same-speaker pairs, 25 different-speaker pairs).
5. Compute: same-speaker re-identification rate, false-merge rate.

This is the direct regression test for the `1648e9c` regression.

---

## Tuning Milestones

| Milestone | Target | Measurement |
|---|---|---|
| Cross-chunk identity (baseline restored) | >90% same-speaker re-ID rate | `test_cross_session.py` |
| AMI DER (headset mix) | <20% DER | `run_diarization_benchmark.py --dataset ami` |
| VoxConverse DER | <25% DER | `run_diarization_benchmark.py --dataset voxconverse` |
| Optimal threshold confirmed | Best DER on 10-meeting sweep | `sweep_threshold.py` |

---

## Current Status

| Item | Status |
|---|---|
| Threshold confirmed 0.75 (sweep 2026-04-12) | ✅ Done — 0.65 interim reverted |
| Download scripts | ✅ Done (`benchmarks/scripts/`) |
| Diarization benchmark script | ✅ Done |
| Threshold sweep | ✅ Done — 0.75 optimal on VoxConverse dev |
| Cross-session identity test | ✅ Done — 86.7% re-ID, 0.0% false-merge @ 0.75 |
| VoxConverse baseline DER measured | ✅ 6.0% weighted DER @ 0.75 (20 files, 1.63h) |
| AMI baseline DER measured | ⬜ Pending |

---

## Why 0.65 Is the Right Interim Threshold

The existing `docs/benchmarks/speaker-embeddings.md` data shows:

```
WeSpeaker intra-speaker mean similarity:  0.726
WeSpeaker inter-speaker mean similarity:  0.142
Optimal decision boundary (midpoint):     0.434
```

The theoretical midpoint between inter and intra means is 0.434. In practice, a higher threshold
is desirable to avoid false merges, but 0.75 is **above the intra-speaker mean**, meaning typical
same-speaker pairs fail to match. 0.65 is:
- 0.076 below the intra-speaker mean (≈11% margin)
- 0.508 above the inter-speaker mean (≈3.6× margin)

The sweep (Phase 4) will find whether 0.65–0.70 is empirically optimal on the benchmark corpora.
