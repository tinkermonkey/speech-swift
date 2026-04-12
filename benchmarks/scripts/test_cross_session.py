#!/usr/bin/env python3
"""
Cross-session speaker identity test.

Validates that the same physical speaker is re-identified across separate
POST /registry/sessions calls — the failure mode introduced by commit 1648e9c
(threshold 0.75 > intra-speaker mean 0.726, causing new identity on each chunk).

Method:
  For each VoxConverse recording that has 1 dominant speaker:
    1. Split the WAV at the midpoint into two halves ("session A" and "session B")
    2. Reset the registry
    3. Submit session A → record the speaker_id returned for the dominant speaker
    4. Submit session B (without reset) → check the dominant speaker gets the same id
    5. Record: match (same id) or split (different id)
  Additionally test different-speaker pairs to measure false-merge rate.

Metrics reported:
  - Same-speaker re-ID rate  (target: >90%)
  - False-merge rate          (target: <5%)

Usage:
    python test_cross_session.py
    python test_cross_session.py --threshold 0.65
    python test_cross_session.py --num-pairs 25 --server http://localhost:9090

Requirements:
    pip install soundfile requests rich numpy
    VoxConverse must already be downloaded:
      python benchmarks/scripts/download_voxconverse.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

SERVER_DEFAULT = "http://localhost:8080"
DATA_DIR = Path(__file__).parent.parent / "data" / "voxconverse"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def check_dependencies() -> None:
    missing = []
    for pkg in ("soundfile", "numpy", "requests", "rich"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            f"ERROR: Missing packages: {', '.join(missing)}\n"
            "Run: pip install soundfile numpy requests rich",
            file=sys.stderr,
        )
        sys.exit(1)


def reset_registry(server: str) -> None:
    import requests

    resp = requests.delete(f"{server}/registry/speakers", timeout=30)
    if resp.status_code not in (200, 204):
        print(
            f"WARNING: Registry reset returned HTTP {resp.status_code}",
            file=sys.stderr,
        )


def submit_wav(wav_path: Path, server: str, threshold: float | None) -> dict:
    import requests

    url = f"{server}/registry/sessions"
    if threshold is not None:
        url += f"?threshold={threshold}"

    with open(wav_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (wav_path.name, f, "audio/wav")},
            timeout=300,
        )
    resp.raise_for_status()
    return resp.json()


def dominant_speaker_id(response: dict) -> str | None:
    """Return the speaker_label with the most total duration."""
    durations: dict[str, float] = {}
    for seg in response.get("segments", []):
        label = seg["speaker_label"]
        durations[label] = durations.get(label, 0.0) + seg["duration"]
    if not durations:
        return None
    return max(durations, key=lambda k: durations[k])


def split_wav(wav_path: Path, tmp_dir: str) -> tuple[Path, Path]:
    """Split a WAV at the midpoint, write two halves, return their paths."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    mid = len(audio) // 2
    half_a = Path(tmp_dir) / f"{wav_path.stem}_A.wav"
    half_b = Path(tmp_dir) / f"{wav_path.stem}_B.wav"
    sf.write(half_a, audio[:mid], sr)
    sf.write(half_b, audio[mid:], sr)
    return half_a, half_b


def load_rttm_speakers(rttm_path: Path) -> dict[str, float]:
    """Return {speaker_id: total_duration} from an RTTM file."""
    speakers: dict[str, float] = {}
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            speakers[parts[7]] = speakers.get(parts[7], 0.0) + float(parts[4])
    return speakers


def select_test_recordings(num_pairs: int) -> list[tuple[Path, str]]:
    """
    Pick VoxConverse recordings suitable for same-speaker cross-session tests.

    Criteria:
      - RTTM exists
      - A single speaker dominates ≥60% of the recording (clean dominant speaker)
    Returns list of (wav_path, dominant_rttm_speaker_id).
    """
    audio_dir = DATA_DIR / "audio"
    rttm_dir = DATA_DIR / "rttm"

    if not audio_dir.exists():
        print(
            f"ERROR: VoxConverse audio not found at {audio_dir}\n"
            "Run: python benchmarks/scripts/download_voxconverse.py",
            file=sys.stderr,
        )
        sys.exit(1)

    candidates = []
    for wav in sorted(audio_dir.rglob("*.wav")):
        rttm = rttm_dir / (wav.stem + ".rttm")
        if not rttm.exists():
            continue
        speakers = load_rttm_speakers(rttm)
        if not speakers:
            continue
        total = sum(speakers.values())
        dominant_spk, dominant_dur = max(speakers.items(), key=lambda kv: kv[1])
        if dominant_dur / total >= 0.60:
            candidates.append((wav, dominant_spk))
        if len(candidates) >= num_pairs * 3:  # gather extras for diff-speaker pairs
            break

    return candidates


def run_same_speaker_tests(
    recordings: list[tuple[Path, str]],
    server: str,
    threshold: float | None,
    console,
) -> list[dict]:
    """
    For each recording: split → submit A → submit B (no reset between) → compare ids.
    Registry is reset before each pair.
    """
    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for wav, rttm_dominant in recordings:
            reset_registry(server)
            try:
                half_a, half_b = split_wav(wav, tmp_dir)

                resp_a = submit_wav(half_a, server, threshold)
                spk_a = dominant_speaker_id(resp_a)

                resp_b = submit_wav(half_b, server, threshold)
                spk_b = dominant_speaker_id(resp_b)

                matched = spk_a is not None and spk_b is not None and spk_a == spk_b
                results.append(
                    {
                        "file": wav.stem,
                        "type": "same_speaker",
                        "rttm_dominant": rttm_dominant,
                        "session_a_label": spk_a,
                        "session_b_label": spk_b,
                        "matched": matched,
                    }
                )
                status = "[green]✓ match[/green]" if matched else "[red]✗ split[/red]"
                console.print(f"  {wav.stem:40s}  {status}  (A={spk_a}, B={spk_b})")
            except Exception as exc:
                console.print(f"  [red]ERROR[/red] {wav.stem}: {exc}")
                results.append({"file": wav.stem, "type": "same_speaker", "error": str(exc)})

    return results


def run_different_speaker_tests(
    recordings: list[tuple[Path, str]],
    server: str,
    threshold: float | None,
    console,
) -> list[dict]:
    """
    For pairs of different speakers: submit each whole recording → check ids differ.
    Registry is reset before each pair.
    """
    results = []
    pairs = [(recordings[i], recordings[i + 1]) for i in range(0, len(recordings) - 1, 2)]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for (wav_a, _), (wav_b, _) in pairs:
            reset_registry(server)
            try:
                resp_a = submit_wav(wav_a, server, threshold)
                spk_a = dominant_speaker_id(resp_a)

                resp_b = submit_wav(wav_b, server, threshold)
                spk_b = dominant_speaker_id(resp_b)

                false_merge = spk_a is not None and spk_b is not None and spk_a == spk_b
                results.append(
                    {
                        "file_a": wav_a.stem,
                        "file_b": wav_b.stem,
                        "type": "different_speaker",
                        "session_a_label": spk_a,
                        "session_b_label": spk_b,
                        "false_merge": false_merge,
                    }
                )
                status = "[red]✗ false merge[/red]" if false_merge else "[green]✓ separated[/green]"
                console.print(f"  {wav_a.stem} vs {wav_b.stem}  {status}")
            except Exception as exc:
                console.print(f"  [red]ERROR[/red] {wav_a.stem} vs {wav_b.stem}: {exc}")
                results.append(
                    {
                        "file_a": wav_a.stem,
                        "file_b": wav_b.stem,
                        "type": "different_speaker",
                        "error": str(exc),
                    }
                )

    return results


def main() -> None:
    check_dependencies()

    from datetime import datetime
    import json
    from rich.console import Console
    from rich.table import Table

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Similarity threshold override (default: server default)",
    )
    ap.add_argument(
        "--num-pairs",
        type=int,
        default=25,
        metavar="N",
        help="Number of same-speaker pairs to test (default: 25)",
    )
    ap.add_argument(
        "--server",
        default=SERVER_DEFAULT,
        help=f"audio-server base URL (default: {SERVER_DEFAULT})",
    )
    args = ap.parse_args()

    console = Console()

    console.print(
        f"\n[bold]Cross-session speaker identity test[/bold]\n"
        f"threshold={args.threshold or 'server default'}  "
        f"num_pairs={args.num_pairs}  server={args.server}\n"
    )

    # Select recordings
    console.print("Selecting test recordings from VoxConverse …")
    all_candidates = select_test_recordings(args.num_pairs)
    if len(all_candidates) < args.num_pairs:
        console.print(
            f"[yellow]WARNING:[/yellow] Only {len(all_candidates)} suitable recordings found "
            f"(wanted {args.num_pairs}). Proceeding with what's available."
        )

    same_speaker_recs = all_candidates[: args.num_pairs]
    diff_speaker_recs = all_candidates[args.num_pairs : args.num_pairs * 2]

    # Same-speaker re-ID test
    console.rule("Same-speaker re-identification (split halves)")
    same_results = run_same_speaker_tests(same_speaker_recs, args.server, args.threshold, console)

    # Different-speaker separation test
    console.rule("Different-speaker separation (false-merge rate)")
    diff_results = run_different_speaker_tests(diff_speaker_recs, args.server, args.threshold, console)

    # Compute metrics
    same_valid = [r for r in same_results if "error" not in r]
    reid_rate = (
        sum(1 for r in same_valid if r["matched"]) / len(same_valid) * 100
        if same_valid else 0
    )

    diff_valid = [r for r in diff_results if "error" not in r]
    false_merge_rate = (
        sum(1 for r in diff_valid if r["false_merge"]) / len(diff_valid) * 100
        if diff_valid else 0
    )

    # Results table
    table = Table(title="Cross-Session Identity Results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Pass?", justify="center")

    reid_pass = reid_rate >= 90
    fm_pass = false_merge_rate <= 5

    table.add_row(
        "Same-speaker re-ID rate",
        f"{reid_rate:.1f}%",
        "≥ 90%",
        "[green]PASS[/green]" if reid_pass else "[red]FAIL[/red]",
    )
    table.add_row(
        "False-merge rate",
        f"{false_merge_rate:.1f}%",
        "≤ 5%",
        "[green]PASS[/green]" if fm_pass else "[red]FAIL[/red]",
    )
    console.print(table)

    # Persist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    thr_tag = f"_t{args.threshold}" if args.threshold is not None else ""
    out = RESULTS_DIR / f"cross_session{thr_tag}_{ts}.json"
    out.write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "server": args.server,
                "timestamp": datetime.now().isoformat(),
                "reid_rate_pct": round(reid_rate, 2),
                "false_merge_rate_pct": round(false_merge_rate, 2),
                "same_speaker_results": same_results,
                "diff_speaker_results": diff_results,
            },
            indent=2,
        )
    )
    console.print(f"\nResults written to [cyan]{out}[/cyan]")

    # Exit non-zero if either target is missed
    if not (reid_pass and fm_pass):
        sys.exit(1)


if __name__ == "__main__":
    main()
