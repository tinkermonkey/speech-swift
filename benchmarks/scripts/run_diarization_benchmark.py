#!/usr/bin/env python3
"""
Benchmark speech-swift diarization against AMI or VoxConverse.

For each recording the script:
  1. POSTs the WAV to POST /registry/sessions (optionally with ?threshold=N)
  2. Converts the JSON response to a pyannote Annotation
  3. Loads the ground-truth RTTM
  4. Computes DER and JER (0.25 s collar) via pyannote.metrics
  5. Accumulates per-file and aggregate results
  6. Writes a JSON report to benchmarks/results/

Usage:
    python run_diarization_benchmark.py --dataset voxconverse
    python run_diarization_benchmark.py --dataset ami --limit 10
    python run_diarization_benchmark.py --dataset voxconverse --threshold 0.70
    python run_diarization_benchmark.py --dataset ami --server http://localhost:9090

Requirements:
    pip install pyannote.metrics pyannote.audio soundfile requests rich
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

SERVER_DEFAULT = "http://localhost:8080"
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def check_dependencies() -> None:
    missing = []
    for pkg in ("pyannote.core", "pyannote.metrics", "requests", "soundfile", "rich"):
        try:
            __import__(pkg.replace(".", "_") if "." in pkg else pkg)
        except ImportError:
            # Try the actual import path
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
    if missing:
        print(
            f"ERROR: Missing packages: {', '.join(missing)}\n"
            "Run: pip install pyannote.metrics pyannote.audio soundfile requests rich",
            file=sys.stderr,
        )
        sys.exit(1)


def wav_to_annotation(response_json: dict):
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for seg in response_json.get("segments", []):
        ann[Segment(seg["start"], seg["end"])] = seg["speaker_label"]
    return ann


def rttm_to_annotation(rttm_path: Path):
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            ann[Segment(start, start + duration)] = speaker
    return ann


def submit_wav(wav_path: Path, server: str, threshold: float | None) -> dict:
    import requests

    url = f"{server}/registry/sessions"
    if threshold is not None:
        url += f"?threshold={threshold}"

    with open(wav_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (wav_path.name, f, "audio/wav")},
            timeout=600,
        )
    resp.raise_for_status()
    return resp.json()


def reset_registry(server: str) -> None:
    import requests

    resp = requests.delete(f"{server}/registry/speakers", timeout=30)
    if resp.status_code not in (200, 204):
        print(
            f"WARNING: Registry reset returned HTTP {resp.status_code}",
            file=sys.stderr,
        )


def benchmark(
    dataset: str,
    server: str,
    threshold: float | None,
    limit: int | None,
    no_reset: bool,
) -> dict:
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    from rich.console import Console
    from rich.table import Table
    import soundfile as sf

    console = Console()

    audio_dir = DATA_DIR / dataset / "audio"
    rttm_dir = DATA_DIR / dataset / "rttm"

    if not audio_dir.exists():
        console.print(
            f"[red]ERROR:[/red] Audio directory not found: {audio_dir}\n"
            f"Run: python benchmarks/scripts/download_{dataset}.py"
        )
        sys.exit(1)

    wav_files = sorted(audio_dir.rglob("*.wav"))
    if not wav_files:
        console.print(f"[red]ERROR:[/red] No WAV files found in {audio_dir}")
        sys.exit(1)

    if limit is not None:
        wav_files = wav_files[:limit]

    console.print(
        f"[bold]Dataset:[/bold] {dataset.upper()}  "
        f"[bold]Files:[/bold] {len(wav_files)}  "
        f"[bold]Threshold:[/bold] {threshold if threshold is not None else 'server default'}  "
        f"[bold]Server:[/bold] {server}"
    )

    if not no_reset:
        console.print("Resetting registry …")
        reset_registry(server)

    der_metric = DiarizationErrorRate(collar=0.25)
    jer_metric = JaccardErrorRate(collar=0.25)

    rows = []
    skipped = 0
    t_start = time.time()

    for wav in wav_files:
        rttm = rttm_dir / (wav.stem + ".rttm")
        if not rttm.exists():
            console.print(f"[yellow]skip[/yellow] {wav.name} — no RTTM")
            skipped += 1
            continue

        try:
            t0 = time.time()
            result = submit_wav(wav, server, threshold)
            elapsed = time.time() - t0

            hypothesis = wav_to_annotation(result)
            reference = rttm_to_annotation(rttm)
            duration = sf.info(wav).duration

            der = abs(der_metric(reference, hypothesis)) * 100
            jer = abs(jer_metric(reference, hypothesis)) * 100
            n_spk = result.get("num_speakers", "?")

            rows.append(
                {
                    "file": wav.stem,
                    "duration_s": round(duration, 1),
                    "num_speakers": n_spk,
                    "DER": round(der, 2),
                    "JER": round(jer, 2),
                    "server_elapsed_s": round(elapsed, 1),
                }
            )
            console.print(
                f"[green]✓[/green] {wav.stem:35s}  "
                f"DER={der:5.1f}%  JER={jer:5.1f}%  "
                f"spk={n_spk}  t={elapsed:.0f}s"
            )
        except Exception as exc:
            console.print(f"[red]✗[/red] {wav.name}: {exc}")
            skipped += 1

    total_elapsed = time.time() - t_start

    if not rows:
        console.print("[red]No files evaluated — check server is running and dataset is downloaded.[/red]")
        sys.exit(1)

    agg_der = sum(r["DER"] for r in rows) / len(rows)
    agg_jer = sum(r["JER"] for r in rows) / len(rows)

    # Weighted by duration
    total_dur = sum(r["duration_s"] for r in rows)
    weighted_der = (
        sum(r["DER"] * r["duration_s"] for r in rows) / total_dur if total_dur else 0
    )

    table = Table(title=f"{dataset.upper()} — Aggregate Results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Files evaluated", str(len(rows)))
    table.add_row("Files skipped", str(skipped))
    table.add_row("Total audio", f"{total_dur / 3600:.2f} h")
    table.add_row("Mean DER (macro)", f"{agg_der:.1f}%")
    table.add_row("Mean DER (duration-weighted)", f"{weighted_der:.1f}%")
    table.add_row("Mean JER (macro)", f"{agg_jer:.1f}%")
    table.add_row("Total wall time", f"{total_elapsed:.0f}s")
    console.print(table)

    report = {
        "dataset": dataset,
        "server": server,
        "threshold": threshold,
        "timestamp": datetime.now().isoformat(),
        "num_files": len(rows),
        "num_skipped": skipped,
        "agg_DER_macro": round(agg_der, 3),
        "agg_DER_weighted": round(weighted_der, 3),
        "agg_JER_macro": round(agg_jer, 3),
        "total_duration_h": round(total_dur / 3600, 3),
        "rows": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    thr_tag = f"_t{threshold}" if threshold is not None else ""
    out = RESULTS_DIR / f"{dataset}{thr_tag}_{ts}.json"
    out.write_text(json.dumps(report, indent=2))
    console.print(f"\nResults written to [cyan]{out}[/cyan]")

    return report


def main() -> None:
    check_dependencies()

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dataset",
        choices=["ami", "voxconverse"],
        required=True,
        help="Dataset to benchmark against",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Similarity threshold override (passed as ?threshold= to the server)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N recordings",
    )
    ap.add_argument(
        "--server",
        default=SERVER_DEFAULT,
        help=f"audio-server base URL (default: {SERVER_DEFAULT})",
    )
    ap.add_argument(
        "--no-reset",
        action="store_true",
        help="Skip registry reset before the run (use to accumulate across calls)",
    )
    args = ap.parse_args()

    benchmark(
        dataset=args.dataset,
        server=args.server,
        threshold=args.threshold,
        limit=args.limit,
        no_reset=args.no_reset,
    )


if __name__ == "__main__":
    main()
