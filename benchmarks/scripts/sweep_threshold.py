#!/usr/bin/env python3
"""
Sweep cosine similarity thresholds and find the one that minimises DER.

Uses the ?threshold query param added to POST /registry/sessions — no rebuild
needed between runs.  For each threshold the registry is reset and the
benchmark subset is evaluated from scratch.

Usage:
    python sweep_threshold.py --dataset ami --limit 10
    python sweep_threshold.py --dataset voxconverse --limit 20
    python sweep_threshold.py --dataset ami --limit 10 --thresholds 0.60 0.65 0.70

Thresholds tested (default): 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80

Results are written to benchmarks/results/threshold_sweep_<timestamp>.json
and printed as a ranked table.

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

# Import the benchmark runner from the same package
sys.path.insert(0, str(Path(__file__).parent))
from run_diarization_benchmark import benchmark, check_dependencies, reset_registry

SERVER_DEFAULT = "http://localhost:8080"
RESULTS_DIR = Path(__file__).parent.parent / "results"

DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def run_sweep(
    dataset: str,
    server: str,
    thresholds: list[float],
    limit: int | None,
) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print(
        f"\n[bold]Threshold sweep[/bold]  dataset={dataset}  "
        f"limit={limit or 'all'}  thresholds={thresholds}\n"
    )

    sweep_results = []
    for thr in thresholds:
        console.rule(f"threshold = {thr}")
        reset_registry(server)

        t0 = time.time()
        try:
            report = benchmark(
                dataset=dataset,
                server=server,
                threshold=thr,
                limit=limit,
                no_reset=True,  # already reset above
            )
        except SystemExit:
            console.print(f"[red]threshold {thr} failed — skipping[/red]")
            continue

        elapsed = time.time() - t0
        sweep_results.append(
            {
                "threshold": thr,
                "DER_macro": report["agg_DER_macro"],
                "DER_weighted": report["agg_DER_weighted"],
                "JER_macro": report["agg_JER_macro"],
                "num_files": report["num_files"],
                "elapsed_s": round(elapsed, 0),
            }
        )

    if not sweep_results:
        console.print("[red]No results — check server is running.[/red]")
        sys.exit(1)

    # Sort by weighted DER (most meaningful metric)
    sweep_results.sort(key=lambda r: r["DER_weighted"])
    best = sweep_results[0]

    # Summary table
    table = Table(title="Threshold Sweep — Ranked by Duration-Weighted DER")
    table.add_column("Rank", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("DER (weighted)", justify="right")
    table.add_column("DER (macro)", justify="right")
    table.add_column("JER (macro)", justify="right")
    table.add_column("Files", justify="right")

    for rank, row in enumerate(sweep_results, 1):
        style = "bold green" if rank == 1 else ""
        table.add_row(
            str(rank),
            str(row["threshold"]),
            f"{row['DER_weighted']:.1f}%",
            f"{row['DER_macro']:.1f}%",
            f"{row['JER_macro']:.1f}%",
            str(row["num_files"]),
            style=style,
        )

    console.print(table)
    console.print(
        f"\n[bold green]Best threshold: {best['threshold']}[/bold green]  "
        f"DER (weighted) = {best['DER_weighted']:.1f}%"
    )

    # Write results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"threshold_sweep_{dataset}_{ts}.json"
    out.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "server": server,
                "limit": limit,
                "thresholds_tested": thresholds,
                "best_threshold": best["threshold"],
                "best_DER_weighted": best["DER_weighted"],
                "timestamp": datetime.now().isoformat(),
                "results": sweep_results,
            },
            indent=2,
        )
    )
    console.print(f"Sweep results written to [cyan]{out}[/cyan]")


def main() -> None:
    check_dependencies()

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dataset",
        choices=["ami", "voxconverse"],
        required=True,
        help="Dataset to use for the sweep",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="Number of recordings per threshold (default: 10)",
    )
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        metavar="FLOAT",
        help=f"Thresholds to test (default: {DEFAULT_THRESHOLDS})",
    )
    ap.add_argument(
        "--server",
        default=SERVER_DEFAULT,
        help=f"audio-server base URL (default: {SERVER_DEFAULT})",
    )
    args = ap.parse_args()

    run_sweep(
        dataset=args.dataset,
        server=args.server,
        thresholds=sorted(args.thresholds),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
