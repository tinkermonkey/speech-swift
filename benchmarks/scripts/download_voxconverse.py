#!/usr/bin/env python3
"""
Download VoxConverse v0.3 audio and RTTM ground-truth annotations.

Audio (~43.6 h, 1-21 speakers per recording) is fetched from the Hugging Face
mirror.  RTTM annotations come directly from the official GitHub release.

Usage:
    python download_voxconverse.py [--split dev|test|all] [--limit N]

Outputs:
    benchmarks/data/voxconverse/audio/   WAV files (16 kHz mono)
    benchmarks/data/voxconverse/rttm/    RTTM annotation files

The download requires ~25 GB for the full corpus; use --split dev (5 h) or
--limit N for a smaller subset when iterating quickly.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "voxconverse"

RTTM_ZIP_URL = (
    "https://github.com/joonson/voxconverse/archive/refs/heads/master.zip"
)

HF_REPO = "joonson/voxconverse"      # canonical HF mirror
HF_REPO_FALLBACK = "diarizers-community/voxconverse"


def download_rttm_annotations(data_dir: Path) -> None:
    rttm_dir = data_dir / "rttm"
    if rttm_dir.exists() and any(rttm_dir.glob("*.rttm")):
        print(f"RTTM annotations already present in {rttm_dir}, skipping.")
        return

    rttm_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = data_dir / "_annotations.zip"
    tmp_extract = data_dir / "_annotations_raw"

    print("Downloading RTTM annotations from GitHub …")
    try:
        urllib.request.urlretrieve(RTTM_ZIP_URL, tmp_zip)
    except Exception as exc:
        print(f"ERROR: Could not download RTTM annotations: {exc}", file=sys.stderr)
        print(
            "Manual fallback: clone https://github.com/joonson/voxconverse and copy "
            "the dev/ and test/ *.rttm files into benchmarks/data/voxconverse/rttm/",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Extracting …")
    with zipfile.ZipFile(tmp_zip) as z:
        z.extractall(tmp_extract)

    # Flatten: copy every .rttm from the extracted tree into rttm/
    copied = 0
    for rttm_file in tmp_extract.rglob("*.rttm"):
        shutil.copy(rttm_file, rttm_dir / rttm_file.name)
        copied += 1

    # Clean up scratch files
    tmp_zip.unlink(missing_ok=True)
    shutil.rmtree(tmp_extract, ignore_errors=True)
    print(f"Copied {copied} RTTM files to {rttm_dir}")


def _extract_parquet_wavs(parquet_files: list, audio_dir: Path, rttm_dir: Path) -> None:
    """Extract WAV bytes and generate RTTM files from HF parquet audio dataset."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("WARNING: pyarrow not installed — cannot extract WAVs from parquet. Run: pip install pyarrow", file=sys.stderr)
        return

    rttm_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    for pf_path in sorted(parquet_files):
        table = pq.read_table(pf_path)
        rows = table.to_pydict()
        for i in range(len(rows["audio"])):
            audio = rows["audio"][i]
            filename = Path(audio["path"]).name
            wav_out = audio_dir / filename
            if not wav_out.exists():
                wav_out.write_bytes(audio["bytes"])

            # Generate RTTM from embedded annotations if not already present
            stem = wav_out.stem
            rttm_out = rttm_dir / f"{stem}.rttm"
            if not rttm_out.exists() and "timestamps_start" in rows:
                with open(rttm_out, "w") as f:
                    for start, end, spk in zip(
                        rows["timestamps_start"][i],
                        rows["timestamps_end"][i],
                        rows["speakers"][i],
                    ):
                        dur = end - start
                        f.write(f"SPEAKER {stem} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n")
            extracted += 1

    print(f"Extracted {extracted} WAV files from {len(parquet_files)} parquet shards")


def download_audio(data_dir: Path, split: str, limit: int | None) -> None:
    audio_dir = data_dir / "audio"
    rttm_dir = data_dir / "rttm"
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed.\n"
            "Run: pip install huggingface_hub soundfile",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build ignore patterns to restrict to the requested split(s)
    ignore_patterns: list[str] = []
    if split == "dev":
        ignore_patterns.append("test/*")
    elif split == "test":
        ignore_patterns.append("dev/*")

    repos_to_try = [HF_REPO, HF_REPO_FALLBACK]
    for repo in repos_to_try:
        print(f"Downloading VoxConverse audio ({split} split) from {repo} …")
        try:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=str(audio_dir),
                ignore_patterns=ignore_patterns or None,
            )
            break
        except Exception as exc:
            print(f"  WARNING: {repo} failed: {exc}", file=sys.stderr)
    else:
        print(
            "ERROR: All HF mirrors failed. Manual fallback:\n"
            "  - Download audio from the VoxConverse project page:\n"
            "    https://mm.kaist.ac.kr/datasets/voxconverse/\n"
            "  - Convert to 16 kHz mono WAV and place in:\n"
            f"    {audio_dir}/",
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract WAVs from parquet files if present (HF datasets store audio as parquet)
    parquet_files = list((audio_dir / "data").glob(f"{split if split != 'all' else ''}*.parquet"))
    if not parquet_files and split == "all":
        parquet_files = list((audio_dir / "data").glob("*.parquet"))
    if parquet_files:
        _extract_parquet_wavs(parquet_files, audio_dir, rttm_dir)

    # Optionally cap to N files for quick iteration
    if limit is not None:
        wavs = sorted(audio_dir.glob("*.wav"))
        to_remove = wavs[limit:]
        if to_remove:
            print(f"Limiting to {limit} files; removing {len(to_remove)} extras …")
            for f in to_remove:
                f.unlink()

    wavs = sorted(audio_dir.glob("*.wav"))
    try:
        import soundfile as sf
        total_h = sum(sf.info(w).duration for w in wavs) / 3600
        print(f"Done — {len(wavs)} WAV files, {total_h:.1f} h total")
    except ImportError:
        print(f"Done — {len(wavs)} WAV files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="dev",
        help="Which split to download (default: dev ~5 h)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Keep only the first N WAV files after download",
    )
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_rttm_annotations(DATA_DIR)
    download_audio(DATA_DIR, args.split, args.limit)

    rttm_count = len(list((DATA_DIR / "rttm").glob("*.rttm")))
    wav_count = len(list((DATA_DIR / "audio").rglob("*.wav")))
    print(f"\nVoxConverse ready: {wav_count} WAV files, {rttm_count} RTTM files")
    print(f"Location: {DATA_DIR}")


if __name__ == "__main__":
    main()
