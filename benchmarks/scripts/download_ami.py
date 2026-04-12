#!/usr/bin/env python3
"""
Download the AMI Meeting Corpus (headset-mix audio + RTTM annotations).

The AMI corpus contains ~18.7 h of 4-speaker meeting audio at 16 kHz.
Audio and annotations are fetched from Hugging Face mirrors.

Usage:
    python download_ami.py [--limit N]

Outputs:
    benchmarks/data/ami/audio/   WAV files (16 kHz mono, headset mix)
    benchmarks/data/ami/rttm/    RTTM annotation files

Full download is ~12 GB.  Use --limit N for a smaller subset.

Manual fallback (if HF mirrors fail):
  - Audio:  https://groups.inf.ed.ac.uk/ami/corpus/ (IHM headset-mix WAVs)
  - RTTM:   https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/
    Download AMI_public_manual_1.6.1.zip and extract speaker-turn RTTM files.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "ami"

# Ordered list of HF repos to try — first success wins
HF_REPOS = [
    "diarizers-community/ami",             # widely mirrored
    "edinburghspeech/ami",                 # Edinburgh official mirror
]

# Meeting IDs in AMI (used to verify download completeness)
AMI_MEETING_IDS = [
    "EN2001a", "EN2001b", "EN2001d", "EN2001e",
    "EN2003a", "EN2004a", "EN2005a", "EN2006a", "EN2006b",
    "EN2009b", "EN2009c", "EN2009d",
    "ES2002a", "ES2002b", "ES2002c", "ES2002d",
    "ES2003a", "ES2003b", "ES2003c", "ES2003d",
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "ES2005a", "ES2005b", "ES2005c", "ES2005d",
    "ES2006a", "ES2006b", "ES2006c", "ES2006d",
    "ES2007a", "ES2007b", "ES2007c", "ES2007d",
    "ES2008a", "ES2008b", "ES2008c", "ES2008d",
    "ES2009a", "ES2009b", "ES2009c", "ES2009d",
    "ES2010a", "ES2010b", "ES2010c", "ES2010d",
    "ES2012a", "ES2012b", "ES2012c", "ES2012d",
    "ES2013a", "ES2013b", "ES2013c", "ES2013d",
    "ES2014a", "ES2014b", "ES2014c", "ES2014d",
    "ES2015a", "ES2015b", "ES2015c", "ES2015d",
    "IS1000a", "IS1000b", "IS1000c", "IS1000d",
    "IS1001a", "IS1001b", "IS1001c", "IS1001d",
    "IS1002b", "IS1002c", "IS1002d",
    "IS1003a", "IS1003b", "IS1003c", "IS1003d",
    "IS1004a", "IS1004b", "IS1004c", "IS1004d",
    "IS1005a", "IS1005b", "IS1005c",
    "IS1006a", "IS1006b", "IS1006c", "IS1006d",
    "IS1007a", "IS1007b", "IS1007c", "IS1007d",
    "TS3003a", "TS3003b", "TS3003c", "TS3003d",
    "TS3004a", "TS3004b", "TS3004c", "TS3004d",
    "TS3005a", "TS3005b", "TS3005c", "TS3005d",
    "TS3006a", "TS3006b", "TS3006c", "TS3006d",
    "TS3007a", "TS3007b", "TS3007c", "TS3007d",
    "TS3008a", "TS3008b", "TS3008c", "TS3008d",
    "TS3009a", "TS3009b", "TS3009c", "TS3009d",
    "TS3010a", "TS3010b", "TS3010c", "TS3010d",
    "TS3011a", "TS3011b", "TS3011c", "TS3011d",
    "TS3012a", "TS3012b", "TS3012c", "TS3012d",
]


def download_corpus(data_dir: Path, limit: int | None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed.\n"
            "Run: pip install huggingface_hub soundfile",
            file=sys.stderr,
        )
        sys.exit(1)

    audio_dir = data_dir / "audio"
    rttm_dir = data_dir / "rttm"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)

    local_dir = data_dir / "_hf_download"

    for repo in HF_REPOS:
        print(f"Trying {repo} …")
        try:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=str(local_dir),
                ignore_patterns=["*.zip", "*.tar.gz"],
            )
            print(f"Download succeeded from {repo}")
            break
        except Exception as exc:
            print(f"  WARNING: {repo} failed: {exc}", file=sys.stderr)
    else:
        print(
            "ERROR: All HF mirrors failed.\n"
            "Manual fallback:\n"
            "  1. Download IHM headset-mix WAVs from:\n"
            "     https://groups.inf.ed.ac.uk/ami/corpus/\n"
            "  2. Place WAVs in:  benchmarks/data/ami/audio/\n"
            "  3. Download RTTM annotations from:\n"
            "     https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/\n"
            "     (AMI_public_manual_1.6.1.zip → speaker-turn RTTM files)\n"
            "  4. Place *.rttm in: benchmarks/data/ami/rttm/",
            file=sys.stderr,
        )
        sys.exit(1)

    # Reorganise: separate WAVs and RTTMs into their target dirs
    wavs = list(local_dir.rglob("*.wav"))
    rttms = list(local_dir.rglob("*.rttm"))
    print(f"Found {len(wavs)} WAV, {len(rttms)} RTTM files in download")

    if not wavs and not rttms:
        print(
            "WARNING: No WAV or RTTM files found in the downloaded snapshot.\n"
            "The HF repo may have a different structure. Inspect the download at:\n"
            f"  {local_dir}\n"
            "and manually copy audio to benchmarks/data/ami/audio/ and\n"
            "RTTM files to benchmarks/data/ami/rttm/",
            file=sys.stderr,
        )

    if limit is not None:
        wavs = wavs[:limit]
        print(f"Limiting to {limit} WAV files")

    for wav in wavs:
        shutil.copy(wav, audio_dir / wav.name)
    for rttm in rttms:
        shutil.copy(rttm, rttm_dir / rttm.name)

    shutil.rmtree(local_dir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Keep only the first N WAV files (useful for quick iteration)",
    )
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_corpus(DATA_DIR, args.limit)

    rttm_count = len(list((DATA_DIR / "rttm").glob("*.rttm")))
    wav_count = len(list((DATA_DIR / "audio").glob("*.wav")))

    try:
        import soundfile as sf
        total_h = sum(sf.info(w).duration for w in (DATA_DIR / "audio").glob("*.wav")) / 3600
        print(f"\nAMI ready: {wav_count} WAV files ({total_h:.1f} h), {rttm_count} RTTM files")
    except ImportError:
        print(f"\nAMI ready: {wav_count} WAV files, {rttm_count} RTTM files")

    print(f"Location: {DATA_DIR}")

    # Warn about missing RTTM coverage
    wav_stems = {f.stem for f in (DATA_DIR / "audio").glob("*.wav")}
    rttm_stems = {f.stem for f in (DATA_DIR / "rttm").glob("*.rttm")}
    missing_rttm = wav_stems - rttm_stems
    if missing_rttm:
        print(
            f"\nWARNING: {len(missing_rttm)} WAV files have no matching RTTM — "
            "they will be skipped during benchmarking."
        )


if __name__ == "__main__":
    main()
