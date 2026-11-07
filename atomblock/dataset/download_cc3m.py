#!/usr/bin/env python3
"""Download the CC3M webdataset into a local cache directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pixparse/cc3m-wds.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    load_dataset("pixparse/cc3m-wds", cache_dir=str(args.cache_dir))


if __name__ == "__main__":
    main()
