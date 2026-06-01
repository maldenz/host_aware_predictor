from __future__ import annotations

import argparse
from pathlib import Path

from host_aware_predictor.data.dragonn_mpra import find_split_file, print_hdf5_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect MPRA-DragoNN HDF5 split files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/mpra_dragonn"))
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for split in args.splits:
        path = find_split_file(args.data_dir, split)
        print_hdf5_summary(path)
        print()


if __name__ == "__main__":
    main()
