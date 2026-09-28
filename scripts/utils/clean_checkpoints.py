#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Strip training-only fields from Lightning checkpoints, keeping only what is needed for inference.

Retained fields:
    epoch, global_step, pytorch-lightning_version, state_dict

Removed fields:
    loops, callbacks, optimizer_states, lr_schedulers, current_model_state, averaging_state

This typically reduces checkpoint size by ~4x (e.g. 1.7 GB -> 450 MB).

Usage:
    python scripts/clean_checkpoints.py ckpts/*.ckpt
    python scripts/clean_checkpoints.py ckpts/*.ckpt --in-place
"""

import argparse
import shutil
from pathlib import Path

import torch

KEEP = {"epoch", "global_step", "pytorch-lightning_version", "state_dict"}


def clean_checkpoint(src: Path, dst: Path) -> None:
    print("  Loading ...")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    removed = sorted(set(ckpt.keys()) - KEEP)
    cleaned = {k: v for k, v in ckpt.items() if k in KEEP}

    print(f"  Removed fields: {removed}")
    torch.save(cleaned, dst)
    print(f"  Saved to {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "checkpoint", nargs="+", type=Path, help="checkpoint file(s) to clean"
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="overwrite originals (backs up to *.ckpt.bak first)",
    )
    parser.add_argument(
        "--suffix",
        default="-cleaned",
        help="output filename suffix when not --in-place (default: -cleaned)",
    )
    args = parser.parse_args()

    for src in args.checkpoint:
        if not src.exists():
            print(f"ERROR: {src} not found, skipping")
            continue

        if args.in_place:
            backup = src.with_suffix(".ckpt.bak")
            shutil.copy2(src, backup)

            print(f"\n{src.name}  (backup -> {backup.name})")
            dst = src
        else:
            dst = src.with_name(src.stem + args.suffix + src.suffix)
            print(f"\n{src.name}  ->  {dst.name}")

        clean_checkpoint(src, dst)


if __name__ == "__main__":
    main()
