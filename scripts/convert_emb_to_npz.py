#!/usr/bin/env python3
"""
Convert PecanPy .emb text format to compressed NumPy .npz format.

This script:
1. Reads embeddings from .emb file (text format)
2. Converts to NumPy arrays with float32 precision
3. Saves as compressed .npz file
4. Optionally removes the original .emb file

Usage:
    python scripts/convert_emb_to_npz.py <embeddings.emb> [--keep-original]
    python scripts/convert_emb_to_npz.py --all [--keep-original]
"""

import argparse
import numpy as np
import os
import sys
from pathlib import Path


def convert_emb_to_npz(emb_file, keep_original=False, verbose=True):
    """Convert a single .emb file to .npz format.

    Args:
        emb_file: Path to embeddings.emb file
        keep_original: If True, keep the original .emb file
        verbose: If True, print progress messages

    Returns:
        str: Path to created .npz file
    """
    emb_path = Path(emb_file)

    if not emb_path.exists():
        raise FileNotFoundError(f"File not found: {emb_file}")

    if not emb_path.suffix == '.emb':
        raise ValueError(f"Expected .emb file, got: {emb_file}")

    # Output path
    npz_path = emb_path.with_suffix('.npz')

    if verbose:
        print(f"Converting {emb_path.name}...")

    # Read header first to get dimensions
    with open(emb_path, 'r') as f:
        header = f.readline().strip().split()
        num_nodes = int(header[0])
        embedding_dim = int(header[1])

    if verbose:
        print(f"  {num_nodes:,} nodes × {embedding_dim} dimensions")

    # Pre-allocate numpy arrays for memory efficiency
    node_ids_array = np.empty(num_nodes, dtype='U50')  # Assume max 50 char IDs
    embeddings_array = np.empty((num_nodes, embedding_dim), dtype=np.float32)

    # Read embeddings directly into pre-allocated arrays
    with open(emb_path, 'r') as f:
        # Skip header
        f.readline()

        for i, line in enumerate(f):
            if i >= num_nodes:
                print(f"\nWarning: More lines than expected ({num_nodes})")
                break

            parts = line.strip().split()
            if len(parts) != embedding_dim + 1:
                print(f"\nWarning: Line {i+1} has {len(parts)-1} values, expected {embedding_dim}")
                continue

            node_ids_array[i] = parts[0]
            embeddings_array[i] = [float(x) for x in parts[1:]]

            # Progress indicator for large files
            if verbose and (i + 1) % 500000 == 0:
                print(f"  Progress: {i+1:,} / {num_nodes:,} nodes ({(i+1)/num_nodes*100:.1f}%)")

    # Save as compressed npz
    np.savez_compressed(
        npz_path,
        node_ids=node_ids_array,
        embeddings=embeddings_array,
        num_nodes=num_nodes,
        embedding_dim=embedding_dim
    )

    # Get file sizes
    emb_size = emb_path.stat().st_size
    npz_size = npz_path.stat().st_size
    compression_ratio = (1 - npz_size / emb_size) * 100

    if verbose:
        print(f" Done!")
        print(f"  Original:   {emb_size / 1024**3:8.2f} GB")
        print(f"  Compressed: {npz_size / 1024**3:8.2f} GB")
        print(f"  Saved:      {compression_ratio:8.1f}%")

    # Remove original if requested
    if not keep_original:
        if verbose:
            print(f"  Removing original .emb file...")
        emb_path.unlink()

    return str(npz_path)


def find_all_emb_files(base_dir="graphs"):
    """Find all .emb files in the graphs directory.

    Args:
        base_dir: Base directory to search (default: "graphs")

    Returns:
        list: List of paths to .emb files
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    return sorted(base_path.rglob("embeddings.emb"))


def main():
    parser = argparse.ArgumentParser(
        description="Convert PecanPy .emb files to compressed NumPy .npz format"
    )
    parser.add_argument(
        "emb_file",
        nargs="?",
        help="Path to embeddings.emb file (or use --all)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all .emb files in graphs/ directory"
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original .emb file after conversion"
    )
    parser.add_argument(
        "--base-dir",
        default="graphs",
        help="Base directory to search for .emb files (default: graphs)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.emb_file:
        parser.error("Must specify either emb_file or --all")

    if args.all and args.emb_file:
        parser.error("Cannot specify both emb_file and --all")

    # Convert files
    if args.all:
        emb_files = find_all_emb_files(args.base_dir)

        if not emb_files:
            print(f"No .emb files found in {args.base_dir}/")
            return 0

        print(f"Found {len(emb_files)} .emb files to convert")
        print("=" * 70)

        total_saved = 0
        total_original = 0

        for i, emb_file in enumerate(emb_files, 1):
            print(f"\n[{i}/{len(emb_files)}] {emb_file.relative_to(args.base_dir)}")

            try:
                emb_size = emb_file.stat().st_size
                npz_path = convert_emb_to_npz(emb_file, args.keep_original)
                npz_size = Path(npz_path).stat().st_size

                total_original += emb_size
                total_saved += (emb_size - npz_size)

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        print("\n" + "=" * 70)
        print(f"Conversion complete!")
        print(f"  Total original size: {total_original / 1024**3:8.2f} GB")
        print(f"  Total compressed:    {(total_original - total_saved) / 1024**3:8.2f} GB")
        print(f"  Total saved:         {total_saved / 1024**3:8.2f} GB ({total_saved/total_original*100:.1f}%)")

    else:
        convert_emb_to_npz(args.emb_file, args.keep_original)

    return 0


if __name__ == "__main__":
    sys.exit(main())
