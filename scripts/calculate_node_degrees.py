#!/usr/bin/env python3
"""
Calculate node degrees from edge list file.

Reads a .edg file and calculates the total degree (in-degree + out-degree) for each node.
Saves results as a JSON file in the same directory.

Usage:
    python scripts/calculate_node_degrees.py <path/to/edges.edg>
    python scripts/calculate_node_degrees.py graphs/robokop_base_nonredundant_human_only/graph/edges.edg
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


def calculate_node_degrees(edges_file):
    """Calculate node degrees from edge list.

    Args:
        edges_file: Path to .edg file (tab-separated: source\ttarget)

    Returns:
        dict: Mapping from node_id to total degree
    """
    edges_path = Path(edges_file)

    if not edges_path.exists():
        raise FileNotFoundError(f"Edge file not found: {edges_file}")

    print(f"Reading edges from {edges_path.name}...")

    # Use defaultdict to count degrees
    node_degrees = defaultdict(int)
    edge_count = 0

    with open(edges_path, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split('\t')

            if len(parts) < 2:
                print(f"Warning: Line {line_num} has {len(parts)} columns, expected at least 2")
                continue

            source, target = parts[0], parts[1]

            # Increment degree for both source and target
            node_degrees[source] += 1
            node_degrees[target] += 1

            edge_count += 1

            # Progress indicator
            if edge_count % 1000000 == 0:
                print(f"  Processed {edge_count:,} edges, {len(node_degrees):,} unique nodes")

    print(f"\nProcessing complete:")
    print(f"  Total edges: {edge_count:,}")
    print(f"  Unique nodes: {len(node_degrees):,}")

    # Convert defaultdict to regular dict and sort by degree
    degrees_dict = dict(node_degrees)

    # Calculate statistics
    degrees = list(degrees_dict.values())
    min_degree = min(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    avg_degree = sum(degrees) / len(degrees) if degrees else 0

    print(f"  Degree range: {min_degree} - {max_degree}")
    print(f"  Average degree: {avg_degree:.2f}")

    return degrees_dict, {
        "edge_count": edge_count,
        "node_count": len(node_degrees),
        "min_degree": min_degree,
        "max_degree": max_degree,
        "average_degree": avg_degree
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate node degrees from edge list file"
    )
    parser.add_argument(
        "edges_file",
        help="Path to edges.edg file"
    )
    parser.add_argument(
        "--output",
        help="Output JSON file path (default: same directory as edges.edg)"
    )

    args = parser.parse_args()

    # Calculate degrees
    degrees, stats = calculate_node_degrees(args.edges_file)

    # Determine output path
    edges_path = Path(args.edges_file)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = edges_path.parent / "node_degrees.json"

    # Save results
    print(f"\nSaving node degrees to {output_path}...")

    output_data = {
        "source_file": str(edges_path),
        "statistics": stats,
        "node_degrees": degrees
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(degrees):,} node degrees to {output_path}")

    # Also save top 20 highest degree nodes for reference
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
    print(f"\nTop 20 highest degree nodes:")
    for node_id, degree in top_nodes:
        print(f"  {node_id}: {degree}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
