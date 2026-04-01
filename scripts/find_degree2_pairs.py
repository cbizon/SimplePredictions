#!/usr/bin/env python3
"""
Find degree-2 nodes and identify the pairs they connect.

A degree-2 node connects exactly two other nodes. This script:
1. Finds all degree-2 nodes from node_degrees.json
2. Identifies the pairs of nodes each degree-2 node connects
3. Counts how many degree-2 nodes connect each pair
4. Looks up node names from nodes.jsonl
5. Outputs results as a TSV file

Usage:
    python scripts/find_degree2_pairs.py <graph_dir>
    python scripts/find_degree2_pairs.py graphs/robokop_base_nonredundant_human_only/graph
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import jsonlines


def find_degree2_nodes(degrees_file):
    """Find all nodes with degree 2.

    Args:
        degrees_file: Path to node_degrees.json

    Returns:
        set: Node IDs with degree 2
    """
    print(f"Loading node degrees from {degrees_file}...")

    with open(degrees_file, 'r') as f:
        data = json.load(f)

    node_degrees = data['node_degrees']

    # Find degree-2 nodes
    degree2_nodes = {node_id for node_id, degree in node_degrees.items() if degree == 2}

    print(f"  Found {len(degree2_nodes):,} degree-2 nodes out of {len(node_degrees):,} total nodes")

    return degree2_nodes


def find_pairs_connected_by_degree2_nodes(edges_file, degree2_nodes):
    """Find pairs of nodes connected by degree-2 nodes.

    Args:
        edges_file: Path to edges.edg file
        degree2_nodes: Set of degree-2 node IDs

    Returns:
        dict: Mapping from (node1, node2) tuple to count of degree-2 nodes connecting them
    """
    print(f"\nScanning edges to find pairs connected by degree-2 nodes...")

    # Map each degree-2 node to the nodes it connects
    degree2_connections = defaultdict(set)

    edge_count = 0
    with open(edges_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            source, target = parts[0], parts[1]

            # If source is degree-2, track what it connects to
            if source in degree2_nodes:
                degree2_connections[source].add(target)

            # If target is degree-2, track what it connects to
            if target in degree2_nodes:
                degree2_connections[target].add(source)

            edge_count += 1

            if edge_count % 1000000 == 0:
                print(f"  Processed {edge_count:,} edges...")

    print(f"  Total edges processed: {edge_count:,}")
    print(f"  Degree-2 nodes with connections found: {len(degree2_connections):,}")

    # Now find pairs - each degree-2 node should connect exactly 2 nodes
    pair_counts = defaultdict(int)
    invalid_count = 0

    for deg2_node, connected_nodes in degree2_connections.items():
        if len(connected_nodes) != 2:
            # This shouldn't happen for true degree-2 nodes
            invalid_count += 1
            continue

        # Sort the pair to ensure consistent ordering (node1, node2)
        node1, node2 = sorted(connected_nodes)
        pair_counts[(node1, node2)] += 1

    print(f"  Valid degree-2 node pairs: {len(pair_counts):,}")
    if invalid_count > 0:
        print(f"  Warning: {invalid_count} degree-2 nodes had != 2 connections")

    return pair_counts


def load_node_names(nodes_file):
    """Load node names from nodes.jsonl file.

    Args:
        nodes_file: Path to nodes.jsonl

    Returns:
        dict: Mapping from node_id to node name
    """
    print(f"\nLoading node names from {nodes_file}...")

    node_names = {}
    count = 0

    with jsonlines.open(nodes_file) as reader:
        for node in reader:
            node_id = node.get('id')
            name = node.get('name', node_id)  # Use ID if no name
            if node_id:
                node_names[node_id] = name
                count += 1

            if count % 100000 == 0:
                print(f"  Loaded {count:,} node names...")

    print(f"  Total node names loaded: {len(node_names):,}")

    return node_names


def save_pairs_tsv(pair_counts, node_names, output_file):
    """Save pairs to TSV file with names.

    Args:
        pair_counts: Dict mapping (node1, node2) to count
        node_names: Dict mapping node_id to name
        output_file: Path to output TSV file
    """
    print(f"\nSaving pairs to {output_file}...")

    # Sort by count (descending)
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

    with open(output_file, 'w') as f:
        # Write header
        f.write("node1_id\tnode1_name\tnode2_id\tnode2_name\tdegree2_count\n")

        # Write pairs
        for (node1, node2), count in sorted_pairs:
            node1_name = node_names.get(node1, node1)
            node2_name = node_names.get(node2, node2)

            f.write(f"{node1}\t{node1_name}\t{node2}\t{node2_name}\t{count}\n")

    print(f"  Saved {len(sorted_pairs):,} pairs")

    # Show top 10
    print(f"\nTop 10 pairs by degree-2 node count:")
    for (node1, node2), count in sorted_pairs[:10]:
        node1_name = node_names.get(node1, node1)
        node2_name = node_names.get(node2, node2)
        print(f"  {count:5d}  {node1} ({node1_name[:40]}...) <-> {node2} ({node2_name[:40]}...)")


def main():
    parser = argparse.ArgumentParser(
        description="Find pairs of nodes connected by degree-2 nodes"
    )
    parser.add_argument(
        "graph_dir",
        help="Path to graph directory (e.g., graphs/robokop_base_nonredundant_human_only/graph)"
    )
    parser.add_argument(
        "--output",
        help="Output TSV file path (default: <graph_dir>/degree2_pairs.tsv)"
    )

    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)

    # Check required files exist
    degrees_file = graph_dir / "node_degrees.json"
    edges_file = graph_dir / "edges.edg"

    # Try to find nodes.jsonl - could be in graph dir or input_graphs
    nodes_file = graph_dir / "nodes.jsonl"
    if not nodes_file.exists():
        # Try input_graphs/robokop_base_nonredundant/nodes.jsonl
        nodes_file = Path("input_graphs/robokop_base_nonredundant/nodes.jsonl")

    if not degrees_file.exists():
        print(f"Error: node_degrees.json not found in {graph_dir}")
        print(f"Run: python scripts/calculate_node_degrees.py {edges_file}")
        return 1

    if not edges_file.exists():
        print(f"Error: edges.edg not found: {edges_file}")
        return 1

    if not nodes_file.exists():
        print(f"Error: nodes.jsonl not found")
        print(f"Tried: {graph_dir / 'nodes.jsonl'}")
        print(f"Tried: input_graphs/robokop_base_nonredundant/nodes.jsonl")
        return 1

    # Determine output path
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = graph_dir / "degree2_pairs.tsv"

    # Step 1: Find degree-2 nodes
    degree2_nodes = find_degree2_nodes(degrees_file)

    if not degree2_nodes:
        print("No degree-2 nodes found!")
        return 0

    # Step 2: Find pairs connected by degree-2 nodes
    pair_counts = find_pairs_connected_by_degree2_nodes(edges_file, degree2_nodes)

    if not pair_counts:
        print("No pairs found!")
        return 0

    # Step 3: Load node names
    node_names = load_node_names(nodes_file)

    # Step 4: Save results
    save_pairs_tsv(pair_counts, node_names, output_file)

    print(f"\nDone! Results saved to {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
