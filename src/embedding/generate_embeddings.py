#!/usr/bin/env python3
"""Generate node embeddings with automatic versioning and provenance tracking.

This script generates node2vec embeddings using pecanpy and automatically manages
versioned output directories with full provenance metadata.
"""
import os
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path


def get_next_embedding_version(embeddings_dir):
    """Find the next available embedding version number.
    
    Args:
        embeddings_dir: Base embeddings directory
        
    Returns:
        int: Next available version number
    """
    if not os.path.exists(embeddings_dir):
        return 0
    
    existing_versions = []
    for item in os.listdir(embeddings_dir):
        if item.startswith("embeddings_") and os.path.isdir(os.path.join(embeddings_dir, item)):
            try:
                version_num = int(item.split("_")[1])
                existing_versions.append(version_num)
            except (IndexError, ValueError):
                continue
    
    return max(existing_versions) + 1 if existing_versions else 0


def count_edges_and_nodes(graph_file):
    """Count edges and unique nodes in the graph file.
    
    Args:
        graph_file: Path to .edg file
        
    Returns:
        tuple: (edge_count, node_count)
    """
    nodes = set()
    edge_count = 0
    
    with open(graph_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                nodes.add(parts[0])
                nodes.add(parts[1])
                edge_count += 1
    
    return edge_count, len(nodes)


def generate_embeddings(graph_file, 
                       dimensions=512,
                       walk_length=30,
                       num_walks=10,
                       window_size=10,
                       p=1,
                       q=1,
                       workers=4):
    """Generate node2vec embeddings with automatic versioning and provenance.
    
    Args:
        graph_file: Path to input .edg file
        dimensions: Embedding dimensions
        walk_length: Length of random walks
        num_walks: Number of walks per node
        window_size: Context window size
        p: Return parameter
        q: In-out parameter
        workers: Number of parallel workers
        
    Returns:
        str: Path to the generated embeddings directory
    """
    # Validate input file
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")
    
    # Determine output directory structure
    graph_dir = os.path.dirname(graph_file)  # e.g., graphs/robokop_base/CCDD/graph
    base_dir = os.path.dirname(graph_dir)    # e.g., graphs/robokop_base/CCDD
    embeddings_dir = os.path.join(base_dir, "embeddings")
    
    # Create embeddings directory if it doesn't exist
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Get next version number
    version = get_next_embedding_version(embeddings_dir)
    version_dir = os.path.join(embeddings_dir, f"embeddings_{version}")
    os.makedirs(version_dir, exist_ok=True)
    
    # Output file path
    output_file = os.path.join(version_dir, "embeddings.emb")
    
    # Count edges and nodes
    edge_count, node_count = count_edges_and_nodes(graph_file)
    
    print(f"Generating embeddings version {version}")
    print(f"Input: {graph_file} ({edge_count} edges, {node_count} nodes)")
    print(f"Output: {output_file}")
    print(f"Parameters: dim={dimensions}, walk_len={walk_length}, num_walks={num_walks}")
    
    # Build pecanpy command
    cmd = [
        "pecanpy",
        "--input", graph_file,
        "--output", output_file,
        "--dimensions", str(dimensions),
        "--walk-length", str(walk_length),
        "--num-walks", str(num_walks),
        "--window-size", str(window_size),
        "--p", str(p),
        "--q", str(q),
        "--workers", str(workers)
    ]
    
    # Record start time
    start_time = datetime.now()
    
    # Run pecanpy
    try:
        print("Running pecanpy...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Pecanpy completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running pecanpy: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    
    # Record end time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Create provenance metadata
    provenance = {
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "script": "generate_embeddings.py",
        "version": f"embeddings_{version}",
        "algorithm": "node2vec",
        "tool": "pecanpy",
        "input_graph_file": graph_file,
        "output_embeddings_file": output_file,
        "parameters": {
            "dimensions": dimensions,
            "walk_length": walk_length,
            "num_walks": num_walks,
            "window_size": window_size,
            "p": p,
            "q": q,
            "workers": workers
        },
        "graph_info": {
            "edge_count": edge_count,
            "node_count": node_count
        },
        "pecanpy_output": {
            "stdout": result.stdout,
            "stderr": result.stderr
        },
        "description": f"Node2vec embeddings version {version} generated from {os.path.basename(graph_file)}"
    }
    
    # Save provenance file
    provenance_file = os.path.join(version_dir, "provenance.json")
    with open(provenance_file, 'w') as f:
        json.dump(provenance, f, indent=2)
    
    print(f"Provenance saved: {provenance_file}")
    print(f"Embeddings generated: {output_file}")
    print(f"Duration: {duration:.2f} seconds")
    
    return version_dir


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Generate node2vec embeddings with versioning")
    parser.add_argument("--graph-file", required=True,
                       help="Path to input .edg file")
    parser.add_argument("--dimensions", type=int, default=512,
                       help="Embedding dimensions (default: 512)")
    parser.add_argument("--walk-length", type=int, default=30,
                       help="Length of random walks (default: 30)")
    parser.add_argument("--num-walks", type=int, default=10,
                       help="Number of walks per node (default: 10)")
    parser.add_argument("--window-size", type=int, default=10,
                       help="Context window size (default: 10)")
    parser.add_argument("--p", type=float, default=1,
                       help="Return parameter (default: 1)")
    parser.add_argument("--q", type=float, default=1,
                       help="In-out parameter (default: 1)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers (default: 4)")
    
    args = parser.parse_args()
    
    version_dir = generate_embeddings(
        graph_file=args.graph_file,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        window_size=args.window_size,
        p=args.p,
        q=args.q,
        workers=args.workers
    )
    
    print(f"\nEmbedding generation complete!")
    print(f"Output directory: {version_dir}")


if __name__ == "__main__":
    main()