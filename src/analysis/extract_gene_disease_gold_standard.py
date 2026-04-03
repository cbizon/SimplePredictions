#!/usr/bin/env python3
"""Extract a gene-disease gold standard directly from a KGX graph.

The gold standard is defined as all non-text-mined, non-subclass direct
Gene-DiseaseOrPhenotypicFeature edges in the input graph. Edges are collapsed
to unique (gene, disease) pairs, with predicate provenance retained.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import jsonlines


def load_gene_and_disease_nodes(nodes_file):
    """Load only the node IDs and labels relevant to gene-disease extraction."""
    genes = {}
    diseases = {}

    with jsonlines.open(nodes_file) as reader:
        for node in reader:
            node_id = node["id"]
            categories = set(node.get("category", []))
            label = node.get("name", node_id)

            if "biolink:Gene" in categories:
                genes[node_id] = label

            if "biolink:DiseaseOrPhenotypicFeature" in categories:
                diseases[node_id] = label

    return genes, diseases


def canonical_gene_disease_pair(edge, gene_nodes, disease_nodes):
    """Return a canonical (gene_id, disease_id) tuple or None."""
    subject = edge.get("subject")
    obj = edge.get("object")

    if subject in gene_nodes and obj in disease_nodes:
        return subject, obj

    if obj in gene_nodes and subject in disease_nodes:
        return obj, subject

    return None


def extract_gene_disease_gold_standard(nodes_file, edges_file, output_csv, summary_json=None):
    """Extract unique gene-disease pairs from the input graph."""
    gene_nodes, disease_nodes = load_gene_and_disease_nodes(nodes_file)
    pair_to_predicates = defaultdict(set)
    predicate_counts = Counter()
    raw_edge_count = 0

    with jsonlines.open(edges_file) as reader:
        for edge in reader:
            if edge.get("predicate") == "biolink:subclass_of":
                continue

            if edge.get("agent_type") == "text_mining_agent":
                continue

            pair = canonical_gene_disease_pair(edge, gene_nodes, disease_nodes)
            if pair is None:
                continue

            predicate = edge.get("predicate", "unknown")
            pair_to_predicates[pair].add(predicate)
            predicate_counts[predicate] += 1
            raw_edge_count += 1

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "gene_id",
                "gene_label",
                "disease_id",
                "disease_label",
                "pair",
                "predicate_count",
                "predicates",
            ],
        )
        writer.writeheader()

        for gene_id, disease_id in sorted(pair_to_predicates):
            predicates = sorted(pair_to_predicates[(gene_id, disease_id)])
            writer.writerow(
                {
                    "gene_id": gene_id,
                    "gene_label": gene_nodes.get(gene_id, gene_id),
                    "disease_id": disease_id,
                    "disease_label": disease_nodes.get(disease_id, disease_id),
                    "pair": f"{gene_id}|{disease_id}",
                    "predicate_count": len(predicates),
                    "predicates": ";".join(predicates),
                }
            )

    summary = {
        "nodes_file": str(nodes_file),
        "edges_file": str(edges_file),
        "output_csv": str(output_csv),
        "gene_nodes_available": len(gene_nodes),
        "disease_nodes_available": len(disease_nodes),
        "raw_gene_disease_edges": raw_edge_count,
        "unique_gene_disease_pairs": len(pair_to_predicates),
        "predicate_counts": dict(sorted(predicate_counts.items(), key=lambda item: item[1], reverse=True)),
    }

    if summary_json is not None:
        summary_path = Path(summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as handle:
            json.dump(summary, handle, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Extract non-text-mined gene-disease gold-standard pairs from a graph")
    parser.add_argument("--input-dir", required=True, help="Directory containing nodes.jsonl and edges.jsonl")
    parser.add_argument("--nodes-filename", default="nodes.jsonl", help="Nodes filename inside --input-dir")
    parser.add_argument("--edges-filename", default="edges.jsonl", help="Edges filename inside --input-dir")
    parser.add_argument("--output-csv", required=True, help="CSV path for the extracted unique gene-disease pairs")
    parser.add_argument("--summary-json", help="Optional JSON path for extraction summary")

    args = parser.parse_args()

    nodes_file = Path(args.input_dir) / args.nodes_filename
    edges_file = Path(args.input_dir) / args.edges_filename

    summary = extract_gene_disease_gold_standard(
        nodes_file=nodes_file,
        edges_file=edges_file,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
    )

    print(f"Extracted {summary['unique_gene_disease_pairs']:,} unique gene-disease pairs")
    print(f"Raw qualifying gene-disease edges: {summary['raw_gene_disease_edges']:,}")


if __name__ == "__main__":
    main()
