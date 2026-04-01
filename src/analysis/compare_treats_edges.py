#!/usr/bin/env python
"""
Compare treats edges between ground truth and input graph.

This script analyzes the overlap between drug-disease treats edges from:
1. Ground truth indications CSV file
2. Input graph edges JSONL file

Outputs:
- 2x2 contingency table showing in/out counts
- CSV file with ground truth pairs NOT in graph (with names)
- CSV file with graph pairs NOT in ground truth (with names)
"""

import argparse
import csv
import jsonlines
from pathlib import Path
from typing import Dict, Set, Tuple
from collections import defaultdict


def load_ground_truth(csv_file: str) -> Tuple[Set[Tuple[str, str]], Dict[str, str]]:
    """
    Load ground truth drug-disease pairs from CSV file.

    Returns:
        - Set of (drug_id, disease_id) tuples
        - Dict mapping IDs to names for both drugs and diseases
    """
    pairs = set()
    id_to_name = {}

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug_id = row['final normalized drug id']
            disease_id = row['final normalized disease id']
            drug_name = row['final normalized drug label']
            disease_name = row['final normalized disease label']

            pairs.add((drug_id, disease_id))
            id_to_name[drug_id] = drug_name
            id_to_name[disease_id] = disease_name

    return pairs, id_to_name


def load_nodes(nodes_file: str) -> Dict[str, str]:
    """
    Load node ID to name mapping from nodes JSONL file.

    Returns:
        Dict mapping node ID to node name
    """
    id_to_name = {}

    with jsonlines.open(nodes_file) as reader:
        for node in reader:
            node_id = node['id']
            node_name = node.get('name', node_id)
            id_to_name[node_id] = node_name

    return id_to_name


def load_graph_treats_edges(edges_file: str) -> Set[Tuple[str, str]]:
    """
    Load treats edges from graph edges JSONL file.

    Returns:
        Set of (subject_id, object_id) tuples for treats edges
    """
    treats_edges = set()

    with jsonlines.open(edges_file) as reader:
        for edge in reader:
            predicate = edge.get('predicate', '')
            if predicate == 'biolink:treats':
                subject = edge['subject']
                obj = edge['object']
                treats_edges.add((subject, obj))

    return treats_edges


def create_2x2_table(gt_pairs: Set[Tuple[str, str]],
                     graph_pairs: Set[Tuple[str, str]]) -> Dict[str, int]:
    """
    Create 2x2 contingency table.

    Returns:
        Dict with counts for:
        - in_gt_in_graph: Pairs in both ground truth and graph
        - in_gt_not_in_graph: Pairs in ground truth but not in graph
        - not_in_gt_in_graph: Pairs in graph but not in ground truth
    """
    in_both = gt_pairs & graph_pairs
    in_gt_only = gt_pairs - graph_pairs
    in_graph_only = graph_pairs - gt_pairs

    return {
        'in_gt_in_graph': len(in_both),
        'in_gt_not_in_graph': len(in_gt_only),
        'not_in_gt_in_graph': len(in_graph_only)
    }


def analyze_entity_presence(pairs: Set[Tuple[str, str]],
                           gt_pairs: Set[Tuple[str, str]],
                           graph_pairs: Set[Tuple[str, str]]) -> Dict[str, int]:
    """
    Analyze whether drugs and diseases from pairs exist in the other set.

    Args:
        pairs: Set of pairs to analyze
        gt_pairs: Ground truth pairs
        graph_pairs: Graph pairs

    Returns:
        Dict with counts for each combination:
        - drug_in_disease_in: Both drug and disease appear in other set
        - drug_in_disease_out: Drug appears but disease doesn't
        - drug_out_disease_in: Disease appears but drug doesn't
        - drug_out_disease_out: Neither drug nor disease appear
    """
    # Extract drugs and diseases from each set
    gt_drugs = {drug for drug, _ in gt_pairs}
    gt_diseases = {disease for _, disease in gt_pairs}
    graph_drugs = {drug for drug, _ in graph_pairs}
    graph_diseases = {disease for _, disease in graph_pairs}

    # Determine which set to check against
    if pairs == (gt_pairs - graph_pairs):
        # Analyzing GT pairs not in graph, check against graph
        other_drugs = graph_drugs
        other_diseases = graph_diseases
    else:
        # Analyzing graph pairs not in GT, check against GT
        other_drugs = gt_drugs
        other_diseases = gt_diseases

    counts = {
        'drug_in_disease_in': 0,
        'drug_in_disease_out': 0,
        'drug_out_disease_in': 0,
        'drug_out_disease_out': 0
    }

    for drug, disease in pairs:
        drug_in = drug in other_drugs
        disease_in = disease in other_diseases

        if drug_in and disease_in:
            counts['drug_in_disease_in'] += 1
        elif drug_in and not disease_in:
            counts['drug_in_disease_out'] += 1
        elif not drug_in and disease_in:
            counts['drug_out_disease_in'] += 1
        else:
            counts['drug_out_disease_out'] += 1

    return counts


def write_missing_pairs(pairs: Set[Tuple[str, str]],
                        id_to_name: Dict[str, str],
                        output_file: str,
                        header_prefix: str,
                        gt_pairs: Set[Tuple[str, str]],
                        graph_pairs: Set[Tuple[str, str]]):
    """
    Write pairs to CSV file with human-readable names and entity presence info.

    Args:
        pairs: Set of (subject_id, object_id) tuples
        id_to_name: Dict mapping IDs to names
        output_file: Output CSV file path
        header_prefix: Prefix for column headers (e.g., "drug" or "subject")
        gt_pairs: Ground truth pairs (for checking presence)
        graph_pairs: Graph pairs (for checking presence)
    """
    # Extract drugs and diseases from each set
    gt_drugs = {drug for drug, _ in gt_pairs}
    gt_diseases = {disease for _, disease in gt_pairs}
    graph_drugs = {drug for drug, _ in graph_pairs}
    graph_diseases = {disease for _, disease in graph_pairs}

    # Determine which set to check against
    if pairs == (gt_pairs - graph_pairs):
        # Analyzing GT pairs not in graph, check against graph
        other_drugs = graph_drugs
        other_diseases = graph_diseases
    else:
        # Analyzing graph pairs not in GT, check against GT
        other_drugs = gt_drugs
        other_diseases = gt_diseases

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            f'{header_prefix}_id',
            f'{header_prefix}_name',
            'disease_id',
            'disease_name',
            'drug_in_other',
            'disease_in_other'
        ])

        # Sort pairs, handling None values
        sorted_pairs = sorted(pairs, key=lambda x: (x[0] or '', x[1] or ''))

        for subj_id, obj_id in sorted_pairs:
            subj_name = id_to_name.get(subj_id, subj_id) if subj_id else 'UNKNOWN'
            obj_name = id_to_name.get(obj_id, obj_id) if obj_id else 'UNKNOWN'
            drug_in = 'yes' if subj_id in other_drugs else 'no'
            disease_in = 'yes' if obj_id in other_diseases else 'no'
            writer.writerow([subj_id, subj_name, obj_id, obj_name, drug_in, disease_in])


def main():
    parser = argparse.ArgumentParser(
        description='Compare treats edges between ground truth and graph'
    )
    parser.add_argument(
        '--ground-truth',
        required=True,
        help='Path to ground truth CSV file'
    )
    parser.add_argument(
        '--graph-dir',
        required=True,
        help='Path to graph directory containing nodes.jsonl and edges.jsonl'
    )
    parser.add_argument(
        '--output-dir',
        default='analysis_results',
        help='Output directory for results (default: analysis_results)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading ground truth...")
    gt_pairs, gt_id_to_name = load_ground_truth(args.ground_truth)
    print(f"  Found {len(gt_pairs)} ground truth pairs")

    print("\nLoading graph nodes...")
    nodes_file = Path(args.graph_dir) / 'nodes.jsonl'
    graph_id_to_name = load_nodes(str(nodes_file))
    print(f"  Found {len(graph_id_to_name)} nodes")

    print("\nLoading graph treats edges...")
    edges_file = Path(args.graph_dir) / 'edges.jsonl'
    graph_pairs = load_graph_treats_edges(str(edges_file))
    print(f"  Found {len(graph_pairs)} treats edges")

    # Merge name mappings (prefer ground truth names)
    combined_id_to_name = {**graph_id_to_name, **gt_id_to_name}

    # Create 2x2 table
    print("\nCreating 2x2 contingency table...")
    table = create_2x2_table(gt_pairs, graph_pairs)

    # Print table
    print("\n" + "="*60)
    print("2x2 CONTINGENCY TABLE")
    print("="*60)
    print(f"                           | In Graph  | Not in Graph |")
    print(f"---------------------------|-----------|--------------|")
    print(f"In Ground Truth           | {table['in_gt_in_graph']:9,} | {table['in_gt_not_in_graph']:12,} |")
    print(f"Not in Ground Truth       | {table['not_in_gt_in_graph']:9,} |      N/A     |")
    print("="*60)

    # Calculate percentages
    total_gt = len(gt_pairs)
    total_graph = len(graph_pairs)

    print(f"\nGround Truth Coverage:")
    print(f"  {table['in_gt_in_graph']}/{total_gt} ({100*table['in_gt_in_graph']/total_gt:.1f}%) in graph")
    print(f"  {table['in_gt_not_in_graph']}/{total_gt} ({100*table['in_gt_not_in_graph']/total_gt:.1f}%) missing from graph")

    print(f"\nGraph Treats Edges:")
    print(f"  {table['in_gt_in_graph']}/{total_graph} ({100*table['in_gt_in_graph']/total_graph:.1f}%) in ground truth")
    print(f"  {table['not_in_gt_in_graph']}/{total_graph} ({100*table['not_in_gt_in_graph']/total_graph:.1f}%) not in ground truth")

    # Analyze entity presence for mismatches
    print("\nAnalyzing entity presence in mismatches...")

    in_gt_only = gt_pairs - graph_pairs
    gt_entity_analysis = analyze_entity_presence(in_gt_only, gt_pairs, graph_pairs)

    in_graph_only = graph_pairs - gt_pairs
    graph_entity_analysis = analyze_entity_presence(in_graph_only, gt_pairs, graph_pairs)

    print("\nGround Truth pairs NOT in Graph - Entity Presence:")
    print(f"  Drug IN graph, Disease IN graph: {gt_entity_analysis['drug_in_disease_in']:,}")
    print(f"  Drug IN graph, Disease NOT in graph: {gt_entity_analysis['drug_in_disease_out']:,}")
    print(f"  Drug NOT in graph, Disease IN graph: {gt_entity_analysis['drug_out_disease_in']:,}")
    print(f"  Drug NOT in graph, Disease NOT in graph: {gt_entity_analysis['drug_out_disease_out']:,}")

    print("\nGraph treats edges NOT in Ground Truth - Entity Presence:")
    print(f"  Drug IN GT, Disease IN GT: {graph_entity_analysis['drug_in_disease_in']:,}")
    print(f"  Drug IN GT, Disease NOT in GT: {graph_entity_analysis['drug_in_disease_out']:,}")
    print(f"  Drug NOT in GT, Disease IN GT: {graph_entity_analysis['drug_out_disease_in']:,}")
    print(f"  Drug NOT in GT, Disease NOT in GT: {graph_entity_analysis['drug_out_disease_out']:,}")

    # Write missing pairs files
    print("\nWriting results files...")

    gt_only_file = output_dir / 'in_groundtruth_not_in_graph.csv'
    write_missing_pairs(in_gt_only, combined_id_to_name, str(gt_only_file), 'drug', gt_pairs, graph_pairs)
    print(f"  {gt_only_file}")

    graph_only_file = output_dir / 'in_graph_not_in_groundtruth.csv'
    write_missing_pairs(in_graph_only, combined_id_to_name, str(graph_only_file), 'drug', gt_pairs, graph_pairs)
    print(f"  {graph_only_file}")

    # Save table as CSV
    table_file = output_dir / 'contingency_table.csv'
    with open(table_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'In Graph', 'Not in Graph', 'Total'])
        writer.writerow([
            'In Ground Truth',
            table['in_gt_in_graph'],
            table['in_gt_not_in_graph'],
            total_gt
        ])
        writer.writerow([
            'Not in Ground Truth',
            table['not_in_gt_in_graph'],
            'N/A',
            'N/A'
        ])
    print(f"  {table_file}")

    # Save entity presence analysis
    entity_analysis_file = output_dir / 'entity_presence_analysis.csv'
    with open(entity_analysis_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Source', 'Drug in Other', 'Disease in Other', 'Count'])
        writer.writerow(['GT not in Graph', 'yes', 'yes', gt_entity_analysis['drug_in_disease_in']])
        writer.writerow(['GT not in Graph', 'yes', 'no', gt_entity_analysis['drug_in_disease_out']])
        writer.writerow(['GT not in Graph', 'no', 'yes', gt_entity_analysis['drug_out_disease_in']])
        writer.writerow(['GT not in Graph', 'no', 'no', gt_entity_analysis['drug_out_disease_out']])
        writer.writerow(['Graph not in GT', 'yes', 'yes', graph_entity_analysis['drug_in_disease_in']])
        writer.writerow(['Graph not in GT', 'yes', 'no', graph_entity_analysis['drug_in_disease_out']])
        writer.writerow(['Graph not in GT', 'no', 'yes', graph_entity_analysis['drug_out_disease_in']])
        writer.writerow(['Graph not in GT', 'no', 'no', graph_entity_analysis['drug_out_disease_out']])
    print(f"  {entity_analysis_file}")

    print("\nDone!")


if __name__ == '__main__':
    main()
