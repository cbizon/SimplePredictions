#!/usr/bin/env python3
"""Tests for extracting gene-disease gold-standard pairs from a graph."""

import csv
import json
import os
import tempfile

import jsonlines

from src.analysis.extract_gene_disease_gold_standard import extract_gene_disease_gold_standard


def test_extract_gene_disease_gold_standard_filters_and_canonicalizes():
    """Extractor should drop text-mined/subclass edges and canonicalize disease->gene edges."""
    nodes = [
        {"id": "HGNC:1", "name": "GENE1", "category": ["biolink:Gene"]},
        {"id": "MONDO:1", "name": "Disease 1", "category": ["biolink:DiseaseOrPhenotypicFeature"]},
        {"id": "MONDO:2", "name": "Disease 2", "category": ["biolink:DiseaseOrPhenotypicFeature"]},
        {"id": "CHEBI:1", "name": "Chemical 1", "category": ["biolink:ChemicalEntity"]},
    ]

    edges = [
        {"subject": "HGNC:1", "object": "MONDO:1", "predicate": "biolink:genetically_associated_with", "agent_type": "manual_agent"},
        {"subject": "MONDO:1", "object": "HGNC:1", "predicate": "biolink:causes", "agent_type": "manual_agent"},
        {"subject": "HGNC:1", "object": "MONDO:2", "predicate": "biolink:has_phenotype", "agent_type": "text_mining_agent"},
        {"subject": "HGNC:1", "object": "MONDO:2", "predicate": "biolink:subclass_of", "agent_type": "manual_agent"},
        {"subject": "CHEBI:1", "object": "MONDO:1", "predicate": "biolink:treats", "agent_type": "manual_agent"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        nodes_file = os.path.join(tmpdir, "nodes.jsonl")
        edges_file = os.path.join(tmpdir, "edges.jsonl")
        output_csv = os.path.join(tmpdir, "gold.csv")
        summary_json = os.path.join(tmpdir, "summary.json")

        with jsonlines.open(nodes_file, "w") as writer:
            for node in nodes:
                writer.write(node)

        with jsonlines.open(edges_file, "w") as writer:
            for edge in edges:
                writer.write(edge)

        summary = extract_gene_disease_gold_standard(
            nodes_file=nodes_file,
            edges_file=edges_file,
            output_csv=output_csv,
            summary_json=summary_json,
        )

        assert summary["raw_gene_disease_edges"] == 2
        assert summary["unique_gene_disease_pairs"] == 1
        assert summary["predicate_counts"]["biolink:genetically_associated_with"] == 1
        assert summary["predicate_counts"]["biolink:causes"] == 1

        with open(output_csv, newline="") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 1
        assert rows[0]["gene_id"] == "HGNC:1"
        assert rows[0]["disease_id"] == "MONDO:1"
        assert rows[0]["predicate_count"] == "2"
        assert rows[0]["predicates"] == "biolink:causes;biolink:genetically_associated_with"

        with open(summary_json) as handle:
            summary_from_disk = json.load(handle)

        assert summary_from_disk["unique_gene_disease_pairs"] == 1
