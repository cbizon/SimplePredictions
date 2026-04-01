#!/usr/bin/env python3
"""Node Normalization API client for taxa lookups.

This module provides utilities to interact with the Node Normalization API
to look up taxa information for biological entities (genes, proteins, etc.).
"""
import requests
from typing import Set, Dict, List, Optional
import time


class NodeNormalizationClient:
    """Client for Node Normalization API."""

    def __init__(self, base_url: str = "https://nodenormalization-sri.renci.org"):
        """Initialize Node Normalization client.

        Args:
            base_url: Base URL for the Node Normalization API
        """
        self.base_url = base_url.rstrip('/')
        self.normalize_url = f"{self.base_url}/get_normalized_nodes"

    def get_normalized_nodes_batch(self, curies: List[str], batch_size: int = 10000, max_retries: int = 5) -> Dict[str, Dict]:
        """Get normalized nodes for a list of CURIEs in batches with exponential backoff.

        Args:
            curies: List of CURIE identifiers
            batch_size: Number of CURIEs to query per request (default: 10000)
            max_retries: Maximum number of retry attempts per batch (default: 5)

        Returns:
            Dict mapping CURIE to normalized node information
        """
        results = {}

        # Process in batches
        for i in range(0, len(curies), batch_size):
            batch = curies[i:i + batch_size]
            batch_num = i//batch_size + 1

            # Retry with exponential backoff
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.normalize_url,
                        json={"curies": batch, "include_taxa": True},
                        timeout=60
                    )
                    response.raise_for_status()
                    batch_results = response.json()
                    results.update(batch_results)

                    print(f"Processed batch {batch_num} ({len(batch)} CURIEs)")

                    # Be nice to the API - small delay between batches
                    if i + batch_size < len(curies):
                        time.sleep(0.1)

                    break  # Success, move to next batch

                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                        print(f"Warning: Failed to fetch batch {batch_num} (attempt {attempt + 1}/{max_retries}): {e}")
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Error: Failed to fetch batch {batch_num} after {max_retries} attempts: {e}")
                        # Continue with other batches even if one fails completely
                        continue

        return results

    def extract_taxa_from_normalized_nodes(self, normalized_data: Dict[str, Dict]) -> Dict[str, Optional[str]]:
        """Extract taxa information from normalized node data.

        The normalized node data includes a top-level 'taxa' field with NCBITaxon identifiers.

        Args:
            normalized_data: Dictionary mapping CURIEs to their normalized node information

        Returns:
            Dict mapping CURIE to taxon ID (e.g., 'NCBITaxon:9606') or None
        """
        taxa_map = {}

        for curie, data in normalized_data.items():
            taxon = None

            if isinstance(data, dict) and data is not None:
                # Check for taxa field in normalized node data
                taxa_list = data.get('taxa', [])
                if taxa_list and len(taxa_list) > 0:
                    # Take the first taxon if multiple are present
                    taxon = taxa_list[0]

            taxa_map[curie] = taxon

        return taxa_map


def collect_genes_and_proteins(nodes_file: str) -> Set[str]:
    """Collect all Gene and Protein node IDs from nodes file.

    Args:
        nodes_file: Path to nodes.jsonl file

    Returns:
        Set of node IDs that are Genes or Proteins
    """
    import jsonlines

    gene_protein_ids = set()

    print(f"Scanning nodes file for Genes and Proteins: {nodes_file}")

    with jsonlines.open(nodes_file) as reader:
        for node in reader:
            categories = set(node.get("category", []))

            # Check if node is a Gene or Protein
            if "biolink:Gene" in categories or "biolink:Protein" in categories:
                gene_protein_ids.add(node["id"])

    print(f"Found {len(gene_protein_ids)} Gene/Protein nodes")
    return gene_protein_ids


def identify_nonhuman_genes_proteins(nodes_file: str,
                                     batch_size: int = 10000,
                                     human_taxon: str = "NCBITaxon:9606") -> Set[str]:
    """Identify non-human genes and proteins using Node Normalization API.

    Args:
        nodes_file: Path to nodes.jsonl file
        batch_size: Batch size for Node Normalization API calls
        human_taxon: CURIE for human taxon (default: NCBITaxon:9606)

    Returns:
        Set of non-human gene/protein node IDs to be filtered out
    """
    # Step 1: Collect all gene/protein IDs
    gene_protein_ids = collect_genes_and_proteins(nodes_file)

    if not gene_protein_ids:
        print("No genes or proteins found in nodes file")
        return set()

    # Step 2: Query Node Normalization API in batches
    print(f"\nQuerying Node Normalization API for taxa information (batch size: {batch_size})...")
    client = NodeNormalizationClient()
    normalized_data = client.get_normalized_nodes_batch(list(gene_protein_ids), batch_size=batch_size)

    # Step 3: Extract taxa information
    print("\nExtracting taxa from normalized node data...")
    taxa_map = client.extract_taxa_from_normalized_nodes(normalized_data)

    # Step 4: Identify non-human entities
    nonhuman_ids = set()
    human_count = 0
    no_taxon_count = 0

    for gene_protein_id in gene_protein_ids:
        taxon = taxa_map.get(gene_protein_id)

        if taxon is None:
            # No taxon = ortholog group spanning multiple species - filter it out
            nonhuman_ids.add(gene_protein_id)
            no_taxon_count += 1
        elif taxon == human_taxon:
            # Human gene/protein - keep it
            human_count += 1
        else:
            # Non-human taxon - filter it out
            nonhuman_ids.add(gene_protein_id)

    print(f"\nTaxa analysis results:")
    print(f"  Human genes/proteins: {human_count}")
    print(f"  Non-human genes/proteins (to be filtered): {len(nonhuman_ids)}")
    print(f"  No taxon (ortholog groups, filtered out): {no_taxon_count}")

    return nonhuman_ids
