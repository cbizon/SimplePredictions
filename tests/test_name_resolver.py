#!/usr/bin/env python3
"""Tests for name_resolver module (Node Normalization API)."""
import os
import tempfile
import pytest
import json
from unittest.mock import patch, Mock

from src.graph_modification.name_resolver import (
    NodeNormalizationClient,
    collect_genes_and_proteins,
    identify_nonhuman_genes_proteins
)


@pytest.fixture
def sample_normalized_response():
    """Sample response from Node Normalization API."""
    return {
        "HGNC:5": {
            "id": {"identifier": "NCBIGene:1", "label": "A1BG"},
            "taxa": ["NCBITaxon:9606"]
        },
        "HGNC:11998": {
            "id": {"identifier": "NCBIGene:7157", "label": "TP53"},
            "taxa": ["NCBITaxon:9606"]
        },
        "MGI:95523": {
            "id": {"identifier": "NCBIGene:14183", "label": "Fgfr2"},
            "taxa": ["NCBITaxon:10090"]
        }
    }


@pytest.fixture
def sample_nodes_with_genes():
    """Sample nodes including genes from different species."""
    return [
        {
            "id": "HGNC:5",
            "name": "Human Gene",
            "category": ["biolink:Gene", "biolink:NamedThing"]
        },
        {
            "id": "MGI:95523",
            "name": "Mouse Gene",
            "category": ["biolink:Gene", "biolink:NamedThing"]
        },
        {
            "id": "UniProtKB:P12345",
            "name": "Test Protein",
            "category": ["biolink:Protein", "biolink:NamedThing"]
        },
        {
            "id": "CHEBI:123",
            "name": "Test Chemical",
            "category": ["biolink:ChemicalEntity", "biolink:NamedThing"]
        },
        {
            "id": "NCBITaxon:9606",
            "name": "Homo sapiens",
            "category": ["biolink:OrganismTaxon", "biolink:NamedThing"]
        }
    ]


@pytest.fixture
def temp_nodes_file_with_genes(sample_nodes_with_genes):
    """Create temporary nodes file with genes."""
    import jsonlines

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        with jsonlines.Writer(f) as writer:
            for node in sample_nodes_with_genes:
                writer.write(node)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


def test_node_normalization_client_initialization():
    """Test NodeNormalizationClient initialization."""
    client = NodeNormalizationClient()
    assert client.base_url == "https://nodenormalization-sri.renci.org"
    assert client.normalize_url == "https://nodenormalization-sri.renci.org/get_normalized_nodes"

    # Test custom base URL
    custom_client = NodeNormalizationClient(base_url="https://custom.example.com/")
    assert custom_client.base_url == "https://custom.example.com"
    assert custom_client.normalize_url == "https://custom.example.com/get_normalized_nodes"


def test_extract_taxa_from_normalized_nodes(sample_normalized_response):
    """Test extracting taxa from normalized node data."""
    client = NodeNormalizationClient()
    taxa_map = client.extract_taxa_from_normalized_nodes(sample_normalized_response)

    # Human genes should map to NCBITaxon:9606
    assert taxa_map["HGNC:5"] == "NCBITaxon:9606"
    assert taxa_map["HGNC:11998"] == "NCBITaxon:9606"

    # Mouse gene should map to NCBITaxon:10090
    assert taxa_map["MGI:95523"] == "NCBITaxon:10090"


def test_extract_taxa_no_taxon():
    """Test extracting taxa when no taxon is present."""
    client = NodeNormalizationClient()
    normalized_data = {
        "SOME:ID": {
            "id": {"identifier": "SOME:ID", "label": "Some Entity"}
        }
    }

    taxa_map = client.extract_taxa_from_normalized_nodes(normalized_data)
    assert taxa_map["SOME:ID"] is None


def test_extract_taxa_null_node():
    """Test extracting taxa when node data is None."""
    client = NodeNormalizationClient()
    normalized_data = {
        "SOME:ID": None
    }

    taxa_map = client.extract_taxa_from_normalized_nodes(normalized_data)
    assert taxa_map["SOME:ID"] is None


def test_collect_genes_and_proteins(temp_nodes_file_with_genes):
    """Test collecting gene and protein IDs from nodes file."""
    gene_protein_ids = collect_genes_and_proteins(temp_nodes_file_with_genes)

    # Should include genes and proteins but not other types
    assert "HGNC:5" in gene_protein_ids
    assert "MGI:95523" in gene_protein_ids
    assert "UniProtKB:P12345" in gene_protein_ids
    assert "CHEBI:123" not in gene_protein_ids
    assert "NCBITaxon:9606" not in gene_protein_ids

    assert len(gene_protein_ids) == 3


@patch('src.graph_modification.name_resolver.requests.post')
def test_get_normalized_nodes_batch(mock_post, sample_normalized_response):
    """Test batched normalized node lookup."""
    mock_response = Mock()
    mock_response.json.return_value = sample_normalized_response
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    client = NodeNormalizationClient()
    curies = ["HGNC:5", "HGNC:11998", "MGI:95523"]

    results = client.get_normalized_nodes_batch(curies, batch_size=2)

    # Should make 2 API calls (2 batches)
    assert mock_post.call_count == 2

    # Check first batch
    first_call_args = mock_post.call_args_list[0]
    assert first_call_args[1]['json']['curies'] == ["HGNC:5", "HGNC:11998"]
    assert first_call_args[1]['json']['include_taxa'] == True

    # Check second batch
    second_call_args = mock_post.call_args_list[1]
    assert second_call_args[1]['json']['curies'] == ["MGI:95523"]
    assert second_call_args[1]['json']['include_taxa'] == True


@patch('src.graph_modification.name_resolver.requests.post')
@patch('src.graph_modification.name_resolver.time.sleep')  # Mock sleep to speed up test
def test_get_normalized_nodes_batch_failure_handling(mock_sleep, mock_post):
    """Test that batch processing continues on failure with exponential backoff."""
    import requests

    # First batch succeeds, second fails all retries, third succeeds
    mock_response_1 = Mock()
    mock_response_1.json.return_value = {"HGNC:5": {"taxa": ["NCBITaxon:9606"]}}
    mock_response_1.raise_for_status.return_value = None

    mock_response_3 = Mock()
    mock_response_3.json.return_value = {"HGNC:789": {"taxa": ["NCBITaxon:9606"]}}
    mock_response_3.raise_for_status.return_value = None

    # First batch succeeds, second batch fails all 5 retry attempts, third batch succeeds
    mock_post.side_effect = [
        mock_response_1,  # Batch 1 succeeds
        requests.exceptions.RequestException("API Error"),  # Batch 2 attempt 1
        requests.exceptions.RequestException("API Error"),  # Batch 2 attempt 2
        requests.exceptions.RequestException("API Error"),  # Batch 2 attempt 3
        requests.exceptions.RequestException("API Error"),  # Batch 2 attempt 4
        requests.exceptions.RequestException("API Error"),  # Batch 2 attempt 5
        mock_response_3  # Batch 3 succeeds
    ]

    client = NodeNormalizationClient()
    curies = ["HGNC:5", "HGNC:11", "HGNC:789"]

    results = client.get_normalized_nodes_batch(curies, batch_size=1, max_retries=5)

    # Should have results from first and third batches
    assert "HGNC:5" in results
    assert "HGNC:789" in results
    assert "HGNC:11" not in results

    # Verify exponential backoff was used (1, 2, 4, 8 seconds for 4 retries)
    expected_sleep_calls = [
        1,  # First retry
        2,  # Second retry
        4,  # Third retry
        8   # Fourth retry
    ]
    actual_sleep_calls = [call[0][0] for call in mock_sleep.call_args_list if call[0][0] > 0.1]
    assert actual_sleep_calls == expected_sleep_calls


@patch('src.graph_modification.name_resolver.NodeNormalizationClient.get_normalized_nodes_batch')
@patch('src.graph_modification.name_resolver.NodeNormalizationClient.extract_taxa_from_normalized_nodes')
def test_identify_nonhuman_genes_proteins(mock_extract_taxa, mock_get_normalized, temp_nodes_file_with_genes):
    """Test identifying non-human genes and proteins."""
    # Mock the API response
    mock_get_normalized.return_value = {
        "HGNC:5": {"taxa": ["NCBITaxon:9606"]},
        "MGI:95523": {"taxa": ["NCBITaxon:10090"]},
        "UniProtKB:P12345": {}
    }

    # Mock taxa extraction
    mock_extract_taxa.return_value = {
        "HGNC:5": "NCBITaxon:9606",  # Human
        "MGI:95523": "NCBITaxon:10090",  # Mouse
        "UniProtKB:P12345": None  # No taxon (ortholog group)
    }

    nonhuman_ids = identify_nonhuman_genes_proteins(
        temp_nodes_file_with_genes,
        batch_size=10000,
        human_taxon="NCBITaxon:9606"
    )

    # Should identify mouse gene as non-human
    assert "MGI:95523" in nonhuman_ids

    # Should NOT identify human gene as non-human
    assert "HGNC:5" not in nonhuman_ids

    # Should identify no-taxon entities as non-human (ortholog groups)
    assert "UniProtKB:P12345" in nonhuman_ids

    # Should have two non-human genes (mouse + ortholog group)
    assert len(nonhuman_ids) == 2


def test_collect_genes_and_proteins_empty_file():
    """Test collecting genes from empty file."""
    import jsonlines

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name

    try:
        gene_protein_ids = collect_genes_and_proteins(temp_path)
        assert len(gene_protein_ids) == 0
    finally:
        os.unlink(temp_path)


def test_extract_taxa_with_multiple_taxa():
    """Test extracting taxa when multiple taxa are present (takes first one)."""
    client = NodeNormalizationClient()
    normalized_data = {
        "HGNC:5": {
            "taxa": ["NCBITaxon:9606", "NCBITaxon:10090"],
            "id": {"identifier": "HGNC:5"}
        }
    }

    taxa_map = client.extract_taxa_from_normalized_nodes(normalized_data)
    # Should take the first taxon
    assert taxa_map["HGNC:5"] == "NCBITaxon:9606"
