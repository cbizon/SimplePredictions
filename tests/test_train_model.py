#!/usr/bin/env python3
"""Tests for train_model module."""
import os
import tempfile
import pytest
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.modeling.train_model import (
    get_next_model_version, get_embedding_info, load_embeddings,
    load_ground_truth, create_feature_vectors, generate_negative_samples,
    extract_node_ids_from_positives
)


@pytest.fixture
def temp_models_dir():
    """Create temporary models directory with existing versions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        models_dir = os.path.join(temp_dir, "models")
        os.makedirs(models_dir)
        
        # Create some existing model versions
        os.makedirs(os.path.join(models_dir, "model_0"))
        os.makedirs(os.path.join(models_dir, "model_2"))
        os.makedirs(os.path.join(models_dir, "model_5"))
        
        yield models_dir


@pytest.fixture
def sample_embeddings_file():
    """Create a sample embeddings file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.emb', delete=False) as f:
        # Write embeddings file in node2vec format
        f.write("3 4\n")  # 3 nodes, 4 dimensions
        f.write("CHEBI:123 0.1 0.2 0.3 0.4\n")
        f.write("MONDO:456 0.5 0.6 0.7 0.8\n")
        f.write("HGNC:789 0.9 1.0 1.1 1.2\n")
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_ground_truth_file():
    """Create a sample ground truth CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Write CSV with the expected column names
        f.write("final normalized drug id,final normalized disease id,other_col\n")
        f.write("CHEBI:123,MONDO:456,value1\n")
        f.write("CHEBI:789,MONDO:123,value2\n")
        f.write("CHEBI:456,MONDO:789,value3\n")
        f.write(",MONDO:999,value4\n")  # Row with missing drug ID
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_embeddings():
    """Sample embeddings dictionary."""
    return {
        "CHEBI:123": np.array([0.1, 0.2, 0.3, 0.4]),
        "MONDO:456": np.array([0.5, 0.6, 0.7, 0.8]),
        "HGNC:789": np.array([0.9, 1.0, 1.1, 1.2]),
        "CHEBI:456": np.array([1.3, 1.4, 1.5, 1.6]),
        "MONDO:123": np.array([1.7, 1.8, 1.9, 2.0])
    }


def test_get_next_model_version_existing_models(temp_models_dir):
    """Test version numbering with existing models."""
    version = get_next_model_version(temp_models_dir)
    assert version == 6  # Should be max(0, 2, 5) + 1


def test_get_next_model_version_empty_dir():
    """Test version numbering with empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        models_dir = os.path.join(temp_dir, "models")
        os.makedirs(models_dir)
        
        version = get_next_model_version(models_dir)
        assert version == 0


def test_get_next_model_version_nonexistent_dir():
    """Test version numbering with non-existent directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        nonexistent_dir = os.path.join(temp_dir, "nonexistent")
        
        version = get_next_model_version(nonexistent_dir)
        assert version == 0


def test_get_embedding_info(sample_embeddings_file):
    """Test extracting embedding metadata."""
    info = get_embedding_info(sample_embeddings_file)
    
    assert info["embeddings_file"] == sample_embeddings_file
    assert info["num_nodes"] == 3
    assert info["embedding_dim"] == 4
    assert "error" not in info


def test_get_embedding_info_nonexistent_file():
    """Test embedding info with non-existent file."""
    info = get_embedding_info("/nonexistent/path")
    
    assert "error" in info
    assert "not found" in info["error"]


def test_get_embedding_info_with_provenance(sample_embeddings_file):
    """Test embedding info with provenance file."""
    # Create provenance file in same directory
    embedding_dir = os.path.dirname(sample_embeddings_file)
    provenance_file = os.path.join(embedding_dir, "provenance.json")
    
    provenance_data = {
        "timestamp": "2023-01-01T00:00:00",
        "algorithm": "node2vec",
        "parameters": {"dimensions": 4}
    }
    
    with open(provenance_file, 'w') as f:
        json.dump(provenance_data, f)
    
    try:
        info = get_embedding_info(sample_embeddings_file)
        
        assert "embedding_provenance" in info
        assert info["embedding_provenance"]["algorithm"] == "node2vec"
        assert info["embedding_provenance"]["parameters"]["dimensions"] == 4
    finally:
        os.unlink(provenance_file)


def test_load_embeddings(sample_embeddings_file):
    """Test loading embeddings from file."""
    embeddings = load_embeddings(sample_embeddings_file)
    
    assert len(embeddings) == 3
    assert "CHEBI:123" in embeddings
    assert "MONDO:456" in embeddings
    assert "HGNC:789" in embeddings
    
    # Test specific values
    np.testing.assert_array_equal(embeddings["CHEBI:123"], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_equal(embeddings["MONDO:456"], [0.5, 0.6, 0.7, 0.8])
    np.testing.assert_array_equal(embeddings["HGNC:789"], [0.9, 1.0, 1.1, 1.2])


def test_load_ground_truth(sample_ground_truth_file):
    """Test loading ground truth without embedding filtering."""
    positive_pairs, stats = load_ground_truth(sample_ground_truth_file)
    
    # Should have 3 valid pairs (excluding the row with missing drug ID)
    assert len(positive_pairs) == 3
    assert ("CHEBI:123", "MONDO:456") in positive_pairs
    assert ("CHEBI:789", "MONDO:123") in positive_pairs
    assert ("CHEBI:456", "MONDO:789") in positive_pairs
    
    # Check stats
    assert stats["total_rows"] == 4
    assert stats["clean_rows"] == 3
    assert stats["final_positive_pairs"] == 3
    assert stats["unique_drugs"] == 3
    assert stats["unique_diseases"] == 3


def test_load_ground_truth_with_embedding_filter(sample_ground_truth_file, sample_embeddings):
    """Test loading ground truth with embedding filtering."""
    positive_pairs, stats = load_ground_truth(sample_ground_truth_file, sample_embeddings)
    
    # Should only include pairs where both drug and disease have embeddings
    # Check that filtering actually occurred and got some pairs
    assert len(positive_pairs) >= 1
    assert ("CHEBI:123", "MONDO:456") in positive_pairs
    
    assert stats["final_positive_pairs"] == len(positive_pairs)


def test_load_ground_truth_missing_columns():
    """Test ground truth loading with missing required columns."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("wrong_drug_col,wrong_disease_col\n")
        f.write("CHEBI:123,MONDO:456\n")
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="Expected columns"):
            load_ground_truth(temp_path)
    finally:
        os.unlink(temp_path)


def test_create_feature_vectors(sample_embeddings):
    """Test creating feature vectors from embeddings."""
    pairs = [("CHEBI:123", "MONDO:456"), ("CHEBI:456", "MONDO:123")]
    
    features, pair_labels = create_feature_vectors(sample_embeddings, pairs)
    
    assert features.shape == (2, 8)  # 2 pairs, 4+4 dimensions
    assert len(pair_labels) == 2
    assert pair_labels[0] == ("CHEBI:123", "MONDO:456")
    assert pair_labels[1] == ("CHEBI:456", "MONDO:123")
    
    # Test concatenation is correct
    expected_first = np.concatenate([sample_embeddings["CHEBI:123"], sample_embeddings["MONDO:456"]])
    np.testing.assert_array_equal(features[0], expected_first)


def test_create_feature_vectors_missing_embeddings(sample_embeddings):
    """Test feature vector creation with missing embeddings."""
    pairs = [("CHEBI:123", "MONDO:456"), ("NONEXISTENT:123", "MONDO:456")]
    
    features, pair_labels = create_feature_vectors(sample_embeddings, pairs)
    
    # Should only include the first pair
    assert features.shape == (1, 8)
    assert len(pair_labels) == 1
    assert pair_labels[0] == ("CHEBI:123", "MONDO:456")


def test_generate_negative_samples():
    """Test negative sample generation."""
    positive_pairs = {("CHEBI:123", "MONDO:456"), ("CHEBI:789", "MONDO:123")}
    all_drugs = ["CHEBI:123", "CHEBI:456", "CHEBI:789"]
    all_diseases = ["MONDO:123", "MONDO:456", "MONDO:789"]
    
    negative_pairs = generate_negative_samples(positive_pairs, all_drugs, all_diseases, ratio=2)
    
    assert len(negative_pairs) == 4  # 2 positive * ratio 2
    
    # Check that none of the negative pairs are in positive pairs
    assert not negative_pairs.intersection(positive_pairs)
    
    # Check that all pairs use valid drug and disease IDs
    for drug_id, disease_id in negative_pairs:
        assert drug_id in all_drugs
        assert disease_id in all_diseases


def test_generate_negative_samples_reproducible():
    """Test that negative sample generation is reproducible."""
    positive_pairs = {("CHEBI:123", "MONDO:456")}
    all_drugs = ["CHEBI:123", "CHEBI:456", "CHEBI:789"]
    all_diseases = ["MONDO:123", "MONDO:456", "MONDO:789"]
    
    # Generate twice with same seed
    neg1 = generate_negative_samples(positive_pairs, all_drugs, all_diseases, ratio=3)
    neg2 = generate_negative_samples(positive_pairs, all_drugs, all_diseases, ratio=3)
    
    assert neg1 == neg2  # Should be identical due to fixed seed


def test_extract_node_ids_from_positives():
    """Test extracting unique node IDs from positive examples."""
    positive_pairs = {
        ("CHEBI:123", "MONDO:456"),
        ("CHEBI:123", "MONDO:789"),  # Duplicate drug
        ("CHEBI:456", "MONDO:456")   # Duplicate disease
    }
    
    drug_ids, disease_ids = extract_node_ids_from_positives(positive_pairs)
    
    assert set(drug_ids) == {"CHEBI:123", "CHEBI:456"}
    assert set(disease_ids) == {"MONDO:456", "MONDO:789"}
    assert len(drug_ids) == 2
    assert len(disease_ids) == 2


def test_extract_node_ids_empty_pairs():
    """Test extracting node IDs from empty set."""
    drug_ids, disease_ids = extract_node_ids_from_positives(set())
    
    assert drug_ids == []
    assert disease_ids == []