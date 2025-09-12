#!/usr/bin/env python3
"""Tests for evaluate_model module."""
import os
import tempfile
import pytest
import json
import numpy as np
import pickle
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.modeling.evaluate_model import (
    get_next_evaluation_version, get_model_info, calculate_precision_at_k,
    calculate_recall_at_k, calculate_hits_at_k, plot_ranking_metrics,
    create_precision_recall_curve
)


@pytest.fixture
def temp_evaluations_dir():
    """Create temporary evaluations directory with existing versions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        evaluations_dir = os.path.join(temp_dir, "evaluations")
        os.makedirs(evaluations_dir)
        
        # Create some existing evaluation versions
        os.makedirs(os.path.join(evaluations_dir, "evaluation_0"))
        os.makedirs(os.path.join(evaluations_dir, "evaluation_3"))
        os.makedirs(os.path.join(evaluations_dir, "evaluation_7"))
        
        yield evaluations_dir


@pytest.fixture
def sample_model_file():
    """Create a sample pickled model file."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        # Create a dummy sklearn model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create dummy training data
        X_dummy = np.random.randn(10, 4)
        y_dummy = np.random.randint(0, 2, 10)
        model.fit(X_dummy, y_dummy)
        
        pickle.dump(model, f)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_model_provenance():
    """Create a sample model provenance file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        provenance_data = {
            "timestamp": "2023-01-01T00:00:00",
            "algorithm": "random_forest",
            "version": "model_0",
            "model_parameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "data_splits": {
                "total_samples": 1000,
                "positive_samples": 500,
                "negative_samples": 500
            }
        }
        json.dump(provenance_data, f)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


def test_get_next_evaluation_version_existing_evaluations(temp_evaluations_dir):
    """Test version numbering with existing evaluations."""
    version = get_next_evaluation_version(temp_evaluations_dir)
    assert version == 8  # Should be max(0, 3, 7) + 1


def test_get_next_evaluation_version_empty_dir():
    """Test version numbering with empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        evaluations_dir = os.path.join(temp_dir, "evaluations")
        os.makedirs(evaluations_dir)
        
        version = get_next_evaluation_version(evaluations_dir)
        assert version == 0


def test_get_next_evaluation_version_nonexistent_dir():
    """Test version numbering with non-existent directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        nonexistent_dir = os.path.join(temp_dir, "nonexistent")
        
        version = get_next_evaluation_version(nonexistent_dir)
        assert version == 0


def test_get_model_info(sample_model_file, sample_model_provenance):
    """Test extracting model metadata."""
    # Place provenance file next to model file
    model_dir = os.path.dirname(sample_model_file)
    provenance_file = os.path.join(model_dir, "provenance.json")
    
    # Copy provenance data to the correct location
    with open(sample_model_provenance, 'r') as src:
        with open(provenance_file, 'w') as dst:
            dst.write(src.read())
    
    try:
        info = get_model_info(sample_model_file)
        
        assert info["model_file"] == sample_model_file
        assert "model_size_bytes" in info
        assert "model_provenance" in info
        assert info["model_provenance"]["algorithm"] == "random_forest"
        assert info["model_provenance"]["model_parameters"]["n_estimators"] == 100
    finally:
        if os.path.exists(provenance_file):
            os.unlink(provenance_file)


def test_get_model_info_no_provenance(sample_model_file):
    """Test model info without provenance file."""
    info = get_model_info(sample_model_file)
    
    assert info["model_file"] == sample_model_file
    assert "model_size_bytes" in info
    assert "model_provenance" not in info


def test_get_model_info_nonexistent_file():
    """Test model info with non-existent file."""
    info = get_model_info("/nonexistent/path.pkl")
    
    assert "error" in info
    assert "not found" in info["error"]


def test_calculate_precision_at_k():
    """Test precision@K calculation."""
    # Create test data: 5 samples, first 3 are positive
    y_true = np.array([1, 1, 1, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])  # Perfectly ranked
    k_values = [1, 2, 3, 5]
    
    precision_at_k = calculate_precision_at_k(y_true, y_scores, k_values)
    
    # Precision@1: 1/1 = 1.0 (top 1 prediction is correct)
    # Precision@2: 2/2 = 1.0 (top 2 predictions are correct)
    # Precision@3: 3/3 = 1.0 (top 3 predictions are correct)
    # Precision@5: 3/5 = 0.6 (3 correct out of 5 total)
    expected = {1: 1.0, 2: 1.0, 3: 1.0, 5: 0.6}
    
    assert precision_at_k == expected


def test_calculate_precision_at_k_imperfect_ranking():
    """Test precision@K with imperfect ranking."""
    y_true = np.array([1, 0, 1, 0, 1])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])  # Mixed ranking
    k_values = [1, 2, 3]
    
    precision_at_k = calculate_precision_at_k(y_true, y_scores, k_values)
    
    # Precision@1: 1/1 = 1.0 (top prediction is positive)
    # Precision@2: 1/2 = 0.5 (1 positive out of top 2)
    # Precision@3: 2/3 = 0.667 (2 positive out of top 3)
    expected = {1: 1.0, 2: 0.5, 3: 2/3}
    
    for k in k_values:
        assert abs(precision_at_k[k] - expected[k]) < 1e-10


def test_calculate_recall_at_k():
    """Test recall@K calculation."""
    # 3 total positives in dataset
    y_true = np.array([1, 1, 1, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])  # Perfectly ranked
    k_values = [1, 2, 3, 5]
    
    recall_at_k = calculate_recall_at_k(y_true, y_scores, k_values)
    
    # Recall@1: 1/3 = 0.333 (1 found out of 3 total positives)
    # Recall@2: 2/3 = 0.667 (2 found out of 3 total positives)
    # Recall@3: 3/3 = 1.0 (all 3 positives found)
    # Recall@5: 3/3 = 1.0 (all positives found, can't exceed 1.0)
    expected = {1: 1/3, 2: 2/3, 3: 1.0, 5: 1.0}
    
    for k in k_values:
        assert abs(recall_at_k[k] - expected[k]) < 1e-10


def test_calculate_recall_at_k_no_positives():
    """Test recall@K when there are no positives."""
    y_true = np.array([0, 0, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    k_values = [1, 2, 3]
    
    recall_at_k = calculate_recall_at_k(y_true, y_scores, k_values)
    
    # All recalls should be 0.0 when there are no positives
    for k in k_values:
        assert recall_at_k[k] == 0.0


def test_calculate_hits_at_k():
    """Test hits@K calculation."""
    y_true = np.array([1, 0, 1, 0, 1])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    k_values = [1, 2, 3, 5]
    
    hits_at_k = calculate_hits_at_k(y_true, y_scores, k_values)
    
    # Hits@K is binary: 1 if at least one positive in top K, 0 otherwise
    # Hits@1: 1 (top prediction is positive)
    # Hits@2: 1 (at least one positive in top 2)
    # Hits@3: 1 (at least one positive in top 3)
    # Hits@5: 1 (at least one positive in top 5)
    expected = {1: 1, 2: 1, 3: 1, 5: 1}
    
    assert hits_at_k == expected


def test_calculate_hits_at_k_no_hits():
    """Test hits@K when there are no hits."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])  # All positives ranked low
    k_values = [1, 2]
    
    hits_at_k = calculate_hits_at_k(y_true, y_scores, k_values)
    
    # No hits in top 1 or top 2
    assert hits_at_k[1] == 0
    assert hits_at_k[2] == 0


def test_ranking_metrics_edge_cases():
    """Test ranking metrics with edge cases."""
    # Single sample
    y_true = np.array([1])
    y_scores = np.array([0.5])
    k_values = [1, 2]  # k=2 exceeds dataset size
    
    precision_at_k = calculate_precision_at_k(y_true, y_scores, k_values)
    recall_at_k = calculate_recall_at_k(y_true, y_scores, k_values)
    hits_at_k = calculate_hits_at_k(y_true, y_scores, k_values)
    
    # For k=1: perfect scores
    assert precision_at_k[1] == 1.0
    assert recall_at_k[1] == 1.0
    assert hits_at_k[1] == 1
    
    # For k=2: should return NaN when k > array length
    assert np.isnan(precision_at_k[2])
    assert np.isnan(recall_at_k[2])
    assert np.isnan(hits_at_k[2])


def test_ranking_metrics_empty_arrays():
    """Test ranking metrics with empty arrays."""
    y_true = np.array([])
    y_scores = np.array([])
    k_values = [1]
    
    precision_at_k = calculate_precision_at_k(y_true, y_scores, k_values)
    recall_at_k = calculate_recall_at_k(y_true, y_scores, k_values)
    hits_at_k = calculate_hits_at_k(y_true, y_scores, k_values)
    
    # Should return NaN for empty arrays
    assert np.isnan(precision_at_k[1])
    assert np.isnan(recall_at_k[1])
    assert np.isnan(hits_at_k[1])


@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
def test_plot_ranking_metrics(mock_close, mock_savefig):
    """Test plotting ranking metrics with new multi-model format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data in new multi-model format
        metrics_data = {
            'Test_Model': {
                'precision_at_k': {1: 1.0, 5: 0.8, 10: 0.6},
                'recall_at_k': {1: 0.1, 5: 0.4, 10: 0.6},
                'hits_at_k': {1: 1.0, 5: 1.0, 10: 1.0},
                'k_values': [1, 5, 10]
            }
        }
        
        # Should not raise an exception
        plot_ranking_metrics(metrics_data, temp_dir)
        
        # Check that matplotlib functions were called
        assert mock_savefig.called
        assert mock_close.called


@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
def test_create_precision_recall_curve(mock_close, mock_savefig):
    """Test creating precision-recall curve with new multi-model format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data in new multi-model format
        pr_curve_data = {
            'Test_Model': {
                'y_true': np.array([1, 0, 1, 0, 1]),
                'y_scores': np.array([0.9, 0.8, 0.7, 0.6, 0.5])
            }
        }
        
        # Should not raise an exception
        create_precision_recall_curve(pr_curve_data, temp_dir)
        
        # Check that matplotlib functions were called
        assert mock_savefig.called
        assert mock_close.called


def test_ranking_metrics_consistency():
    """Test that ranking metrics are internally consistent."""
    # Create a more complex test case
    np.random.seed(42)
    n_samples = 100
    n_positives = 30
    
    # Create labels with known number of positives
    y_true = np.zeros(n_samples)
    y_true[:n_positives] = 1
    np.random.shuffle(y_true)
    
    # Create scores that are somewhat correlated with labels
    y_scores = y_true * 0.6 + np.random.randn(n_samples) * 0.3
    
    k_values = [1, 5, 10, 20, 50]
    
    precision_at_k = calculate_precision_at_k(y_true, y_scores, k_values)
    recall_at_k = calculate_recall_at_k(y_true, y_scores, k_values)
    hits_at_k = calculate_hits_at_k(y_true, y_scores, k_values)
    
    # Basic consistency checks
    for k in k_values:
        # Precision should be between 0 and 1
        assert 0.0 <= precision_at_k[k] <= 1.0
        
        # Recall should be between 0 and 1
        assert 0.0 <= recall_at_k[k] <= 1.0
        
        # Hits@K should be binary (0 or 1)
        assert hits_at_k[k] in [0, 1]
        
        # Calculate actual number of hits in top K
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_true = y_true[sorted_indices]
        top_k_true = sorted_true[:k]
        actual_hit_count = np.sum(top_k_true)
        
        # Relationship: precision = actual_hit_count / k
        expected_precision = actual_hit_count / k
        assert abs(precision_at_k[k] - expected_precision) < 1e-10
        
        # Relationship: recall = actual_hit_count / n_positives
        expected_recall = actual_hit_count / n_positives
        assert abs(recall_at_k[k] - expected_recall) < 1e-10
        
        # Hits@K should be 1 if any hits, 0 otherwise
        expected_hits_at_k = 1 if actual_hit_count > 0 else 0
        assert hits_at_k[k] == expected_hits_at_k
    
    # Monotonicity checks (recall should generally increase with k)
    for i in range(len(k_values) - 1):
        k1, k2 = k_values[i], k_values[i + 1]
        assert recall_at_k[k1] <= recall_at_k[k2] + 1e-10  # Allow small numerical errors
        # Hits@K doesn't have monotonicity since it's binary