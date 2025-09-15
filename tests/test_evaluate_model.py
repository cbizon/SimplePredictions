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
    calculate_recall_at_k, calculate_hits_at_k_per_disease, plot_ranking_metrics,
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
                'total_recall_at_k': {1: 0.05, 5: 0.2, 10: 0.3},
                'hits_at_k': {1: 1.0, 5: 1.0, 10: 1.0},
                'total_recall_max': 0.5,
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

def test_calculate_hits_at_k_per_disease():
    """Test per-disease hits@K calculation."""
    # Test data: disease1 has hits, disease2 has no hits at K=1
    test_pairs = [
        ('drug1', 'disease1'),  # True positive (high score)
        ('drug2', 'disease1'),  # False positive
        ('drug3', 'disease2'),  # False positive (high score)
        ('drug4', 'disease2'),  # True positive (low score)
    ]
    y_true = np.array([1, 0, 0, 1])
    y_scores = np.array([0.9, 0.3, 0.8, 0.2])  # disease2 true positive ranked last
    k_values = [1, 2]
    
    hits_at_k = calculate_hits_at_k_per_disease(test_pairs, y_true, y_scores, k_values)
    
    # H@1: disease1 gets hit (drug1), disease2 gets no hit (drug3)
    # Fraction with hits = 1/2 = 0.5
    # H@2: disease1 gets hit, disease2 gets hit (drug4 in top 2)  
    # Fraction with hits = 2/2 = 1.0
    assert abs(hits_at_k[1] - 0.5) < 1e-10
    assert abs(hits_at_k[2] - 1.0) < 1e-10


def test_missing_embeddings_score_zero():
    """Test that missing embeddings result in zero/low prediction scores."""
    from src.modeling.train_model import create_feature_vectors
    from sklearn.ensemble import RandomForestClassifier
    
    # Create test embeddings with distinctive patterns
    embeddings = {
        "DRUG:1": np.array([1.0, 0.0, 1.0, 0.0]),  # Distinctive pattern
        "DRUG:3": np.array([0.0, 1.0, 0.0, 1.0]),  # Different pattern
        "DISEASE:1": np.array([1.0, 1.0, 0.0, 0.0]),  # Positive pattern
        "DISEASE:3": np.array([0.0, 0.0, 1.0, 1.0])   # Negative pattern
        # Missing: "DRUG:2", "DISEASE:2"
    }
    
    # Train with more diverse data
    positive_pairs = [("DRUG:1", "DISEASE:1"), ("DRUG:3", "DISEASE:1")]
    negative_pairs = [("DRUG:1", "DISEASE:3"), ("DRUG:3", "DISEASE:3")]
    
    pos_features, pos_labels = create_feature_vectors(embeddings, positive_pairs)
    neg_features, neg_labels = create_feature_vectors(embeddings, negative_pairs)
    
    # Add some zero-padded negative examples to train model to recognize zeros as negative
    zero_negative_pairs = [("DRUG:1", "DISEASE:2"), ("DRUG:2", "DISEASE:1")]  # Missing embeddings
    zero_neg_features, zero_neg_labels = create_feature_vectors(embeddings, zero_negative_pairs, pad_missing=True)
    
    # Combine training data
    X_train = np.vstack([pos_features, neg_features, zero_neg_features])
    y_train = np.array([1] * len(pos_features) + [0] * len(neg_features) + [0] * len(zero_neg_features))
    
    # Train model with more trees for stability
    rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    
    # Test predictions on pairs with missing embeddings
    test_pairs = [
        ("DRUG:1", "DISEASE:1"),     # Both present, should be positive
        ("DRUG:2", "DISEASE:1"),     # Missing drug embedding
        ("DRUG:1", "DISEASE:2"),     # Missing disease embedding  
        ("DRUG:2", "DISEASE:2")      # Both embeddings missing
    ]
    
    test_features, test_labels = create_feature_vectors(embeddings, test_pairs, pad_missing=True)
    predictions = rf.predict_proba(test_features)[:, 1]
    
    real_emb_score = predictions[0]  # Both embeddings present
    missing_drug_score = predictions[1]  # Drug embedding missing
    missing_disease_score = predictions[2]  # Disease embedding missing
    both_missing_score = predictions[3]  # Both embeddings missing
    
    print(f"Scores - Real: {real_emb_score:.3f}, Missing drug: {missing_drug_score:.3f}, Missing disease: {missing_disease_score:.3f}, Both missing: {both_missing_score:.3f}")
    
    # The key test: missing embeddings should result in lower confidence
    # Since we trained the model to associate zeros with negative class
    assert real_emb_score >= 0.7, f"Known positive should score high, got {real_emb_score:.3f}"
    
    # Missing embeddings should score lower (model should be less confident about zero vectors)
    assert missing_drug_score <= 0.6, f"Missing drug should score lower, got {missing_drug_score:.3f}"
    assert missing_disease_score <= 0.6, f"Missing disease should score lower, got {missing_disease_score:.3f}"
    assert both_missing_score <= 0.6, f"Both missing should score lower, got {both_missing_score:.3f}"