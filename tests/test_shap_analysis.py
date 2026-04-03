#!/usr/bin/env python3
"""Tests for SHAP analysis module."""

import os
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tempfile

from src.modeling.shap_analysis import compute_shap_for_top_k, save_shap_analysis


def test_compute_shap_for_top_k():
    """Test SHAP computation for top-K predictions."""
    # Create synthetic data
    np.random.seed(42)

    # Simulate 100 drug-disease pairs with 10-dimensional embeddings each
    n_samples = 100
    embedding_dim = 10

    # Create feature vectors (concatenated drug + disease embeddings)
    feature_vectors = np.random.randn(n_samples, embedding_dim * 2)

    # Create synthetic labels
    y_train = np.random.randint(0, 2, n_samples)

    # Train a simple Random Forest
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    model.fit(feature_vectors, y_train)

    # Get predictions
    predictions = model.predict_proba(feature_vectors)[:, 1]

    # Create pairs
    pairs = [(f"DRUG:{i}", f"DISEASE:{i}") for i in range(n_samples)]

    # Create embeddings dictionary (for reference)
    embeddings = {}
    for i in range(n_samples):
        embeddings[f"DRUG:{i}"] = feature_vectors[i, :embedding_dim]
        embeddings[f"DISEASE:{i}"] = feature_vectors[i, embedding_dim:]

    # Compute SHAP for top 5 predictions (no y_true filtering)
    top_k = 5
    shap_results = compute_shap_for_top_k(
        model=model,
        feature_vectors=feature_vectors,
        predictions=predictions,
        pairs=pairs,
        embeddings=embeddings,
        top_k=top_k,
        y_true=None
    )

    # Verify results structure
    assert "top_k_predictions" in shap_results
    assert "embedding_dim" in shap_results
    assert "analysis_metadata" in shap_results

    # Check metadata
    assert shap_results["embedding_dim"] == embedding_dim
    assert shap_results["analysis_metadata"]["top_k"] == top_k
    assert shap_results["analysis_metadata"]["total_predictions_analyzed"] == top_k
    assert shap_results["analysis_metadata"]["embedding_dimension"] == embedding_dim
    assert shap_results["analysis_metadata"]["total_features"] == embedding_dim * 2

    # Check predictions
    predictions_list = shap_results["top_k_predictions"]
    assert len(predictions_list) == top_k

    # Verify each prediction has required fields
    for i, pred in enumerate(predictions_list):
        assert pred["rank"] == i + 1
        assert "drug_id" in pred
        assert "disease_id" in pred
        assert "prediction_score" in pred
        assert "shap_analysis" in pred

        shap_analysis = pred["shap_analysis"]
        assert "drug" in shap_analysis
        assert "disease" in shap_analysis
        assert "summary" in shap_analysis

        # Check drug analysis
        drug_analysis = shap_analysis["drug"]
        assert "all_features" in drug_analysis
        assert "top_features" in drug_analysis
        assert "num_nonzero_features" in drug_analysis
        assert "shap_sum" in drug_analysis
        assert "shap_abs_sum" in drug_analysis
        # all_features is now a list of dicts, not a dict
        assert isinstance(drug_analysis["all_features"], list)
        assert len(drug_analysis["all_features"]) <= embedding_dim  # Filtered for non-zero
        assert len(drug_analysis["all_features"]) == drug_analysis["num_nonzero_features"]
        assert len(drug_analysis["top_features"]) <= 10
        # Check that all_features is sorted by absolute value
        if len(drug_analysis["all_features"]) > 1:
            abs_vals = [abs(f["shap_value"]) for f in drug_analysis["all_features"]]
            assert abs_vals == sorted(abs_vals, reverse=True)

        # Check disease analysis
        disease_analysis = shap_analysis["disease"]
        assert "all_features" in disease_analysis
        assert "top_features" in disease_analysis
        assert "num_nonzero_features" in disease_analysis
        assert "shap_sum" in disease_analysis
        assert "shap_abs_sum" in disease_analysis
        # all_features is now a list of dicts, not a dict
        assert isinstance(disease_analysis["all_features"], list)
        assert len(disease_analysis["all_features"]) <= embedding_dim  # Filtered for non-zero
        assert len(disease_analysis["all_features"]) == disease_analysis["num_nonzero_features"]
        assert len(disease_analysis["top_features"]) <= 10
        # Check that all_features is sorted by absolute value
        if len(disease_analysis["all_features"]) > 1:
            abs_vals = [abs(f["shap_value"]) for f in disease_analysis["all_features"]]
            assert abs_vals == sorted(abs_vals, reverse=True)

        # Check summary
        summary = shap_analysis["summary"]
        assert "total_shap_sum" in summary
        assert "drug_contribution" in summary
        assert "disease_contribution" in summary
        assert "drug_abs_contribution" in summary
        assert "disease_abs_contribution" in summary

        # Verify sums are consistent
        expected_sum = drug_analysis["shap_sum"] + disease_analysis["shap_sum"]
        assert abs(summary["total_shap_sum"] - expected_sum) < 1e-6

    # Verify predictions are sorted by score (descending)
    scores = [pred["prediction_score"] for pred in predictions_list]
    assert scores == sorted(scores, reverse=True)

    print("All SHAP analysis tests passed!")


def test_save_shap_analysis():
    """Test saving SHAP analysis results to file."""
    # Create minimal SHAP results
    shap_results = {
        "top_k_predictions": [
            {
                "rank": 1,
                "drug_id": "DRUG:1",
                "disease_id": "DISEASE:1",
                "prediction_score": 0.95,
                "shap_analysis": {
                    "drug": {
                        "all_features": [
                            {"feature": "drug_emb_1", "shap_value": 0.2},
                            {"feature": "drug_emb_0", "shap_value": 0.1}
                        ],
                        "top_features": [{"feature": "drug_emb_1", "shap_value": 0.2}],
                        "num_nonzero_features": 2,
                        "shap_sum": 0.3,
                        "shap_abs_sum": 0.3
                    },
                    "disease": {
                        "all_features": [
                            {"feature": "disease_emb_0", "shap_value": -0.1},
                            {"feature": "disease_emb_1", "shap_value": 0.05}
                        ],
                        "top_features": [{"feature": "disease_emb_0", "shap_value": -0.1}],
                        "num_nonzero_features": 2,
                        "shap_sum": -0.05,
                        "shap_abs_sum": 0.15
                    },
                    "summary": {
                        "total_shap_sum": 0.25,
                        "drug_contribution": 0.3,
                        "disease_contribution": -0.05,
                        "drug_abs_contribution": 0.3,
                        "disease_abs_contribution": 0.15
                    }
                }
            }
        ],
        "embedding_dim": 2,
        "analysis_metadata": {
            "top_k": 1,
            "total_predictions_analyzed": 1,
            "embedding_dimension": 2,
            "total_features": 4
        }
    }

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name

    try:
        save_shap_analysis(shap_results, temp_file)

        # Verify file exists
        assert os.path.exists(temp_file)

        # Load and verify contents
        import json
        with open(temp_file, 'r') as f:
            loaded_results = json.load(f)

        assert loaded_results == shap_results

        print("SHAP save test passed!")

    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_compute_shap_for_top_k_true_positives():
    """Test SHAP computation for top-K true positive predictions."""
    # Create synthetic data
    np.random.seed(42)

    # Simulate 100 drug-disease pairs with 10-dimensional embeddings each
    n_samples = 100
    embedding_dim = 10

    # Create feature vectors (concatenated drug + disease embeddings)
    feature_vectors = np.random.randn(n_samples, embedding_dim * 2)

    # Create synthetic labels with 20 true positives
    y_train = np.random.randint(0, 2, n_samples)
    y_true = np.random.randint(0, 2, n_samples)

    # Ensure we have at least 10 true positives
    true_positive_indices = np.random.choice(n_samples, size=20, replace=False)
    y_true[true_positive_indices] = 1

    # Train a simple Random Forest
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    model.fit(feature_vectors, y_train)

    # Get predictions
    predictions = model.predict_proba(feature_vectors)[:, 1]

    # Create pairs
    pairs = [(f"DRUG:{i}", f"DISEASE:{i}") for i in range(n_samples)]

    # Create embeddings dictionary (for reference)
    embeddings = {}
    for i in range(n_samples):
        embeddings[f"DRUG:{i}"] = feature_vectors[i, :embedding_dim]
        embeddings[f"DISEASE:{i}"] = feature_vectors[i, embedding_dim:]

    # Compute SHAP for top 5 TRUE POSITIVE predictions
    top_k = 5
    shap_results = compute_shap_for_top_k(
        model=model,
        feature_vectors=feature_vectors,
        predictions=predictions,
        pairs=pairs,
        embeddings=embeddings,
        top_k=top_k,
        y_true=y_true
    )

    # Verify all returned predictions are true positives
    predictions_list = shap_results["top_k_predictions"]
    assert len(predictions_list) == top_k

    for pred in predictions_list:
        # Find the pair in the original data
        drug_id = pred["drug_id"]
        disease_id = pred["disease_id"]
        pair_idx = pairs.index((drug_id, disease_id))

        # Verify it's a true positive
        assert y_true[pair_idx] == 1, f"Prediction at rank {pred['rank']} is not a true positive!"

    # Verify predictions are sorted by score (descending)
    scores = [pred["prediction_score"] for pred in predictions_list]
    assert scores == sorted(scores, reverse=True)

    # Verify all true positives have higher scores than the lowest selected
    min_selected_score = min(scores)
    tp_indices = np.where(y_true == 1)[0]
    for idx in tp_indices:
        if idx not in [pairs.index((p["drug_id"], p["disease_id"])) for p in predictions_list]:
            # This true positive wasn't selected, so it should have a lower score
            assert predictions[idx] <= min_selected_score

    print("True positive filtering test passed!")


if __name__ == "__main__":
    test_compute_shap_for_top_k()
    test_save_shap_analysis()
    test_compute_shap_for_top_k_true_positives()
    print("\nAll tests passed!")
