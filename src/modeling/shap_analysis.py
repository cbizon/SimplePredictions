#!/usr/bin/env python3
"""SHAP analysis for Random Forest models.

This module provides SHAP (SHapley Additive exPlanations) analysis for
top-K predictions, enabling interpretation of which features contribute
to specific predictions.
"""
import numpy as np
import shap
import json
from typing import Dict, List, Tuple, Any


def compute_shap_for_top_k(
    model,
    feature_vectors: np.ndarray,
    predictions: np.ndarray,
    pairs: List[Tuple[str, str]],
    embeddings: Dict[str, np.ndarray],
    top_k: int = 10,
    y_true: np.ndarray = None
) -> Dict[str, Any]:
    """Compute SHAP values for top-K predictions.

    Args:
        model: Trained Random Forest model
        feature_vectors: Array of concatenated feature vectors (n_samples, n_features)
        predictions: Array of prediction scores (n_samples,)
        pairs: List of (drug_id, disease_id) tuples corresponding to feature_vectors
        embeddings: Dictionary mapping node IDs to their embedding vectors
        top_k: Number of top predictions to analyze
        y_true: Optional array of true labels. If provided, only analyze true positives

    Returns:
        dict: SHAP analysis results containing:
            - top_k_predictions: List of dicts with pair info, score, and SHAP values
            - embedding_dim: Dimension of individual node embeddings
            - analysis_metadata: Info about the analysis
    """
    if y_true is not None:
        # Filter for true positives only
        true_positive_mask = y_true == 1
        tp_indices = np.where(true_positive_mask)[0]

        if len(tp_indices) == 0:
            raise ValueError("No true positives found in the evaluation set")

        print(f"Found {len(tp_indices)} true positives in evaluation set")

        # If top_k is 0 or negative, analyze ALL true positives
        if top_k <= 0:
            top_k = len(tp_indices)
            print(f"\n=== Computing SHAP values for ALL {top_k} TRUE POSITIVE predictions ===")
        else:
            print(f"\n=== Computing SHAP values for top {top_k} TRUE POSITIVE predictions ===")
            # Cap at number of available true positives
            top_k = min(top_k, len(tp_indices))

        # Get predictions for true positives only
        tp_predictions = predictions[tp_indices]

        # Sort true positives by prediction score
        tp_sorted_idx = np.argsort(tp_predictions)[::-1][:top_k]

        # Map back to original indices
        top_k_indices = tp_indices[tp_sorted_idx]
        top_k_features = feature_vectors[top_k_indices]
        top_k_scores = predictions[top_k_indices]
        top_k_pairs = [pairs[i] for i in top_k_indices]

        print(f"Selected {len(top_k_indices)} true positive predictions")
        print(f"Score range: {top_k_scores.min():.4f} - {top_k_scores.max():.4f}")
    else:
        print(f"\n=== Computing SHAP values for top {top_k} predictions (all) ===")

        # Get top K predictions overall
        top_k_indices = np.argsort(predictions)[::-1][:top_k]
        top_k_features = feature_vectors[top_k_indices]
        top_k_scores = predictions[top_k_indices]
        top_k_pairs = [pairs[i] for i in top_k_indices]

        print(f"Selected top {len(top_k_indices)} predictions")
        print(f"Score range: {top_k_scores.min():.4f} - {top_k_scores.max():.4f}")

    # Get embedding dimension (assuming concatenated drug+disease embeddings)
    total_features = feature_vectors.shape[1]
    embedding_dim = total_features // 2

    print(f"Total features: {total_features}")
    print(f"Embedding dimension per node: {embedding_dim}")

    # Create SHAP explainer
    print("Creating TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values for top K predictions
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(top_k_features)

    # For binary classification, shap_values might be a list [class_0, class_1]
    # We want SHAP values for the positive class (class 1)
    if isinstance(shap_values, list):
        shap_values_positive = shap_values[1]  # SHAP values for positive class
    else:
        shap_values_positive = shap_values

    print(f"SHAP values shape: {shap_values_positive.shape}")

    # Handle multi-output case (shape: n_samples, n_features, n_outputs)
    # For binary classification with predict_proba, we might get (n_samples, n_features, 2)
    if len(shap_values_positive.shape) == 3:
        # Take SHAP values for the positive class (index 1)
        shap_values_positive = shap_values_positive[:, :, 1]
        print(f"Extracted positive class SHAP values, new shape: {shap_values_positive.shape}")

    # Process results for each prediction
    results = []

    for i, (drug_id, disease_id) in enumerate(top_k_pairs):
        prediction_score = float(top_k_scores[i])
        shap_vals = shap_values_positive[i]

        # Split SHAP values into drug and disease components
        drug_shap = shap_vals[:embedding_dim]
        disease_shap = shap_vals[embedding_dim:]

        # Get feature names (indices into the original embeddings)
        drug_feature_names = [f"drug_emb_{j}" for j in range(embedding_dim)]
        disease_feature_names = [f"disease_emb_{j}" for j in range(embedding_dim)]

        # Combine drug and disease SHAP values with feature names
        # Filter out zero/near-zero values (threshold: 1e-10)
        threshold = 1e-10
        drug_shap_dict = {
            name: float(val)
            for name, val in zip(drug_feature_names, drug_shap)
            if abs(val) > threshold
        }
        disease_shap_dict = {
            name: float(val)
            for name, val in zip(disease_feature_names, disease_shap)
            if abs(val) > threshold
        }

        # Sort all features by absolute SHAP value (descending)
        drug_all_features_sorted = sorted(
            drug_shap_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        disease_all_features_sorted = sorted(
            disease_shap_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Get top contributing features by absolute SHAP value
        drug_top_features = drug_all_features_sorted[:10]  # Top 10 drug features
        disease_top_features = disease_all_features_sorted[:10]  # Top 10 disease features

        # Calculate summary statistics
        drug_shap_sum = float(np.sum(drug_shap))
        disease_shap_sum = float(np.sum(disease_shap))
        total_shap_sum = drug_shap_sum + disease_shap_sum

        drug_shap_abs_sum = float(np.sum(np.abs(drug_shap)))
        disease_shap_abs_sum = float(np.sum(np.abs(disease_shap)))

        result = {
            "rank": i + 1,
            "drug_id": drug_id,
            "disease_id": disease_id,
            "prediction_score": prediction_score,
            "shap_analysis": {
                "drug": {
                    "all_features": [{"feature": name, "shap_value": val} for name, val in drug_all_features_sorted],
                    "top_features": [{"feature": name, "shap_value": val} for name, val in drug_top_features],
                    "num_nonzero_features": len(drug_shap_dict),
                    "shap_sum": drug_shap_sum,
                    "shap_abs_sum": drug_shap_abs_sum
                },
                "disease": {
                    "all_features": [{"feature": name, "shap_value": val} for name, val in disease_all_features_sorted],
                    "top_features": [{"feature": name, "shap_value": val} for name, val in disease_top_features],
                    "num_nonzero_features": len(disease_shap_dict),
                    "shap_sum": disease_shap_sum,
                    "shap_abs_sum": disease_shap_abs_sum
                },
                "summary": {
                    "total_shap_sum": total_shap_sum,
                    "drug_contribution": drug_shap_sum,
                    "disease_contribution": disease_shap_sum,
                    "drug_abs_contribution": drug_shap_abs_sum,
                    "disease_abs_contribution": disease_shap_abs_sum
                }
            }
        }

        results.append(result)

        print(f"\nRank {i+1}: {drug_id} -> {disease_id}")
        print(f"  Score: {prediction_score:.4f}")
        print(f"  SHAP sum: {total_shap_sum:.4f} (drug: {drug_shap_sum:.4f}, disease: {disease_shap_sum:.4f})")
        print(f"  Top drug feature: {drug_top_features[0][0]} = {drug_top_features[0][1]:.4f}")
        print(f"  Top disease feature: {disease_top_features[0][0]} = {disease_top_features[0][1]:.4f}")

    # Create final output structure
    output = {
        "top_k_predictions": results,
        "embedding_dim": embedding_dim,
        "analysis_metadata": {
            "top_k": top_k,
            "total_predictions_analyzed": len(results),
            "embedding_dimension": embedding_dim,
            "total_features": total_features,
            "shap_method": "TreeExplainer",
            "model_type": type(model).__name__
        }
    }

    print(f"\n=== SHAP analysis complete ===")
    print(f"Analyzed {len(results)} predictions")

    return output


def save_shap_analysis(shap_results: Dict[str, Any], output_file: str):
    """Save SHAP analysis results to JSON file.

    Args:
        shap_results: SHAP analysis results from compute_shap_for_top_k
        output_file: Path to output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(shap_results, f, indent=2)

    print(f"SHAP analysis saved to: {output_file}")
