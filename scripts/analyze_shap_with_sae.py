#!/usr/bin/env python3
"""Example script for analyzing SHAP results with SAE feature mappings.

This script demonstrates how to combine SHAP analysis results with SAE
(Sparse Autoencoder) feature interpretations to understand which biological
concepts drive specific drug-disease predictions.

Usage:
    python scripts/analyze_shap_with_sae.py \
        --shap-file graphs/.../shap_top_10.json \
        --sae-features sae_feature_mappings.json \
        --top-n 10
"""
import json
import argparse
from pathlib import Path


def load_sae_mappings(sae_file):
    """Load SAE feature mappings.

    Expected format:
    {
        "drug_emb_18": "Cardiovascular activity pathway",
        "disease_emb_167": "Metabolic syndrome cluster",
        ...
    }
    """
    with open(sae_file, 'r') as f:
        return json.load(f)


def analyze_prediction(prediction, sae_mappings, top_n=10):
    """Analyze a single prediction with SAE mappings.

    Args:
        prediction: Prediction dict from SHAP results
        sae_mappings: Dict mapping feature names to biological descriptions
        top_n: Number of top features to display
    """
    print(f"\n{'='*80}")
    print(f"Rank {prediction['rank']}: {prediction['drug_id']} → {prediction['disease_id']}")
    print(f"Prediction Score: {prediction['prediction_score']:.4f}")
    print(f"{'='*80}")

    shap_analysis = prediction['shap_analysis']

    # Drug features
    print(f"\nDrug Features (Top {top_n}):")
    print(f"  Total SHAP contribution: {shap_analysis['drug']['shap_sum']:.4f}")
    print(f"  Non-zero features: {shap_analysis['drug']['num_nonzero_features']}")
    print()

    drug_features = shap_analysis['drug']['all_features'][:top_n]
    for i, feat in enumerate(drug_features, 1):
        feature_name = feat['feature']
        shap_value = feat['shap_value']

        # Look up biological interpretation
        interpretation = sae_mappings.get(feature_name, "Unknown feature")

        print(f"  {i:2d}. {feature_name:15s} = {shap_value:8.5f}  |  {interpretation}")

    # Disease features
    print(f"\nDisease Features (Top {top_n}):")
    print(f"  Total SHAP contribution: {shap_analysis['disease']['shap_sum']:.4f}")
    print(f"  Non-zero features: {shap_analysis['disease']['num_nonzero_features']}")
    print()

    disease_features = shap_analysis['disease']['all_features'][:top_n]
    for i, feat in enumerate(disease_features, 1):
        feature_name = feat['feature']
        shap_value = feat['shap_value']

        # Look up biological interpretation
        interpretation = sae_mappings.get(feature_name, "Unknown feature")

        print(f"  {i:2d}. {feature_name:15s} = {shap_value:8.5f}  |  {interpretation}")

    # Summary
    summary = shap_analysis['summary']
    print(f"\nSummary:")
    print(f"  Total SHAP: {summary['total_shap_sum']:.4f}")
    print(f"  Drug contribution: {summary['drug_contribution']:.4f} " +
          f"({summary['drug_contribution']/summary['total_shap_sum']*100:.1f}%)")
    print(f"  Disease contribution: {summary['disease_contribution']:.4f} " +
          f"({summary['disease_contribution']/summary['total_shap_sum']*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SHAP results with SAE feature mappings"
    )
    parser.add_argument("--shap-file", required=True,
                       help="Path to SHAP analysis JSON file (e.g., shap_top_10.json)")
    parser.add_argument("--sae-features", required=True,
                       help="Path to SAE feature mappings JSON file")
    parser.add_argument("--top-n", type=int, default=10,
                       help="Number of top features to display per category (default: 10)")
    parser.add_argument("--prediction-rank", type=int, default=None,
                       help="Analyze specific prediction rank (default: all)")

    args = parser.parse_args()

    # Load SHAP results
    print(f"Loading SHAP results from: {args.shap_file}")
    with open(args.shap_file, 'r') as f:
        shap_results = json.load(f)

    # Load SAE feature mappings
    print(f"Loading SAE feature mappings from: {args.sae_features}")
    sae_mappings = load_sae_mappings(args.sae_features)

    print(f"\nLoaded {len(sae_mappings)} SAE feature interpretations")
    print(f"Analyzing {len(shap_results['top_k_predictions'])} predictions")

    # Analyze predictions
    predictions = shap_results['top_k_predictions']

    if args.prediction_rank is not None:
        # Analyze specific prediction
        for pred in predictions:
            if pred['rank'] == args.prediction_rank:
                analyze_prediction(pred, sae_mappings, args.top_n)
                break
        else:
            print(f"Error: Prediction rank {args.prediction_rank} not found")
    else:
        # Analyze all predictions
        for pred in predictions:
            analyze_prediction(pred, sae_mappings, args.top_n)

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
