#!/usr/bin/env python3
"""Evaluate models with automatic versioning and provenance tracking.

This script evaluates trained models and automatically manages
versioned output directories with full provenance metadata.
"""
import os
import argparse
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import seaborn as sns
from datetime import datetime
from pathlib import Path

from .train_model import load_embeddings, load_ground_truth, create_feature_vectors, extract_node_ids_from_positives, generate_negative_samples
from sklearn.model_selection import train_test_split


def get_next_evaluation_version(evaluations_dir):
    """Find the next available evaluation version number.
    
    Args:
        evaluations_dir: Base evaluations directory
        
    Returns:
        int: Next available version number
    """
    if not os.path.exists(evaluations_dir):
        return 0
    
    existing_versions = []
    for item in os.listdir(evaluations_dir):
        if item.startswith("evaluation_") and os.path.isdir(os.path.join(evaluations_dir, item)):
            try:
                version_num = int(item.split("_")[1])
                existing_versions.append(version_num)
            except (IndexError, ValueError):
                continue
    
    return max(existing_versions) + 1 if existing_versions else 0


def get_model_info(model_file):
    """Extract model metadata from model file and associated provenance.
    
    Args:
        model_file: Path to model pickle file
        
    Returns:
        dict: Model info including provenance if available
    """
    if not os.path.exists(model_file):
        return {"error": f"Model file not found: {model_file}"}
    
    model_dir = os.path.dirname(model_file)
    provenance_file = os.path.join(model_dir, "provenance.json")
    
    model_info = {
        "model_file": model_file,
        "model_dir": model_dir
    }
    
    if os.path.exists(provenance_file):
        with open(provenance_file, 'r') as f:
            model_provenance = json.load(f)
            model_info["model_provenance"] = model_provenance
    
    # Get model file size for basic info
    model_info["model_size_bytes"] = os.path.getsize(model_file)
    
    return model_info


def calculate_precision_at_k(y_true, y_scores, k_values):
    """Calculate Precision@K for different values of K."""
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_true = y_true[sorted_indices]
    
    precision_at_k = {}
    for k in k_values:
        if k <= len(sorted_true):
            top_k_true = sorted_true[:k]
            precision_at_k[k] = np.sum(top_k_true) / k
        else:
            precision_at_k[k] = np.nan
    
    return precision_at_k


def calculate_recall_at_k(y_true, y_scores, k_values):
    """Calculate Recall@K for different values of K."""
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_true = y_true[sorted_indices]
    
    total_positives = np.sum(y_true)
    recall_at_k = {}
    
    for k in k_values:
        if k <= len(sorted_true):
            top_k_true = sorted_true[:k]
            recall_at_k[k] = np.sum(top_k_true) / total_positives if total_positives > 0 else 0
        else:
            recall_at_k[k] = np.nan
    
    return recall_at_k


def calculate_hits_at_k(y_true, y_scores, k_values):
    """Calculate Hits@K (whether at least one positive is in top K)."""
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_true = y_true[sorted_indices]
    
    hits_at_k = {}
    for k in k_values:
        if k <= len(sorted_true):
            top_k_true = sorted_true[:k]
            hits_at_k[k] = 1 if np.sum(top_k_true) > 0 else 0
        else:
            hits_at_k[k] = np.nan
    
    return hits_at_k


def plot_ranking_metrics(precision_at_k, recall_at_k, hits_at_k, k_values, output_dir):
    """Create plots for ranking metrics."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Drug-Disease Prediction Ranking Metrics', fontsize=16)
    
    # Plot 1: Precision@K
    k_vals = [k for k in k_values if not np.isnan(precision_at_k[k])]
    prec_vals = [precision_at_k[k] for k in k_vals]
    
    axes[0, 0].plot(k_vals, prec_vals, 'o-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('K (Top Predictions)')
    axes[0, 0].set_ylabel('Precision@K')
    axes[0, 0].set_title('Precision@K: Accuracy of Top K Predictions')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Plot 2: Recall@K
    recall_vals = [recall_at_k[k] for k in k_vals]
    
    axes[0, 1].plot(k_vals, recall_vals, 'o-', linewidth=2, markersize=6, color='orange')
    axes[0, 1].set_xlabel('K (Top Predictions)')
    axes[0, 1].set_ylabel('Recall@K')
    axes[0, 1].set_title('Recall@K: Coverage of True Positives')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: Hits@K
    hits_vals = [hits_at_k[k] for k in k_vals]
    
    axes[1, 0].plot(k_vals, hits_vals, 'o-', linewidth=2, markersize=6, color='green')
    axes[1, 0].set_xlabel('K (Top Predictions)')
    axes[1, 0].set_ylabel('Hits@K')
    axes[1, 0].set_title('Hits@K: At Least One Hit in Top K')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Plot 4: All metrics together
    axes[1, 1].plot(k_vals, prec_vals, 'o-', label='Precision@K', linewidth=2, markersize=4)
    axes[1, 1].plot(k_vals, recall_vals, 'o-', label='Recall@K', linewidth=2, markersize=4)
    axes[1, 1].plot(k_vals, hits_vals, 'o-', label='Hits@K', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('K (Top Predictions)')
    axes[1, 1].set_ylabel('Metric Value')
    axes[1, 1].set_title('All Ranking Metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, "ranking_metrics.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Ranking metrics plot saved: {plot_file}")
    
    plt.close()


def create_precision_recall_curve(y_true, y_scores, output_dir):
    """Create precision-recall curve plot."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, linewidth=2, label=f'PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision') 
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='red', linestyle='--', 
                label=f'Random Baseline ({baseline:.3f})')
    plt.legend()
    
    # Save plot
    plot_file = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Precision-recall curve saved: {plot_file}")
    
    plt.close()


def evaluate_model_with_provenance(model_path,
                                 graph_dir, 
                                 ground_truth_file,
                                 model_version=None,
                                 embeddings_version=None):
    """Evaluate model with automatic versioning and provenance.
    
    Args:
        model_path: Path to specific model file (e.g., graphs/robokop_base/CCDD/models/model_2/rf_model.pkl)
        graph_dir: Path to graph directory (e.g., graphs/robokop_base/CCDD) 
        ground_truth_file: Path to ground truth CSV file
        model_version: Specific model version (e.g., "model_2")
        embeddings_version: Specific embeddings version (e.g., "embeddings_1")
        
    Returns:
        str: Path to the generated evaluation directory
    """
    # Validate model file
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine output directory structure - top-level evaluations directory
    evaluations_base_dir = "evaluations"
    os.makedirs(evaluations_base_dir, exist_ok=True)
    
    # Get next version number
    version = get_next_evaluation_version(evaluations_base_dir)
    version_dir = os.path.join(evaluations_base_dir, f"evaluation_{version}")
    os.makedirs(version_dir, exist_ok=True)
    
    print(f"Running evaluation version {version}")
    print(f"Model: {model_path}")
    print(f"Output: {version_dir}")
    
    # Record start time
    start_time = datetime.now()
    
    # Get model info for provenance
    model_info = get_model_info(model_path)
    
    # Load the trained model
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
    
    print(f"Loaded model from: {model_path}")
    
    # Determine embeddings file
    if embeddings_version:
        embeddings_file = os.path.join(graph_dir, "embeddings", embeddings_version, "embeddings.emb")
    else:
        # Use the default embeddings.emb file
        embeddings_file = os.path.join(graph_dir, "embeddings", "embeddings.emb")
    
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    # Load embeddings and ground truth
    embeddings = load_embeddings(embeddings_file)
    positive_pairs = load_ground_truth(ground_truth_file, embeddings)
    
    # Create ground truth stats manually for provenance
    ground_truth_stats = {
        "ground_truth_file": ground_truth_file,
        "final_positive_pairs": len(positive_pairs),
        "unique_drugs": len(set(pair[0] for pair in positive_pairs)),
        "unique_diseases": len(set(pair[1] for pair in positive_pairs))
    }
    drug_ids, disease_ids = extract_node_ids_from_positives(positive_pairs)
    
    # Need to determine negative_ratio from model provenance
    negative_ratio = 1  # default
    if "model_provenance" in model_info and "model_parameters" in model_info["model_provenance"]:
        negative_ratio = model_info["model_provenance"]["model_parameters"].get("negative_ratio", 1)
    
    print(f"Using negative_ratio={negative_ratio} from model provenance")
    print(f"Recreating train-test split to get test set only...")
    
    # Recreate the EXACT same data preparation as training
    pos_features, pos_labels = create_feature_vectors(embeddings, positive_pairs)
    pos_targets = np.ones(len(pos_features))
    
    # Generate same negative samples (with same random seed)
    negative_pairs = generate_negative_samples(positive_pairs, drug_ids, disease_ids, negative_ratio)
    neg_features, neg_labels = create_feature_vectors(embeddings, negative_pairs)
    neg_targets = np.zeros(len(neg_features))
    
    # Combine positive and negative samples (same as training)
    X = np.vstack([pos_features, neg_features])
    y = np.hstack([pos_targets, neg_targets])
    pair_labels = pos_labels + neg_labels
    
    # Recreate the EXACT same train-test split (same random seed)
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, range(len(pair_labels)), test_size=0.2, random_state=42, stratify=y
    )
    
    # Get test set pairs and labels
    test_pairs = [pair_labels[i] for i in indices_test]
    
    print(f"Test set: {len(X_test)} samples ({np.sum(y_test)} positive, {len(y_test) - np.sum(y_test)} negative)")
    
    # Generate predictions ONLY for test set
    y_scores = rf_model.predict_proba(X_test)[:, 1]  # Probability of positive class
    y_true = y_test
    
    print(f"Test predictions: {len(y_scores)}")
    print(f"Test positives: {np.sum(y_true)}")
    print(f"Test negatives: {len(y_true) - np.sum(y_true)}")
    
    # Define K values to evaluate
    k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    k_values = [k for k in k_values if k <= len(y_scores)]  # Only valid K values
    
    # Calculate ranking metrics
    precision_at_k = calculate_precision_at_k(y_true, y_scores, k_values)
    recall_at_k = calculate_recall_at_k(y_true, y_scores, k_values)
    hits_at_k = calculate_hits_at_k(y_true, y_scores, k_values)
    
    # Create plots
    plot_ranking_metrics(precision_at_k, recall_at_k, hits_at_k, k_values, version_dir)
    create_precision_recall_curve(y_true, y_scores, version_dir)
    
    # Record end time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Create comprehensive provenance metadata
    provenance = {
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "script": "evaluate_model.py",
        "version": f"evaluation_{version}",
        "model_info": model_info,
        "input_data": {
            "graph_dir": graph_dir,
            "embeddings_file": embeddings_file,
            "embeddings_version": embeddings_version,
            "ground_truth": ground_truth_stats
        },
        "evaluation_parameters": {
            "k_values": k_values,
            "negative_ratio_used": negative_ratio,
            "test_size": 0.2,
            "random_state": 42
        },
        "test_data_info": {
            "total_test_samples": len(y_scores),
            "test_positives": int(np.sum(y_true)),
            "test_negatives": int(len(y_true) - np.sum(y_true)),
            "test_pairs_count": len(test_pairs)
        },
        "ranking_metrics": {
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "hits_at_k": hits_at_k
        },
        "description": f"Model evaluation version {version} for {os.path.basename(model_path)}"
    }
    
    # Save provenance file
    provenance_file = os.path.join(version_dir, "provenance.json")
    with open(provenance_file, 'w') as f:
        json.dump(provenance, f, indent=2)
    
    # Save detailed results
    results = {
        'precision_at_k': precision_at_k,
        'recall_at_k': recall_at_k,
        'hits_at_k': hits_at_k,
        'k_values': k_values,
        'total_test_predictions': len(y_scores),
        'test_positives': int(np.sum(y_true)),
        'test_negatives': int(len(y_true) - np.sum(y_true)),
        'negative_ratio_used': negative_ratio
    }
    
    results_file = os.path.join(version_dir, "test_ranking_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save test set pairs for reference
    test_pairs_file = os.path.join(version_dir, "test_pairs.json")
    test_data = {
        'test_pairs': test_pairs,
        'test_scores': y_scores.tolist(),
        'test_labels': y_true.tolist()
    }
    with open(test_pairs_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Provenance saved: {provenance_file}")
    print(f"Results saved: {results_file}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Print summary
    print("\n=== Test Set Ranking Metrics Summary ===")
    for k in [1, 5, 10, 20, 50, 100]:
        if k in precision_at_k and not np.isnan(precision_at_k[k]):
            print(f"K={k:3d}: Precision={precision_at_k[k]:.4f}, Recall={recall_at_k[k]:.4f}, Hits={hits_at_k[k]:.4f}")
    
    print(f"\nEvaluation complete!")
    print(f"Output directory: {version_dir}")
    
    return version_dir


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Evaluate models with versioning and provenance")
    parser.add_argument("--model-path", required=True,
                       help="Path to specific model file (e.g., graphs/robokop_base/CCDD/models/model_2/rf_model.pkl)")
    parser.add_argument("--graph-dir", required=True,
                       help="Path to graph directory (e.g., graphs/robokop_base/CCDD)")
    parser.add_argument("--ground-truth", required=True,
                       help="Path to ground truth CSV file")
    parser.add_argument("--model-version",
                       help="Specific model version (e.g., model_2)")
    parser.add_argument("--embeddings-version", 
                       help="Specific embeddings version to use (e.g., embeddings_2)")
    
    args = parser.parse_args()
    
    version_dir = evaluate_model_with_provenance(
        model_path=args.model_path,
        graph_dir=args.graph_dir,
        ground_truth_file=args.ground_truth,
        model_version=args.model_version,
        embeddings_version=args.embeddings_version
    )
    
    print(f"\nModel evaluation complete!")
    print(f"Output directory: {version_dir}")


if __name__ == "__main__":
    main()