#!/usr/bin/env python3
"""Evaluate Random Forest predictions with ranking metrics and plots.

This script loads a trained model and evaluates it with ranking-based metrics
like Precision@K, Recall@K, and Hits@K.
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

from train_rf_model import load_embeddings, load_ground_truth, create_feature_vectors, extract_node_ids_from_positives, generate_negative_samples
from sklearn.model_selection import train_test_split


def calculate_precision_at_k(y_true, y_scores, k_values):
    """Calculate Precision@K for different values of K.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores (probabilities)
        k_values: List of K values to evaluate
        
    Returns:
        dict: Precision@K for each K value
    """
    # Sort by score (descending)
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
    """Calculate Recall@K for different values of K.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores (probabilities)
        k_values: List of K values to evaluate
        
    Returns:
        dict: Recall@K for each K value
    """
    # Sort by score (descending)
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
    """Calculate Hits@K (whether at least one positive is in top K).
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores (probabilities)
        k_values: List of K values to evaluate
        
    Returns:
        dict: Hits@K for each K value
    """
    # Sort by score (descending)
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
    """Create plots for ranking metrics.
    
    Args:
        precision_at_k: Dict of Precision@K values
        recall_at_k: Dict of Recall@K values  
        hits_at_k: Dict of Hits@K values
        k_values: List of K values
        output_dir: Directory to save plots
    """
    # Set up the plot style
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
    """Create precision-recall curve plot.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores
        output_dir: Directory to save plot
    """
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


def evaluate_model(graph_dir, ground_truth_file, output_dir=None, negative_ratio=1):
    """Main evaluation pipeline using proper train-test split.
    
    Args:
        graph_dir: Path to graph directory with model
        ground_truth_file: Path to ground truth file
        output_dir: Output directory (default: graph_dir/evaluation)
        negative_ratio: Ratio used during training (to recreate same split)
    """
    if output_dir is None:
        output_dir = os.path.join(graph_dir, "evaluation")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the trained model
    model_file = os.path.join(graph_dir, "models", "rf_model.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}")
    
    with open(model_file, 'rb') as f:
        rf_model = pickle.load(f)
    
    print(f"Loaded model from: {model_file}")
    
    # Load embeddings and ground truth (same as training)
    embeddings_file = os.path.join(graph_dir, "embeddings", "embeddings.emb")
    embeddings = load_embeddings(embeddings_file)
    
    positive_pairs = load_ground_truth(ground_truth_file, embeddings)
    drug_ids, disease_ids = extract_node_ids_from_positives(positive_pairs)
    
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
    plot_ranking_metrics(precision_at_k, recall_at_k, hits_at_k, k_values, output_dir)
    create_precision_recall_curve(y_true, y_scores, output_dir)
    
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
    
    results_file = os.path.join(output_dir, "test_ranking_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save test set pairs for reference
    test_pairs_file = os.path.join(output_dir, "test_pairs.json")
    test_data = {
        'test_pairs': test_pairs,
        'test_scores': y_scores.tolist(),
        'test_labels': y_true.tolist()
    }
    with open(test_pairs_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Print summary
    print("\n=== Test Set Ranking Metrics Summary ===")
    for k in [1, 5, 10, 20, 50, 100]:
        if k in precision_at_k:
            print(f"K={k:3d}: Precision={precision_at_k[k]:.4f}, Recall={recall_at_k[k]:.4f}, Hits={hits_at_k[k]:.4f}")
    
    print(f"\nTest set evaluation results saved to: {output_dir}")
    return results


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Evaluate RF model with ranking metrics")
    parser.add_argument("--graph-dir", required=True,
                       help="Path to graph directory containing trained model")
    parser.add_argument("--ground-truth", required=True,
                       help="Path to ground truth file")
    parser.add_argument("--output-dir",
                       help="Output directory (default: {graph_dir}/evaluation)")
    parser.add_argument("--negative-ratio", type=int, default=1,
                       help="Negative ratio used during training (to recreate same split)")
    
    args = parser.parse_args()
    
    evaluate_model(args.graph_dir, args.ground_truth, args.output_dir, args.negative_ratio)


if __name__ == "__main__":
    main()