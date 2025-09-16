#!/usr/bin/env python3
"""Evaluate models with automatic versioning and provenance tracking.

This script evaluates trained models and automatically manages
versioned output directories with full provenance metadata.

Supports both single model evaluation and multi-model comparison with
comparative plots and comprehensive provenance tracking.
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

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.train_model import load_embeddings, load_ground_truth, create_feature_vectors, extract_node_ids_from_positives, generate_negative_samples
from sklearn.model_selection import train_test_split


def extract_model_metadata(model_dir):
    """Extract metadata from a model directory.
    
    Args:
        model_dir: Path to model directory (e.g., graphs/robokop_base/CCDD/models/model_2)
        
    Returns:
        dict: Model metadata including paths to embeddings, ground truth, etc.
    """
    # Read provenance file
    provenance_file = os.path.join(model_dir, "provenance.json")
    if not os.path.exists(provenance_file):
        raise FileNotFoundError(f"Provenance file not found: {provenance_file}")
    
    with open(provenance_file, 'r') as f:
        provenance = json.load(f)
    
    # Extract key paths from provenance
    input_data = provenance.get("input_data", {})
    
    # Infer graph directory from model directory path
    # New structure: graphs/robokop_base/CCDD/embeddings/embeddings_0/models/model_0 -> graphs/robokop_base/CCDD
    model_dir_parts = Path(model_dir).parts
    if "models" in model_dir_parts and "embeddings" in model_dir_parts:
        embeddings_index = model_dir_parts.index("embeddings")
        graph_dir = str(Path(*model_dir_parts[:embeddings_index]))
    else:
        # Fallback: use the graph_dir from provenance if available
        graph_dir = input_data.get("graph_dir")
        if not graph_dir:
            raise ValueError(f"Cannot determine graph directory from model path: {model_dir}")
    
    # Get model file path
    model_file = os.path.join(model_dir, "rf_model.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    # Extract other metadata
    metadata = {
        "model_dir": model_dir,
        "model_file": model_file,
        "graph_dir": graph_dir,
        "embeddings_file": input_data.get("embeddings_file"),
        "embeddings_version": input_data.get("embeddings_version"),
        "ground_truth_file": input_data.get("ground_truth", {}).get("ground_truth_file"),
        "provenance": provenance
    }
    
    # Validate critical paths exist
    if metadata["embeddings_file"] and not os.path.exists(metadata["embeddings_file"]):
        raise FileNotFoundError(f"Embeddings file not found: {metadata['embeddings_file']}")
    
    if metadata["ground_truth_file"] and not os.path.exists(metadata["ground_truth_file"]):
        raise FileNotFoundError(f"Ground truth file not found: {metadata['ground_truth_file']}")
    
    return metadata





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


def evaluate_single_model_core(rf_model, embeddings, positive_pairs, drug_ids, disease_ids, negative_ratio, model_dir, original_gt_count, ground_truth_file):
    """Core evaluation logic for a single model.
    
    Args:
        rf_model: Trained model
        embeddings: Node embeddings
        positive_pairs: Ground truth positive pairs
        drug_ids: Unique drug IDs from ground truth
        disease_ids: Unique disease IDs from ground truth  
        negative_ratio: Negative sampling ratio used during training
        model_dir: Directory containing the model and training_pairs.json file
        
    Returns:
        dict: Evaluation results including metrics, test data info, and test pairs
    """
    print(f"=== GROUND TRUTH & UNIVERSE ===")
    print(f"Loaded positive pairs: {len(positive_pairs)}")
    print(f"Unique drugs in ground truth: {len(drug_ids)}")  
    print(f"Unique diseases in ground truth: {len(disease_ids)}")
    print(f"Total possible combinations: {len(drug_ids) * len(disease_ids)}")
    
    # Check embedding coverage for all indications
    all_positive_pairs, _ = load_ground_truth(ground_truth_file, embeddings=None)
    all_drug_ids = set(pair[0] for pair in all_positive_pairs)
    all_disease_ids = set(pair[1] for pair in all_positive_pairs)
    drugs_with_embeddings = len([d for d in all_drug_ids if d in embeddings])
    diseases_with_embeddings = len([d for d in all_disease_ids if d in embeddings])
    
    print(f"=== EMBEDDING COVERAGE ===")
    print(f"Drugs in original indications: {len(all_drug_ids)}, with embeddings: {drugs_with_embeddings} ({drugs_with_embeddings/len(all_drug_ids)*100:.1f}%)")
    print(f"Diseases in original indications: {len(all_disease_ids)}, with embeddings: {diseases_with_embeddings} ({diseases_with_embeddings/len(all_disease_ids)*100:.1f}%)")
    
    # Generate ALL possible drug-disease combinations from ground truth
    all_combinations = [(drug, disease) for drug in drug_ids for disease in disease_ids]
    
    # Read training pairs from file (instead of reconstructing)
    training_pairs_file = os.path.join(model_dir, "training_pairs.json")
    
    if os.path.exists(training_pairs_file):
        print(f"=== READING TRAINING PAIRS ===")
        with open(training_pairs_file, 'r') as f:
            training_data = json.load(f)
        
        training_positives = [tuple(pair) for pair in training_data["training_positives"]]
        training_negatives = [tuple(pair) for pair in training_data["training_negatives"]]
        training_pairs_to_exclude = set(training_positives + training_negatives)
        
        print(f"Training positives to exclude: {len(training_positives)}")
        print(f"Training negatives to exclude: {len(training_negatives)}")
        print(f"Total training pairs to exclude: {len(training_pairs_to_exclude)}")
        
        print(f"Ground truth positives used in training: {len(training_positives)}")
        print(f"Ground truth positives remaining for evaluation: {len(positive_pairs) - len(training_positives)}")
    else:
        print(f"Warning: Training pairs file not found: {training_pairs_file}")
        print("Using empty exclusion set (will evaluate on all combinations)")
        training_pairs_to_exclude = set()
    
    # Remove training pairs from evaluation set
    evaluation_combinations = [pair for pair in all_combinations if pair not in training_pairs_to_exclude]
    
    print(f"=== EVALUATION SET ===") 
    print(f"All combinations: {len(all_combinations)}")
    print(f"Training pairs to exclude: {len(training_pairs_to_exclude)}")
    print(f"Evaluation combinations: {len(evaluation_combinations)}")
    
    # Create features for evaluation combinations - pad missing embeddings with zeros
    # This ensures all models evaluate the same set of drug-disease pairs
    eval_features, eval_pairs = create_feature_vectors(embeddings, evaluation_combinations, pad_missing=True)
    
    # Label evaluation combinations - only the test positives, not training positives
    evaluation_positives = set(positive_pairs) - set(training_positives)
    y_true = np.array([1 if pair in evaluation_positives else 0 for pair in eval_pairs])
    
    print(f"=== EVALUATION SET ===") 
    print(f"All combinations: {len(all_combinations)}")
    print(f"Evaluation combinations: {len(evaluation_combinations)}")
    print(f"Evaluation features created: {len(eval_features)}")
    print(f"Evaluation pairs labeled as positive: {int(np.sum(y_true))}")
    
    # Max recall should now be 1.0 since we're only looking for test positives in the evaluation set
    print(f"Max possible recall: 1.0 (all evaluation positives should be findable)")
    print(f"Evaluation positives should equal remaining GT positives: {int(np.sum(y_true)) == len(evaluation_positives)}")
    
    # Generate predictions for ALL evaluation combinations  
    y_scores = rf_model.predict_proba(eval_features)[:, 1]  # Probability of positive class
    
    # Define K values to evaluate - logarithmic spacing plus maximum
    max_k = len(y_scores)
    k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
    k_values = [k for k in k_values if k <= max_k]  # Only valid K values
    
    # Always include the maximum K
    if max_k not in k_values:
        k_values.append(max_k)
    
    k_values = sorted(k_values)
    
    # Calculate ranking metrics (global precision/recall, per-disease hits)
    precision_at_k = calculate_precision_at_k(y_true, y_scores, k_values)
    recall_at_k = calculate_recall_at_k(y_true, y_scores, k_values)
    # Calculate total recall using original ground truth count
    total_recall_denominator = original_gt_count - len(training_positives)
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_true = y_true[sorted_indices]
    total_recall_at_k = {k: np.sum(sorted_true[:k]) / total_recall_denominator if total_recall_denominator > 0 else 0 for k in k_values}
    # Calculate theoretical maximum (all evaluation positives found)
    total_recall_max = int(np.sum(y_true)) / total_recall_denominator if total_recall_denominator > 0 else 0
    hits_at_k = calculate_hits_at_k_per_disease(eval_pairs, y_true, y_scores, k_values)
    
    return {
        'precision_at_k': precision_at_k,
        'recall_at_k': recall_at_k,
        'total_recall_at_k': total_recall_at_k,
        'total_recall_max': total_recall_max,
        'hits_at_k': hits_at_k,
        'k_values': k_values,
        'evaluation_combinations': len(evaluation_combinations),
        'evaluation_positives': int(np.sum(y_true)),
        'evaluation_negatives': int(len(y_true) - np.sum(y_true)),
        'training_pairs_excluded': len(training_pairs_to_exclude),
        'all_combinations': len(all_combinations),
        'y_true': y_true,
        'y_scores': y_scores,
        'test_pairs': eval_pairs
    }


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




def calculate_hits_at_k_per_disease(test_pairs, y_true, y_scores, k_values):
    """Calculate Hits@K as fraction of diseases with at least one hit in top K.
    
    Args:
        test_pairs: List of (drug_id, disease_id) tuples
        y_true: Array of true labels
        y_scores: Array of predicted scores
        k_values: List of K values to evaluate
        
    Returns:
        dict: Hits@K values (fraction of diseases with hits)
    """
    # Group predictions by disease
    disease_groups = {}
    for i, (drug_id, disease_id) in enumerate(test_pairs):
        if disease_id not in disease_groups:
            disease_groups[disease_id] = {'indices': [], 'drug_ids': []}
        disease_groups[disease_id]['indices'].append(i)
        disease_groups[disease_id]['drug_ids'].append(drug_id)
    
    hits_at_k = {k: 0 for k in k_values}
    total_diseases = len(disease_groups)
    
    for disease_id, group in disease_groups.items():
        indices = np.array(group['indices'])
        disease_y_true = y_true[indices]
        disease_y_scores = y_scores[indices]
        
        # Sort by scores (descending)
        sorted_indices = np.argsort(disease_y_scores)[::-1]
        sorted_true = disease_y_true[sorted_indices]
        
        # Check hits@k for this disease
        for k in k_values:
            top_k_true = sorted_true[:k]
            if np.sum(top_k_true) > 0:  # At least one hit
                hits_at_k[k] += 1
    
    # Convert to fraction
    fraction_hits_at_k = {}
    for k in k_values:
        if total_diseases > 0:
            fraction_hits_at_k[k] = hits_at_k[k] / total_diseases
        else:
            fraction_hits_at_k[k] = np.nan
    
    return fraction_hits_at_k


# Legacy global functions for backward compatibility
def calculate_precision_at_k(y_true, y_scores, k_values):
    """Calculate Precision@K for different values of K (global ranking)."""
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
    """Calculate Recall@K for different values of K (global ranking)."""
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




def plot_ranking_metrics(metrics_data, output_dir):
    """Create plots for ranking metrics with two K-range views.
    
    Layout: 3 rows (Precision, Recall, Hits) x 2 columns (K≤1000, Full Range)
    All models are shown together on each plot for comparison.
    
    Args:
        metrics_data: Dictionary with model labels as keys and their metrics as values
                     Format: {label: {precision_at_k, recall_at_k, hits_at_k, k_values}}
        output_dir: Output directory for plots
    """
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create grid: 4 rows x 2 columns
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    fig.suptitle('Drug-Disease Prediction Ranking Metrics', fontsize=16)
    
    # Get all k values (union of all models)
    all_k_values = set()
    for model_data in metrics_data.values():
        all_k_values.update(model_data['k_values'])
    all_k_values = sorted(all_k_values)
    
    k_threshold = 1000  # Threshold for zoomed view
    
    # Row 0: Precision@K
    for i, (label, model_data) in enumerate(metrics_data.items()):
        # Column 0: K ≤ 1000 (zoomed view)
        k_vals_zoom = [k for k in model_data['k_values'] if k <= k_threshold and not np.isnan(model_data['precision_at_k'][k])]
        prec_vals_zoom = [model_data['precision_at_k'][k] for k in k_vals_zoom]
        axes[0, 0].plot(k_vals_zoom, prec_vals_zoom, 'o-', linewidth=2, markersize=6, label=label)
        
        # Column 1: Full range - use ALL k values for this model
        k_vals_full = [k for k in all_k_values if k in model_data['k_values'] and not np.isnan(model_data['precision_at_k'][k])]
        prec_vals_full = [model_data['precision_at_k'][k] for k in k_vals_full]
        axes[0, 1].plot(k_vals_full, prec_vals_full, 'o-', linewidth=2, markersize=4, label=label)
    
    # Configure Precision@K plots
    axes[0, 0].set_ylabel('Precision@K')
    axes[0, 0].set_title(f'Precision@K (K ≤ {k_threshold})')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    if len(metrics_data) > 1:
        axes[0, 0].legend()
    
    axes[0, 1].set_ylabel('Precision@K')
    axes[0, 1].set_title('Precision@K (Full Range)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    if len(metrics_data) > 1:
        axes[0, 1].legend()
    
    # Row 1: Recall@K
    for i, (label, model_data) in enumerate(metrics_data.items()):
        # Column 0: K ≤ 1000 (zoomed view)
        k_vals_zoom = [k for k in model_data['k_values'] if k <= k_threshold and not np.isnan(model_data['recall_at_k'][k])]
        recall_vals_zoom = [model_data['recall_at_k'][k] for k in k_vals_zoom]
        axes[1, 0].plot(k_vals_zoom, recall_vals_zoom, 'o-', linewidth=2, markersize=6, label=label)
        
        # Column 1: Full range - use ALL k values for this model
        k_vals_full = [k for k in all_k_values if k in model_data['k_values'] and not np.isnan(model_data['recall_at_k'][k])]
        recall_vals_full = [model_data['recall_at_k'][k] for k in k_vals_full]
        axes[1, 1].plot(k_vals_full, recall_vals_full, 'o-', linewidth=2, markersize=4, label=label)
    
    # Configure Recall@K plots
    axes[1, 0].set_ylabel('Recall@K')
    axes[1, 0].set_title(f'Recall@K (K ≤ {k_threshold})')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    if len(metrics_data) > 1:
        axes[1, 0].legend()
    
    axes[1, 1].set_ylabel('Recall@K')
    axes[1, 1].set_title('Recall@K (Full Range)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    if len(metrics_data) > 1:
        axes[1, 1].legend()
    
    # Row 2: Total Recall@K
    for i, (label, model_data) in enumerate(metrics_data.items()):
        # Column 0: K ≤ 1000 (zoomed view)
        k_vals_zoom = [k for k in model_data['k_values'] if k <= k_threshold and not np.isnan(model_data['total_recall_at_k'][k])]
        total_recall_vals_zoom = [model_data['total_recall_at_k'][k] for k in k_vals_zoom]
        axes[2, 0].plot(k_vals_zoom, total_recall_vals_zoom, 'o-', linewidth=2, markersize=6, label=label)
        
        # Column 1: Full range - use ALL k values for this model
        k_vals_full = [k for k in all_k_values if k in model_data['k_values'] and not np.isnan(model_data['total_recall_at_k'][k])]
        total_recall_vals_full = [model_data['total_recall_at_k'][k] for k in k_vals_full]
        axes[2, 1].plot(k_vals_full, total_recall_vals_full, 'o-', linewidth=2, markersize=4, label=label)
    
    # Configure Total Recall@K plots
    axes[2, 0].set_ylabel('Total Recall@K')
    axes[2, 0].set_title(f'Total Recall@K (K ≤ {k_threshold})')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_ylim(0, 1)
    
    # Add theoretical maximum lines - each model has its own max
    for i, (label, model_data) in enumerate(metrics_data.items()):
        if 'total_recall_max' in model_data:
            theoretical_max = model_data['total_recall_max']
            # Get color from the lines plotted on the current axes
            color = axes[2, 0].lines[i].get_color() if i < len(axes[2, 0].lines) else f'C{i}'
            axes[2, 0].axhline(y=theoretical_max, color=color, linestyle='--', alpha=0.7, 
                              label=f'{label} Max ({theoretical_max:.3f})')
            axes[2, 1].axhline(y=theoretical_max, color=color, linestyle='--', alpha=0.7,
                              label=f'{label} Max ({theoretical_max:.3f})')
    
    if len(metrics_data) > 1:
        axes[2, 0].legend()
    
    axes[2, 1].set_ylabel('Total Recall@K')
    axes[2, 1].set_title('Total Recall@K (Full Range)')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_ylim(0, 1)
    if len(metrics_data) > 1:
        axes[2, 1].legend()
    
    # Row 3: Hits@K
    for i, (label, model_data) in enumerate(metrics_data.items()):
        # Column 0: K ≤ 1000 (zoomed view)
        k_vals_zoom = [k for k in model_data['k_values'] if k <= k_threshold and not np.isnan(model_data['hits_at_k'][k])]
        hits_vals_zoom = [model_data['hits_at_k'][k] for k in k_vals_zoom]
        axes[3, 0].plot(k_vals_zoom, hits_vals_zoom, 'o-', linewidth=2, markersize=6, label=label)
        
        # Column 1: Full range - use ALL k values for this model
        k_vals_full = [k for k in all_k_values if k in model_data['k_values'] and not np.isnan(model_data['hits_at_k'][k])]
        hits_vals_full = [model_data['hits_at_k'][k] for k in k_vals_full]
        axes[3, 1].plot(k_vals_full, hits_vals_full, 'o-', linewidth=2, markersize=4, label=label)
    
    # Configure Hits@K plots
    axes[3, 0].set_xlabel('K (Top Predictions)')
    axes[3, 0].set_ylabel('Hits@K')
    axes[3, 0].set_title(f'Hits@K (K ≤ {k_threshold})')
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].set_ylim(0, 1)
    if len(metrics_data) > 1:
        axes[3, 0].legend()
    
    axes[3, 1].set_xlabel('K (Top Predictions)')
    axes[3, 1].set_ylabel('Hits@K')
    axes[3, 1].set_title('Hits@K (Full Range)')
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].set_ylim(0, 1)
    if len(metrics_data) > 1:
        axes[3, 1].legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, "ranking_metrics.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Ranking metrics plot saved: {plot_file}")
    
    plt.close()


def create_precision_recall_curve(pr_curve_data, output_dir):
    """Create precision-recall curve plot with support for multiple models.
    
    Args:
        pr_curve_data: Dictionary with model labels as keys and their PR data as values
                      Format: {label: {'y_true': array, 'y_scores': array}}
        output_dir: Output directory for plots
    """
    plt.figure(figsize=(10, 8))
    
    baselines = []
    for label, data in pr_curve_data.items():
        precision, recall, thresholds = precision_recall_curve(data['y_true'], data['y_scores'])
        plt.plot(recall, precision, linewidth=2, label=label)
        
        # Calculate baseline for this model
        baseline = np.sum(data['y_true']) / len(data['y_true'])
        baselines.append(baseline)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision') 
    plt.title('Precision-Recall Curve Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add baseline (use average baseline if multiple models)
    avg_baseline = np.mean(baselines)
    plt.axhline(y=avg_baseline, color='red', linestyle='--', 
                label=f'Random Baseline ({avg_baseline:.3f})')
    
    plt.legend()
    
    # Save plot
    plot_file = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Precision-recall curve saved: {plot_file}")
    
    plt.close()


def evaluate_model_simple(model_dir):
    """Evaluate model and save metrics to model directory.
    
    Args:
        model_dir: Path to model directory (e.g., graphs/.../embeddings/embeddings_0/models/model_0)
        
    Returns:
        str: Path to the evaluation metrics file
    """
    print(f"Evaluating model: {model_dir}")
    
    # Extract metadata from model directory
    metadata = extract_model_metadata(model_dir)
    
    model_path = metadata["model_file"]
    ground_truth_file = metadata["ground_truth_file"]
    
    print(f"Ground truth: {ground_truth_file}")
    print(f"Embeddings: {metadata['embeddings_file']}")
    
    # Validate model file
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Output will go directly to the model directory
    output_file = os.path.join(model_dir, "evaluation_metrics.json")
    
    print(f"Model: {model_path}")
    print(f"Output: {output_file}")
    
    # Record start time
    start_time = datetime.now()
    
    # Load the trained model
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
    
    print(f"Loaded model from: {model_path}")
    
    # Get embeddings file from metadata (models now know their embeddings)
    embeddings_file = metadata["embeddings_file"]
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    # Load embeddings and ground truth
    embeddings = load_embeddings(embeddings_file)
    # Get original ground truth count before filtering
    _, original_ground_truth_stats = load_ground_truth(ground_truth_file, embeddings=None)
    original_gt_count = original_ground_truth_stats["clean_rows"]
    # Load filtered ground truth
    positive_pairs, ground_truth_stats = load_ground_truth(ground_truth_file, embeddings)
    drug_ids, disease_ids = extract_node_ids_from_positives(positive_pairs)
    
    # Get negative_ratio from model provenance
    negative_ratio = metadata["provenance"]["model_parameters"]["negative_ratio"]
    
    print(f"Using negative_ratio={negative_ratio} from model provenance")
    print(f"Creating comprehensive evaluation on all drug-disease combinations...")
    print(f"Using {len(drug_ids)} drugs and {len(disease_ids)} diseases from ground truth")
    
    # Use shared evaluation logic
    eval_results = evaluate_single_model_core(rf_model, embeddings, positive_pairs, drug_ids, disease_ids, negative_ratio, model_dir, original_gt_count, ground_truth_file)
    
    print(f"Generated {eval_results['all_combinations']:,} total drug-disease combinations")
    print(f"Training pairs to exclude: {eval_results['training_pairs_excluded']:,}")
    print(f"Evaluating on {eval_results['evaluation_combinations']:,} combinations")
    print(f"Evaluation set: {eval_results['evaluation_combinations']:,} combinations ({eval_results['evaluation_positives']:,} known positives)")
    
    # Record end time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Create simple evaluation metrics
    evaluation_metrics = {
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "model_dir": model_dir,
        "embeddings_file": embeddings_file,
        "ground_truth_file": ground_truth_file,
        "evaluation_summary": {
            "all_combinations": eval_results['all_combinations'],
            "training_pairs_excluded": eval_results['training_pairs_excluded'],
            "evaluation_combinations": eval_results['evaluation_combinations'],
            "evaluation_positives": eval_results['evaluation_positives'],
            "evaluation_negatives": eval_results['evaluation_negatives']
        },
        "precision_at_k": eval_results['precision_at_k'],
        "recall_at_k": eval_results['recall_at_k'],
        "total_recall_at_k": eval_results['total_recall_at_k'],
        "total_recall_max": eval_results['total_recall_max'],
        "hits_at_k": eval_results['hits_at_k'],
        "k_values": eval_results['k_values']
    }
    
    # Save metrics to model directory
    with open(output_file, 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    
    print(f"\nEvaluation complete!")
    print(f"Metrics saved to: {output_file}")
    
    # Print summary
    print("\n=== Ranking Metrics Summary ===")
    for k in [1, 5, 10, 20, 50, 100]:
        if k in eval_results['precision_at_k']:
            precision = eval_results['precision_at_k'][k]
            recall = eval_results['recall_at_k'][k]
            hits = eval_results['hits_at_k'][k]
            if not np.isnan(precision):
                print(f"K={k:3d}: Precision={precision:.4f}, Recall={recall:.4f}, Hits={hits:.4f}")
    
    return output_file


def evaluate_multiple_models_with_provenance(model_configs):
    """Evaluate multiple models with comparative analysis and provenance.
    
    Args:
        model_configs: List of dictionaries, each containing:
                      - model_dir: Path to model directory
                      - label: Human-readable label for the model
        
    Returns:
        str: Path to the generated evaluation directory
    """
    # Extract metadata for all models first
    model_metadata = []
    for config in model_configs:
        metadata = extract_model_metadata(config['model_dir'])
        model_metadata.append({
            'metadata': metadata,
            'label': config['label']
        })
    
    # Determine output directory structure
    evaluations_base_dir = "evaluations"
    os.makedirs(evaluations_base_dir, exist_ok=True)
    
    # Get next version number
    version = get_next_evaluation_version(evaluations_base_dir)
    version_dir = os.path.join(evaluations_base_dir, f"evaluation_{version}")
    os.makedirs(version_dir, exist_ok=True)
    
    print(f"Running multi-model evaluation version {version}")
    print(f"Models: {[config['label'] for config in model_configs]}")
    print(f"Output: {version_dir}")
    
    start_time = datetime.now()
    
    # Store results for each model
    all_results = {}
    all_metrics_data = {}
    all_pr_data = {}
    all_provenance = []
    
    # Evaluate each model
    for i, model_data in enumerate(model_metadata):
        metadata = model_data['metadata']
        label = model_data['label']
        
        model_path = metadata['model_file']
        graph_dir = metadata['graph_dir']
        ground_truth_file = metadata['ground_truth_file']
        embeddings_version = metadata['embeddings_version']
        
        print(f"\nEvaluating model {i+1}/{len(model_metadata)}: {label}")
        print(f"  Model: {model_path}")
        print(f"  Graph dir: {graph_dir}")
        print(f"  Ground truth: {ground_truth_file}")
        
        # Get model info for provenance
        model_info = get_model_info(model_path)
        
        # Load the trained model
        with open(model_path, 'rb') as f:
            rf_model = pickle.load(f)
        
        # Determine embeddings file - default to embeddings_0 if not specified
        if embeddings_version:
            embeddings_file = os.path.join(graph_dir, "embeddings", embeddings_version, "embeddings.emb")
        else:
            # Try embeddings_0 first, then fall back to direct embeddings.emb
            embeddings_file = os.path.join(graph_dir, "embeddings", "embeddings_0", "embeddings.emb")
            if not os.path.exists(embeddings_file):
                embeddings_file = os.path.join(graph_dir, "embeddings", "embeddings.emb")
        
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        # Load embeddings and ground truth
        embeddings = load_embeddings(embeddings_file)
        # Get original ground truth count before filtering
        _, original_ground_truth_stats = load_ground_truth(ground_truth_file, embeddings=None)
        original_gt_count = original_ground_truth_stats["clean_rows"]
        # Load filtered ground truth
        positive_pairs, ground_truth_stats = load_ground_truth(ground_truth_file, embeddings)
        drug_ids, disease_ids = extract_node_ids_from_positives(positive_pairs)
        
        # Get negative ratio from model provenance
        negative_ratio = 1
        if "model_provenance" in model_info and "model_parameters" in model_info["model_provenance"]:
            negative_ratio = model_info["model_provenance"]["model_parameters"].get("negative_ratio", 1)
        
        print(f"  Using negative_ratio={negative_ratio} from model provenance")
        print(f"  Using {len(drug_ids)} drugs and {len(disease_ids)} diseases from ground truth")
        
        # Use shared evaluation logic
        model_dir = os.path.dirname(model_path)
        eval_results = evaluate_single_model_core(rf_model, embeddings, positive_pairs, drug_ids, disease_ids, negative_ratio, model_dir, original_gt_count, ground_truth_file)
        
        print(f"  Total possible combinations: {eval_results['all_combinations']:,}")
        print(f"  Training pairs to exclude: {eval_results['training_pairs_excluded']:,}")
        print(f"  Evaluating on {eval_results['evaluation_combinations']:,} combinations")
        print(f"  Test predictions: {len(eval_results['y_scores'])} ({eval_results['evaluation_positives']} positive, {eval_results['evaluation_negatives']} negative)")
        
        # Extract results from shared function
        precision_at_k = eval_results['precision_at_k']
        recall_at_k = eval_results['recall_at_k']
        hits_at_k = eval_results['hits_at_k']
        k_values = eval_results['k_values']
        y_true = eval_results['y_true']
        y_scores = eval_results['y_scores']
        test_pairs = eval_results['test_pairs']
        
        # Store results for this model
        all_results[label] = {
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k,
            'hits_at_k': hits_at_k,
            'k_values': k_values,
            'total_evaluation_combinations': eval_results['evaluation_combinations'],
            'evaluation_positives': eval_results['evaluation_positives'],
            'evaluation_negatives': eval_results['evaluation_negatives'],
            'negative_ratio_used': negative_ratio,
            'total_drugs': len(drug_ids),
            'total_diseases': len(disease_ids),
            'training_pairs_excluded': eval_results['training_pairs_excluded']
        }
        
        # Store data for plots
        all_metrics_data[label] = {
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k,
            'total_recall_at_k': eval_results['total_recall_at_k'],
            'total_recall_max': eval_results['total_recall_max'],
            'hits_at_k': hits_at_k,
            'k_values': k_values
        }
        
        all_pr_data[label] = {
            'y_true': y_true,
            'y_scores': y_scores
        }
        
        # Store provenance for this model
        model_provenance = {
            "model_label": label,
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
                "evaluation_scope": "comprehensive_drug_disease_combinations",
                "training_pairs_excluded": True,
                "total_drugs": len(drug_ids),
                "total_diseases": len(disease_ids),
                "total_possible_combinations": eval_results['all_combinations'],
                "training_pairs_excluded_count": eval_results['training_pairs_excluded']
            },
            "test_data_info": {
                "total_evaluation_combinations": eval_results['evaluation_combinations'],
                "evaluation_positives": eval_results['evaluation_positives'],
                "evaluation_negatives": eval_results['evaluation_negatives']
            },
            "ranking_metrics": {
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "hits_at_k": hits_at_k
            }
        }
        all_provenance.append(model_provenance)
    
    # Create comparative plots
    print(f"\nCreating comparative plots...")
    plot_ranking_metrics(all_metrics_data, version_dir)
    create_precision_recall_curve(all_pr_data, version_dir)
    
    # Record end time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Create comprehensive provenance metadata for the entire evaluation
    multi_model_provenance = {
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "script": "evaluate_model.py",
        "evaluation_type": "multi_model_comparison",
        "version": f"evaluation_{version}",
        "num_models": len(model_configs),
        "model_labels": [config['label'] for config in model_configs],
        "individual_model_provenance": all_provenance,
        "description": f"Multi-model evaluation version {version} comparing {len(model_configs)} models"
    }
    
    # Save provenance file
    provenance_file = os.path.join(version_dir, "provenance.json")
    with open(provenance_file, 'w') as f:
        json.dump(multi_model_provenance, f, indent=2)
    
    # Save comparative results
    comparative_results = {
        'evaluation_type': 'multi_model_comparison',
        'models': all_results,
        'model_count': len(model_configs),
        'duration_seconds': duration
    }
    
    results_file = os.path.join(version_dir, "comparative_results.json")
    with open(results_file, 'w') as f:
        json.dump(comparative_results, f, indent=2)
    
    print(f"Provenance saved: {provenance_file}")
    print(f"Comparative results saved: {results_file}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Print comparative summary
    print("\n=== Multi-Model Test Set Ranking Metrics Comparison ===")
    print(f"{'Model':<20} {'P@10':<8} {'R@10':<8} {'H@10':<8} {'P@50':<8} {'R@50':<8} {'H@50':<8}")
    print("-" * 80)
    
    for label, results in all_results.items():
        p10 = results['precision_at_k'].get(10, np.nan)
        r10 = results['recall_at_k'].get(10, np.nan)
        h10 = results['hits_at_k'].get(10, np.nan)
        p50 = results['precision_at_k'].get(50, np.nan)
        r50 = results['recall_at_k'].get(50, np.nan)
        h50 = results['hits_at_k'].get(50, np.nan)
        
        print(f"{label:<20} {p10:<8.4f} {r10:<8.4f} {h10:<8.4f} {p50:<8.4f} {r50:<8.4f} {h50:<8.4f}")
    
    print(f"\nMulti-model evaluation complete!")
    print(f"Output directory: {version_dir}")
    
    return version_dir


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Evaluate single model and save metrics to model directory")
    
    # Single model evaluation only
    parser.add_argument("--model-dir", required=True,
                       help="Path to model directory (e.g., graphs/.../embeddings/embeddings_0/models/model_0)")
    
    args = parser.parse_args()
    
    # Single model evaluation - save to model directory
    output_file = evaluate_model_simple(args.model_dir)
    
    print(f"\nEvaluation complete! Metrics saved to: {output_file}")


if __name__ == "__main__":
    main()