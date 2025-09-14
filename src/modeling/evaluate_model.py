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
    """Create plots for ranking metrics with support for multiple models.
    
    Args:
        metrics_data: Dictionary with model labels as keys and their metrics as values
                     Format: {label: {precision_at_k, recall_at_k, hits_at_k, k_values}}
        output_dir: Output directory for plots
    """
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Drug-Disease Prediction Ranking Metrics', fontsize=16)
    
    # Get all k values (union of all models)
    all_k_values = set()
    for model_data in metrics_data.values():
        all_k_values.update(model_data['k_values'])
    all_k_values = sorted(all_k_values)
    
    # Plot 1: Precision@K
    for i, (label, model_data) in enumerate(metrics_data.items()):
        k_vals = [k for k in model_data['k_values'] if not np.isnan(model_data['precision_at_k'][k])]
        prec_vals = [model_data['precision_at_k'][k] for k in k_vals]
        axes[0, 0].plot(k_vals, prec_vals, 'o-', linewidth=2, markersize=6, label=label)
    
    axes[0, 0].set_xlabel('K (Top Predictions)')
    axes[0, 0].set_ylabel('Precision@K')
    axes[0, 0].set_title('Precision@K: Accuracy of Top K Predictions')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    if len(metrics_data) > 1:
        axes[0, 0].legend()
    
    # Plot 2: Recall@K
    for i, (label, model_data) in enumerate(metrics_data.items()):
        k_vals = [k for k in model_data['k_values'] if not np.isnan(model_data['recall_at_k'][k])]
        recall_vals = [model_data['recall_at_k'][k] for k in k_vals]
        axes[0, 1].plot(k_vals, recall_vals, 'o-', linewidth=2, markersize=6, label=label)
    
    axes[0, 1].set_xlabel('K (Top Predictions)')
    axes[0, 1].set_ylabel('Recall@K')
    axes[0, 1].set_title('Recall@K: Coverage of True Positives')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    if len(metrics_data) > 1:
        axes[0, 1].legend()
    
    # Plot 3: Hits@K
    for i, (label, model_data) in enumerate(metrics_data.items()):
        k_vals = [k for k in model_data['k_values'] if not np.isnan(model_data['hits_at_k'][k])]
        hits_vals = [model_data['hits_at_k'][k] for k in k_vals]
        axes[1, 0].plot(k_vals, hits_vals, 'o-', linewidth=2, markersize=6, label=label)
    
    axes[1, 0].set_xlabel('K (Top Predictions)')
    axes[1, 0].set_ylabel('Hits@K')
    axes[1, 0].set_title('Hits@K: At Least One Hit in Top K')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    if len(metrics_data) > 1:
        axes[1, 0].legend()
    
    # Plot 4: All metrics for first model (or single model if only one)
    first_model_data = next(iter(metrics_data.values()))
    k_vals = [k for k in first_model_data['k_values'] if not np.isnan(first_model_data['precision_at_k'][k])]
    prec_vals = [first_model_data['precision_at_k'][k] for k in k_vals]
    recall_vals = [first_model_data['recall_at_k'][k] for k in k_vals]
    hits_vals = [first_model_data['hits_at_k'][k] for k in k_vals]
    
    axes[1, 1].plot(k_vals, prec_vals, 'o-', label='Precision@K', linewidth=2, markersize=4)
    axes[1, 1].plot(k_vals, recall_vals, 'o-', label='Recall@K', linewidth=2, markersize=4)
    axes[1, 1].plot(k_vals, hits_vals, 'o-', label='Hits@K', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('K (Top Predictions)')
    axes[1, 1].set_ylabel('Metric Value')
    if len(metrics_data) == 1:
        axes[1, 1].set_title('All Ranking Metrics')
    else:
        first_model_label = next(iter(metrics_data.keys()))
        axes[1, 1].set_title(f'All Ranking Metrics - {first_model_label}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
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
        # Try embeddings_0 first, then fall back to direct embeddings.emb
        embeddings_file = os.path.join(graph_dir, "embeddings", "embeddings_0", "embeddings.emb")
        if not os.path.exists(embeddings_file):
            embeddings_file = os.path.join(graph_dir, "embeddings", "embeddings.emb")
    
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    # Load embeddings and ground truth
    embeddings = load_embeddings(embeddings_file)
    positive_pairs, ground_truth_stats = load_ground_truth(ground_truth_file, embeddings)
    drug_ids, disease_ids = extract_node_ids_from_positives(positive_pairs)
    
    # Need to determine negative_ratio from model provenance to recreate training set
    negative_ratio = 1  # default
    if "model_provenance" in model_info and "model_parameters" in model_info["model_provenance"]:
        negative_ratio = model_info["model_provenance"]["model_parameters"].get("negative_ratio", 1)
    
    print(f"Using negative_ratio={negative_ratio} from model provenance")
    print(f"Creating comprehensive evaluation on all drug-disease combinations...")
    
    # Use the unique drugs and diseases from ground truth (already extracted above)
    print(f"Using {len(drug_ids)} drugs and {len(disease_ids)} diseases from ground truth")
    
    # Generate ALL possible drug-disease combinations from ground truth
    all_combinations = [(drug, disease) for drug in drug_ids for disease in disease_ids]
    print(f"Generated {len(all_combinations):,} total drug-disease combinations")
    
    # Determine training pairs to exclude from evaluation
    # Recreate the training data to identify what was seen during training
    pos_features_train, pos_labels_train = create_feature_vectors(embeddings, positive_pairs)
    pos_targets_train = np.ones(len(pos_features_train))
    negative_pairs_train = generate_negative_samples(positive_pairs, drug_ids, disease_ids, negative_ratio)
    neg_features_train, neg_labels_train = create_feature_vectors(embeddings, negative_pairs_train)
    neg_targets_train = np.zeros(len(neg_features_train))
    
    X_train_full = np.vstack([pos_features_train, neg_features_train])
    y_train_full = np.hstack([pos_targets_train, neg_targets_train])
    pair_labels_full = pos_labels_train + neg_labels_train
    
    # Get training/test split using the same split as during training
    X_train, X_test_unused, y_train, y_test_unused, indices_train, indices_test = train_test_split(
        X_train_full, y_train_full, range(len(pair_labels_full)), test_size=0.2, random_state=42, stratify=y_train_full
    )
    training_pairs = set([pair_labels_full[i] for i in indices_train])
    test_pairs_from_split = set([pair_labels_full[i] for i in indices_test])
    
    print(f"Training pairs to exclude: {len(training_pairs):,}")
    print(f"Test pairs from 20% split: {len(test_pairs_from_split):,}")
    
    # Remove only training pairs from evaluation, but include test pairs + unseen combinations
    evaluation_combinations = [pair for pair in all_combinations if pair not in training_pairs]
    
    # Make sure we include the test split pairs (they should already be included but let's be explicit)
    evaluation_combinations_set = set(evaluation_combinations)
    missing_test_pairs = test_pairs_from_split - evaluation_combinations_set
    if missing_test_pairs:
        evaluation_combinations.extend(list(missing_test_pairs))
        print(f"Added {len(missing_test_pairs)} missing test pairs to evaluation")
    
    print(f"Evaluating on {len(evaluation_combinations):,} combinations (includes 20% test split + unseen combinations)")
    
    # Create features for evaluation combinations
    eval_features, eval_pairs = create_feature_vectors(embeddings, evaluation_combinations)
    
    # Label evaluation combinations (known positives = 1, rest = 0)
    y_true = np.array([1 if pair in positive_pairs else 0 for pair in eval_pairs])
    test_pairs = eval_pairs  # For hits@k calculation
    
    print(f"Evaluation set: {len(eval_features):,} combinations ({np.sum(y_true):,} known positives)")
    
    # Generate predictions for ALL evaluation combinations  
    y_scores = rf_model.predict_proba(eval_features)[:, 1]  # Probability of positive class
    
    print(f"Test predictions: {len(y_scores)}")
    print(f"Test positives: {np.sum(y_true)}")
    print(f"Test negatives: {len(y_true) - np.sum(y_true)}")
    
    # Define K values to evaluate
    k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    k_values = [k for k in k_values if k <= len(y_scores)]  # Only valid K values
    
    # Calculate ranking metrics (global precision/recall, per-disease hits)
    precision_at_k = calculate_precision_at_k(y_true, y_scores, k_values)
    recall_at_k = calculate_recall_at_k(y_true, y_scores, k_values)
    hits_at_k = calculate_hits_at_k_per_disease(test_pairs, y_true, y_scores, k_values)
    
    # Create plots for single model
    single_model_metrics = {
        os.path.basename(model_path): {
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k, 
            'hits_at_k': hits_at_k,
            'k_values': k_values
        }
    }
    plot_ranking_metrics(single_model_metrics, version_dir)
    
    single_model_pr_data = {
        os.path.basename(model_path): {
            'y_true': y_true,
            'y_scores': y_scores
        }
    }
    create_precision_recall_curve(single_model_pr_data, version_dir)
    
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
            "total_evaluation_combinations": len(all_combinations),
            "training_pairs_excluded": len(training_pairs), 
            "evaluation_combinations": len(evaluation_combinations),
            "evaluation_positives": int(np.sum(y_true)),
            "evaluation_negatives": int(len(y_true) - np.sum(y_true)),
            "total_drugs": len(drug_ids),
            "total_diseases": len(disease_ids)
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
        'total_evaluation_combinations': len(all_combinations),
        'training_pairs_excluded': len(training_pairs),
        'evaluation_combinations': len(evaluation_combinations), 
        'evaluation_positives': int(np.sum(y_true)),
        'evaluation_negatives': int(len(y_true) - np.sum(y_true)),
        'total_drugs': len(drug_ids),
        'total_diseases': len(disease_ids),
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


def evaluate_multiple_models_with_provenance(model_configs, ground_truth_file):
    """Evaluate multiple models with comparative analysis and provenance.
    
    Args:
        model_configs: List of dictionaries, each containing:
                      - model_path: Path to model file
                      - graph_dir: Path to graph directory  
                      - label: Human-readable label for the model
                      - model_version: Optional model version
                      - embeddings_version: Optional embeddings version
        ground_truth_file: Path to ground truth CSV file
        
    Returns:
        str: Path to the generated evaluation directory
    """
    # Validate all model files first
    for config in model_configs:
        if not os.path.exists(config['model_path']):
            raise FileNotFoundError(f"Model file not found: {config['model_path']}")
    
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
    for i, config in enumerate(model_configs):
        model_path = config['model_path']
        graph_dir = config['graph_dir']
        label = config['label']
        model_version = config.get('model_version')
        embeddings_version = config.get('embeddings_version')
        
        print(f"\nEvaluating model {i+1}/{len(model_configs)}: {label}")
        print(f"  Model: {model_path}")
        
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
        positive_pairs, ground_truth_stats = load_ground_truth(ground_truth_file, embeddings)
        drug_ids, disease_ids = extract_node_ids_from_positives(positive_pairs)
        
        # Get negative ratio from model provenance
        negative_ratio = 1
        if "model_provenance" in model_info and "model_parameters" in model_info["model_provenance"]:
            negative_ratio = model_info["model_provenance"]["model_parameters"].get("negative_ratio", 1)
        
        print(f"  Using negative_ratio={negative_ratio} from model provenance")
        
        # Recreate the training data to identify what was seen during training
        pos_features_train, pos_labels_train = create_feature_vectors(embeddings, positive_pairs)
        pos_targets_train = np.ones(len(pos_features_train))
        training_negative_pairs = generate_negative_samples(positive_pairs, drug_ids, disease_ids, negative_ratio)
        neg_features_train, neg_labels_train = create_feature_vectors(embeddings, training_negative_pairs)
        neg_targets_train = np.zeros(len(neg_features_train))
        
        X_train_full = np.vstack([pos_features_train, neg_features_train])
        y_train_full = np.hstack([pos_targets_train, neg_targets_train])
        pair_labels_full = pos_labels_train + neg_labels_train
        
        # Get training/test split using the same split as during training
        X_train, X_test_unused, y_train, y_test_unused, indices_train, indices_test = train_test_split(
            X_train_full, y_train_full, range(len(pair_labels_full)), test_size=0.2, random_state=42, stratify=y_train_full
        )
        training_pairs = set([pair_labels_full[i] for i in indices_train])
        test_pairs_from_split = set([pair_labels_full[i] for i in indices_test])
        
        # Use the unique drugs and diseases from ground truth (already extracted above)
        print(f"  Using {len(drug_ids)} drugs and {len(disease_ids)} diseases from ground truth")
        
        # Generate ALL possible drug-disease combinations from ground truth
        all_combinations = [(drug, disease) for drug in drug_ids for disease in disease_ids]
        print(f"  Total possible combinations: {len(all_combinations):,}")
        
        print(f"  Training pairs to exclude: {len(training_pairs):,}")
        print(f"  Test pairs from 20% split: {len(test_pairs_from_split):,}")
        
        # Remove only training pairs from evaluation, but include test pairs + unseen combinations
        evaluation_combinations = [pair for pair in all_combinations if pair not in training_pairs]
        
        # Make sure we include the test split pairs (they should already be included but let's be explicit)
        evaluation_combinations_set = set(evaluation_combinations)
        missing_test_pairs = test_pairs_from_split - evaluation_combinations_set
        if missing_test_pairs:
            evaluation_combinations.extend(list(missing_test_pairs))
            print(f"  Added {len(missing_test_pairs)} missing test pairs to evaluation")
        
        print(f"  Evaluating on {len(evaluation_combinations):,} combinations (includes 20% test split + unseen combinations)")
        
        # Generate features for evaluation combinations
        eval_features, eval_labels = create_feature_vectors(embeddings, evaluation_combinations)
        
        # Generate predictions for ALL evaluation combinations
        y_scores = rf_model.predict_proba(eval_features)[:, 1]
        
        # Label evaluation combinations (known positives = 1, rest = 0)
        positive_pairs_set = set(positive_pairs)
        y_true = np.array([1 if pair in positive_pairs_set else 0 for pair in evaluation_combinations])
        
        # Get test set pairs for per-disease metrics
        test_pairs = evaluation_combinations
        
        print(f"  Test predictions: {len(y_scores)} ({np.sum(y_true)} positive, {len(y_true) - np.sum(y_true)} negative)")
        
        # Define K values to evaluate
        k_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
        k_values = [k for k in k_values if k <= len(y_scores)]
        
        # Calculate ranking metrics (global precision/recall, per-disease hits)
        precision_at_k = calculate_precision_at_k(y_true, y_scores, k_values)
        recall_at_k = calculate_recall_at_k(y_true, y_scores, k_values)
        hits_at_k = calculate_hits_at_k_per_disease(test_pairs, y_true, y_scores, k_values)
        
        # Store results for this model
        all_results[label] = {
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k,
            'hits_at_k': hits_at_k,
            'k_values': k_values,
            'total_evaluation_combinations': len(y_scores),
            'evaluation_positives': int(np.sum(y_true)),
            'evaluation_negatives': int(len(y_true) - np.sum(y_true)),
            'negative_ratio_used': negative_ratio,
            'total_drugs': len(drug_ids),
            'total_diseases': len(disease_ids),
            'training_pairs_excluded': len(training_pairs)
        }
        
        # Store data for plots
        all_metrics_data[label] = {
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k,
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
                "total_possible_combinations": len(all_combinations),
                "training_pairs_excluded_count": len(training_pairs)
            },
            "test_data_info": {
                "total_evaluation_combinations": len(y_scores),
                "evaluation_positives": int(np.sum(y_true)),
                "evaluation_negatives": int(len(y_true) - np.sum(y_true))
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
    parser = argparse.ArgumentParser(description="Evaluate models with versioning and provenance")
    
    # Single model evaluation (backward compatibility)
    parser.add_argument("--model-path",
                       help="Path to specific model file (e.g., graphs/robokop_base/CCDD/models/model_2/rf_model.pkl)")
    parser.add_argument("--graph-dir",
                       help="Path to graph directory (e.g., graphs/robokop_base/CCDD)")
    parser.add_argument("--model-version",
                       help="Specific model version (e.g., model_2)")
    parser.add_argument("--embeddings-version", 
                       help="Specific embeddings version to use (e.g., embeddings_2)")
    
    # Multi-model evaluation
    parser.add_argument("--multi-model-config", 
                       help="JSON file containing multiple model configurations")
    parser.add_argument("--model-paths", nargs="+",
                       help="Multiple model paths for comparison")
    parser.add_argument("--model-labels", nargs="+",
                       help="Labels for the models (when using --model-paths)")
    parser.add_argument("--graph-dirs", nargs="+",
                       help="Graph directories for each model (when using --model-paths)")
    
    # Common arguments
    parser.add_argument("--ground-truth", required=True,
                       help="Path to ground truth CSV file")
    
    args = parser.parse_args()
    
    # Multi-model evaluation via config file
    if args.multi_model_config:
        with open(args.multi_model_config, 'r') as f:
            model_configs = json.load(f)
        
        version_dir = evaluate_multiple_models_with_provenance(
            model_configs=model_configs,
            ground_truth_file=args.ground_truth
        )
    
    # Multi-model evaluation via command line args
    elif args.model_paths:
        if not args.model_labels or len(args.model_labels) != len(args.model_paths):
            raise ValueError("Must provide same number of model labels as model paths")
        if not args.graph_dirs or len(args.graph_dirs) != len(args.model_paths):
            raise ValueError("Must provide same number of graph directories as model paths")
        
        model_configs = []
        for i, model_path in enumerate(args.model_paths):
            model_configs.append({
                'model_path': model_path,
                'graph_dir': args.graph_dirs[i],
                'label': args.model_labels[i]
            })
        
        version_dir = evaluate_multiple_models_with_provenance(
            model_configs=model_configs,
            ground_truth_file=args.ground_truth
        )
    
    # Single model evaluation (backward compatibility)
    elif args.model_path and args.graph_dir:
        version_dir = evaluate_model_with_provenance(
            model_path=args.model_path,
            graph_dir=args.graph_dir,
            ground_truth_file=args.ground_truth,
            model_version=args.model_version,
            embeddings_version=args.embeddings_version
        )
    
    else:
        parser.error("Must provide either --model-path and --graph-dir for single model, "
                    "or --multi-model-config for config file, "
                    "or --model-paths, --model-labels, and --graph-dirs for multi-model evaluation")
    
    print(f"\nModel evaluation complete!")
    print(f"Output directory: {version_dir}")


if __name__ == "__main__":
    main()