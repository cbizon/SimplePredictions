#!/usr/bin/env python3
"""Train Random Forest model for Drug-Disease link prediction.

This script loads node embeddings and ground truth data to train and evaluate
a Random Forest classifier for predicting Drug-treats-Disease relationships.
"""
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix
)
import pickle
import json
from collections import defaultdict


def load_embeddings(embeddings_file):
    """Load node embeddings from file.
    
    Args:
        embeddings_file: Path to embeddings.emb file
        
    Returns:
        dict: Mapping from node_id to embedding vector
    """
    embeddings = {}
    with open(embeddings_file, 'r') as f:
        # Skip first line (header with num_nodes, embedding_dim)
        header = f.readline().strip().split()
        num_nodes, embedding_dim = int(header[0]), int(header[1])
        
        for line in f:
            parts = line.strip().split()
            node_id = parts[0]
            embedding = np.array([float(x) for x in parts[1:]])
            embeddings[node_id] = embedding
    
    print(f"Loaded {len(embeddings)} embeddings with dimension {embedding_dim}")
    return embeddings


def load_ground_truth(ground_truth_file, embeddings=None):
    """Load ground truth Drug-Disease pairs from Indication list.
    
    Args:
        ground_truth_file: Path to comma-delimited file with drug and disease columns
        embeddings: Dict of embeddings to filter against (optional)
        
    Returns:
        set: Set of (drug_id, disease_id) tuples representing positive examples
    """
    df = pd.read_csv(ground_truth_file, sep=',')
    
    print(f"Available columns: {list(df.columns)}")
    print(f"File shape: {df.shape}")
    
    # Use the specific column names from the indication list file
    drug_col = 'final normalized drug id'
    disease_col = 'final normalized disease id'
    
    if drug_col not in df.columns or disease_col not in df.columns:
        raise ValueError(f"Expected columns '{drug_col}' and '{disease_col}' in {ground_truth_file}. "
                        f"Available columns: {list(df.columns)}")
    
    print(f"Using drug column: '{drug_col}' and disease column: '{disease_col}'")
    
    # Remove rows with missing values
    df_clean = df[[drug_col, disease_col]].dropna()
    
    # Filter to only include pairs where both drug and disease have embeddings
    if embeddings is not None:
        print(f"Filtering ground truth to only include nodes with embeddings...")
        initial_count = len(df_clean)
        df_clean = df_clean[
            df_clean[drug_col].isin(embeddings) & 
            df_clean[disease_col].isin(embeddings)
        ]
        print(f"Filtered from {initial_count} to {len(df_clean)} pairs with embeddings")
    
    positive_pairs = set(zip(df_clean[drug_col], df_clean[disease_col]))
    print(f"Final positive pairs: {len(positive_pairs)}")
    
    return positive_pairs


def create_feature_vectors(embeddings, drug_disease_pairs):
    """Create feature vectors for specific Drug-Disease pairs only.
    
    Args:
        embeddings: Dict mapping node_id to embedding vector
        drug_disease_pairs: List/set of (drug_id, disease_id) tuples
        
    Returns:
        tuple: (feature_matrix, pair_labels) where feature_matrix is concatenated embeddings
    """
    features = []
    pair_labels = []
    
    for drug_id, disease_id in drug_disease_pairs:
        if drug_id in embeddings and disease_id in embeddings:
            # Concatenate drug and disease embeddings
            drug_emb = embeddings[drug_id]
            disease_emb = embeddings[disease_id]
            combined_features = np.concatenate([drug_emb, disease_emb])
            
            features.append(combined_features)
            pair_labels.append((drug_id, disease_id))
    
    return np.array(features), pair_labels


def generate_negative_samples(positive_pairs, all_drug_ids, all_disease_ids, ratio=1):
    """Generate negative samples for training.
    
    Args:
        positive_pairs: Set of positive (drug_id, disease_id) pairs
        all_drug_ids: List of all available drug IDs
        all_disease_ids: List of all available disease IDs  
        ratio: Ratio of negative to positive samples
        
    Returns:
        set: Set of negative (drug_id, disease_id) pairs
    """
    negative_pairs = set()
    target_size = len(positive_pairs) * ratio
    
    np.random.seed(42)  # For reproducibility
    
    while len(negative_pairs) < target_size:
        drug_id = np.random.choice(all_drug_ids)
        disease_id = np.random.choice(all_disease_ids)
        
        pair = (drug_id, disease_id)
        if pair not in positive_pairs:
            negative_pairs.add(pair)
    
    print(f"Generated {len(negative_pairs)} negative samples")
    return negative_pairs


def extract_node_ids_from_positives(positive_pairs):
    """Extract drug and disease IDs from positive examples only.
    
    Args:
        positive_pairs: Set of (drug_id, disease_id) tuples
        
    Returns:
        tuple: (drug_ids, disease_ids) lists
    """
    drug_ids = list(set(pair[0] for pair in positive_pairs))
    disease_ids = list(set(pair[1] for pair in positive_pairs))
    
    print(f"Found {len(drug_ids)} unique drugs and {len(disease_ids)} unique diseases from positive examples")
    return drug_ids, disease_ids


def train_and_evaluate(graph_dir, ground_truth_file, output_dir, negative_ratio=1):
    """Main training and evaluation pipeline.
    
    Args:
        graph_dir: Path to graph directory (e.g., graphs/CCDD)
        ground_truth_file: Path to ground truth file
        output_dir: Directory to save model and results
        negative_ratio: Ratio of negative to positive samples
    """
    # Load embeddings
    embeddings_file = os.path.join(graph_dir, "embeddings", "embeddings.emb")
    embeddings = load_embeddings(embeddings_file)
    
    # Load ground truth - filter to only nodes with embeddings
    positive_pairs = load_ground_truth(ground_truth_file, embeddings)
    
    # Extract drug and disease IDs from filtered positive examples
    drug_ids, disease_ids = extract_node_ids_from_positives(positive_pairs)
    
    # Create feature vectors for positive samples (guaranteed to work now)
    pos_features, pos_labels = create_feature_vectors(embeddings, positive_pairs)
    pos_targets = np.ones(len(pos_features))
    
    print(f"Created {len(pos_features)} positive feature vectors")
    
    # Calculate target negative count
    target_negative_count = len(pos_features) * negative_ratio
    
    # Generate exact number of negative samples (no need for oversampling)
    negative_pairs = generate_negative_samples(positive_pairs, drug_ids, disease_ids, negative_ratio)
    
    # Create feature vectors for negative samples (guaranteed to work)
    neg_features, neg_labels = create_feature_vectors(embeddings, negative_pairs)
    neg_targets = np.zeros(len(neg_features))
    
    print(f"Created {len(neg_features)} negative feature vectors (target was {target_negative_count})")
    
    # Combine positive and negative samples
    X = np.vstack([pos_features, neg_features])
    y = np.hstack([pos_targets, neg_targets])
    pair_labels = pos_labels + neg_labels
    
    print(f"Total samples: {len(X)} (positive: {len(pos_features)}, negative: {len(neg_features)})")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'average_precision': average_precision_score(y_test, y_pred_proba)
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_file = os.path.join(output_dir, "rf_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(rf, f)
    
    # Save results
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed report
    report_file = os.path.join(output_dir, "classification_report.txt")
    with open(report_file, 'w') as f:
        f.write("Classification Report\n")
        f.write("===================\n\n")
        f.write(f"Dataset: {len(X)} total samples\n")
        f.write(f"Positive samples: {len(pos_features)}\n")
        f.write(f"Negative samples: {len(neg_features)}\n\n")
        
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nDetailed Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
    
    # Print results
    print("\nResults:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nModel saved to: {model_file}")
    print(f"Results saved to: {results_file}")
    print(f"Detailed report saved to: {report_file}")


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Train Random Forest for Drug-Disease prediction")
    parser.add_argument("--graph-dir", required=True,
                       help="Path to graph directory containing embeddings (e.g., graphs/CCDD)")
    parser.add_argument("--ground-truth", required=True,
                       help="Path to ground truth file with Drug_ID and Disease_ID columns")
    parser.add_argument("--output-dir", 
                       help="Output directory (default: {graph_dir}/models)")
    parser.add_argument("--negative-ratio", type=int, default=1,
                       help="Ratio of negative to positive samples (default: 1)")
    
    args = parser.parse_args()
    
    # Set default output directory
    if not args.output_dir:
        args.output_dir = os.path.join(args.graph_dir, "models")
    
    train_and_evaluate(args.graph_dir, args.ground_truth, args.output_dir, args.negative_ratio)


if __name__ == "__main__":
    main()