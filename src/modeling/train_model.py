#!/usr/bin/env python3
"""Train models with automatic versioning and provenance tracking.

This script trains Random Forest models using embeddings and automatically manages
versioned output directories with full provenance metadata.
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
from datetime import datetime
from collections import defaultdict


def get_next_model_version(models_dir):
    """Find the next available model version number.
    
    Args:
        models_dir: Base models directory
        
    Returns:
        int: Next available version number
    """
    if not os.path.exists(models_dir):
        return 0
    
    existing_versions = []
    for item in os.listdir(models_dir):
        if item.startswith("model_") and os.path.isdir(os.path.join(models_dir, item)):
            try:
                version_num = int(item.split("_")[1])
                existing_versions.append(version_num)
            except (IndexError, ValueError):
                continue
    
    return max(existing_versions) + 1 if existing_versions else 0


def get_embedding_info(embeddings_file):
    """Extract embedding metadata from file.
    
    Args:
        embeddings_file: Path to embeddings file
        
    Returns:
        dict: Embedding info including dimensions and count
    """
    if not os.path.exists(embeddings_file):
        return {"error": f"Embeddings file not found: {embeddings_file}"}
    
    with open(embeddings_file, 'r') as f:
        header = f.readline().strip().split()
        num_nodes, embedding_dim = int(header[0]), int(header[1])
    
    # Try to get provenance info if available
    embedding_dir = os.path.dirname(embeddings_file)
    provenance_file = os.path.join(embedding_dir, "provenance.json")
    
    embedding_info = {
        "embeddings_file": embeddings_file,
        "num_nodes": num_nodes,
        "embedding_dim": embedding_dim
    }
    
    if os.path.exists(provenance_file):
        with open(provenance_file, 'r') as f:
            embedding_provenance = json.load(f)
            embedding_info["embedding_provenance"] = embedding_provenance
    
    return embedding_info


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
        tuple: (positive_pairs, ground_truth_stats)
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
    initial_count = len(df_clean)
    
    # Filter to only include pairs where both drug and disease have embeddings
    if embeddings is not None:
        print(f"Filtering ground truth to only include nodes with embeddings...")
        df_clean = df_clean[
            df_clean[drug_col].isin(embeddings) & 
            df_clean[disease_col].isin(embeddings)
        ]
        print(f"Filtered from {initial_count} to {len(df_clean)} pairs with embeddings")
    
    positive_pairs = set(zip(df_clean[drug_col], df_clean[disease_col]))
    
    ground_truth_stats = {
        "ground_truth_file": ground_truth_file,
        "total_rows": len(df),
        "clean_rows": initial_count,
        "final_positive_pairs": len(positive_pairs),
        "unique_drugs": len(set(pair[0] for pair in positive_pairs)),
        "unique_diseases": len(set(pair[1] for pair in positive_pairs))
    }
    
    print(f"Final positive pairs: {len(positive_pairs)}")
    
    return positive_pairs, ground_truth_stats


def create_feature_vectors(embeddings, drug_disease_pairs, pad_missing=False):
    """Create feature vectors for Drug-Disease pairs.
    
    Args:
        embeddings: Dict mapping node_id to embedding vector
        drug_disease_pairs: List/set of (drug_id, disease_id) tuples
        pad_missing: If True, include all pairs using zero vectors for missing embeddings.
                    If False, exclude pairs with missing embeddings (default behavior).
        
    Returns:
        tuple: (feature_matrix, pair_labels) where feature_matrix is concatenated embeddings
    """
    if not embeddings:
        raise ValueError("Empty embeddings dict provided")
    
    features = []
    pair_labels = []
    
    # Get embedding dimension and create zero vector for missing embeddings
    embedding_dim = len(next(iter(embeddings.values())))
    zero_embedding = np.zeros(embedding_dim)
    
    for drug_id, disease_id in drug_disease_pairs:
        if pad_missing:
            # Include all pairs, use zero vectors for missing embeddings
            drug_emb = embeddings.get(drug_id, zero_embedding)
            disease_emb = embeddings.get(disease_id, zero_embedding)
            combined_features = np.concatenate([drug_emb, disease_emb])
            
            features.append(combined_features)
            pair_labels.append((drug_id, disease_id))
        else:
            # Original behavior: only include pairs where both embeddings exist
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


def train_model(graph_dir,
                               ground_truth_file,
                               embeddings_version=None,
                               negative_ratio=1,
                               n_estimators=100,
                               max_depth=10,
                               random_state=42):
    """Train model with automatic versioning and provenance.
    
    Args:
        graph_dir: Path to graph directory (e.g., graphs/robokop_base/CCDD)
        ground_truth_file: Path to ground truth CSV file
        embeddings_version: Specific embeddings version to use (e.g., "embeddings_2")
        negative_ratio: Ratio of negative to positive samples
        n_estimators: Number of trees in random forest
        max_depth: Maximum depth of trees
        random_state: Random state for reproducibility
        
    Returns:
        str: Path to the generated model directory
    """
    # Determine embeddings file
    if embeddings_version:
        embeddings_file = os.path.join(graph_dir, "embeddings", embeddings_version, "embeddings.emb")
    else:
        # Use the default embeddings.emb file
        embeddings_file = os.path.join(graph_dir, "embeddings", "embeddings.emb")
    
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    # Determine output directory structure
    base_dir = graph_dir  # e.g., graphs/robokop_base/CCDD
    models_dir = os.path.join(base_dir, "models")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Get next version number
    version = get_next_model_version(models_dir)
    version_dir = os.path.join(models_dir, f"model_{version}")
    os.makedirs(version_dir, exist_ok=True)
    
    print(f"Training model version {version}")
    print(f"Input embeddings: {embeddings_file}")
    print(f"Output: {version_dir}")
    
    # Record start time
    start_time = datetime.now()
    
    # Get embedding info for provenance
    embedding_info = get_embedding_info(embeddings_file)
    
    # Load embeddings
    embeddings = load_embeddings(embeddings_file)
    
    # Load ground truth - filter to only nodes with embeddings
    positive_pairs, ground_truth_stats = load_ground_truth(ground_truth_file, embeddings)
    
    # Extract drug and disease IDs from filtered positive examples
    drug_ids, disease_ids = extract_node_ids_from_positives(positive_pairs)
    
    # Create feature vectors for positive samples
    pos_features, pos_labels = create_feature_vectors(embeddings, positive_pairs)
    pos_targets = np.ones(len(pos_features))
    
    print(f"Created {len(pos_features)} positive feature vectors")
    
    # Generate negative samples
    negative_pairs = generate_negative_samples(positive_pairs, drug_ids, disease_ids, negative_ratio)
    
    # Create feature vectors for negative samples
    neg_features, neg_labels = create_feature_vectors(embeddings, negative_pairs)
    neg_targets = np.zeros(len(neg_features))
    
    print(f"Created {len(neg_features)} negative feature vectors")
    
    # Combine positive and negative samples
    X = np.vstack([pos_features, neg_features])
    y = np.hstack([pos_targets, neg_targets])
    pair_labels = pos_labels + neg_labels
    
    print(f"Total samples: {len(X)} (positive: {len(pos_features)}, negative: {len(neg_features)})")
    
    # Train-test split
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, range(len(pair_labels)), test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Extract the actual pairs used in training
    training_positives = [pair_labels[i] for i in indices_train if y[i] == 1]
    training_negatives = [pair_labels[i] for i in indices_train if y[i] == 0]
    
    print(f"Training pairs: {len(training_positives)} positives, {len(training_negatives)} negatives")
    
    # Write training pairs file
    training_pairs = {
        "training_positives": training_positives,
        "training_negatives": training_negatives
    }
    training_pairs_file = os.path.join(version_dir, "training_pairs.json")
    with open(training_pairs_file, 'w') as f:
        json.dump(training_pairs, f, indent=2)
    
    print(f"Training pairs saved: {training_pairs_file}")
    
    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
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
    
    # Record end time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save model
    model_file = os.path.join(version_dir, "rf_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(rf, f)
    
    # Save results
    results_file = os.path.join(version_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create comprehensive provenance metadata
    provenance = {
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "script": "train_model.py",
        "version": f"model_{version}",
        "algorithm": "random_forest",
        "model_file": model_file,
        "results_file": results_file,
        "input_data": {
            "graph_dir": graph_dir,
            "embeddings_file": embeddings_file,
            "embeddings_version": embeddings_version,
            "embedding_info": embedding_info,
            "ground_truth": ground_truth_stats
        },
        "model_parameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "negative_ratio": negative_ratio
        },
        "data_splits": {
            "total_samples": len(X),
            "positive_samples": len(pos_features),
            "negative_samples": len(neg_features),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_dimension": X.shape[1]
        },
        "performance_metrics": metrics,
        "description": f"Random Forest model version {version} trained on {len(pos_features)} positive and {len(neg_features)} negative samples"
    }
    
    # Save provenance file
    provenance_file = os.path.join(version_dir, "provenance.json")
    with open(provenance_file, 'w') as f:
        json.dump(provenance, f, indent=2)
    
    # Save detailed report
    report_file = os.path.join(version_dir, "classification_report.txt")
    with open(report_file, 'w') as f:
        f.write("Classification Report\n")
        f.write("===================\n\n")
        f.write(f"Model Version: {version}\n")
        f.write(f"Dataset: {len(X)} total samples\n")
        f.write(f"Positive samples: {len(pos_features)}\n")
        f.write(f"Negative samples: {len(neg_features)}\n")
        f.write(f"Negative ratio: {negative_ratio}\n")
        f.write(f"Training time: {duration:.2f} seconds\n\n")
        
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nDetailed Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
    
    print(f"Provenance saved: {provenance_file}")
    print(f"Model saved: {model_file}")
    print(f"Results saved: {results_file}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Print results
    print("\nResults:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return version_dir


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Train Random Forest model with versioning and provenance")
    parser.add_argument("--graph-dir", required=True,
                       help="Path to graph directory (e.g., graphs/robokop_base/CCDD)")
    parser.add_argument("--ground-truth", required=True,
                       help="Path to ground truth CSV file")
    parser.add_argument("--embeddings-version", 
                       help="Specific embeddings version to use (e.g., embeddings_2)")
    parser.add_argument("--negative-ratio", type=int, default=1,
                       help="Ratio of negative to positive samples (default: 1)")
    parser.add_argument("--n-estimators", type=int, default=100,
                       help="Number of trees in random forest (default: 100)")
    parser.add_argument("--max-depth", type=int, default=10,
                       help="Maximum depth of trees (default: 10)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random state for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    version_dir = train_model(
        graph_dir=args.graph_dir,
        ground_truth_file=args.ground_truth,
        embeddings_version=args.embeddings_version,
        negative_ratio=args.negative_ratio,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    
    print(f"\nModel training complete!")
    print(f"Output directory: {version_dir}")


if __name__ == "__main__":
    main()