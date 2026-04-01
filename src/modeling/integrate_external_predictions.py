#!/usr/bin/env python3
"""Integrate external model predictions into SimplePredictions framework.

This script allows external models to be evaluated using SimplePredictions' standard
metrics and integrated into the web visualization interface.

External models provide:
1. Prediction scores for drug-disease pairs (CSV/TSV/JSON)
2. Optional: Training pairs used (for data leakage prevention)

The script handles:
- Loading ground truth from SimplePredictions
- Mapping external predictions to evaluation universe
- Calculating metrics using SimplePredictions' calculate_metrics_from_predictions()
- Generating evaluation_metrics.json and provenance.json for webapp integration
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.train_model import load_ground_truth
from modeling.evaluate_model import calculate_metrics_from_predictions


def load_predictions(predictions_file):
    """Load predictions from external model.

    Supported formats:
    - CSV/TSV: columns drug_id, disease_id, score
    - JSON: {"predictions": [{"drug": "...", "disease": "...", "score": 0.95}, ...]}

    Args:
        predictions_file: Path to predictions file

    Returns:
        dict: {(drug_id, disease_id): score}
    """
    ext = os.path.splitext(predictions_file)[1].lower()

    if ext == '.json':
        with open(predictions_file, 'r') as f:
            data = json.load(f)

        predictions = {}
        if 'predictions' in data:
            for pred in data['predictions']:
                predictions[(pred['drug'], pred['disease'])] = float(pred['score'])
        else:
            # Assume {drug_id: {disease_id: score}} format
            for drug_id, diseases in data.items():
                for disease_id, score in diseases.items():
                    predictions[(drug_id, disease_id)] = float(score)

    elif ext in ['.csv', '.tsv']:
        delimiter = ',' if ext == '.csv' else '\t'
        df = pd.read_csv(predictions_file, delimiter=delimiter)

        required_cols = ['drug_id', 'disease_id', 'score']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV/TSV must have columns: {required_cols}. Found: {df.columns.tolist()}")

        predictions = {}
        for _, row in df.iterrows():
            predictions[(row['drug_id'], row['disease_id'])] = float(row['score'])
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .json, .csv, or .tsv")

    return predictions


def load_training_pairs(training_pairs_file):
    """Load training pairs from external model.

    Supported formats:
    - JSON: {"training_positives": [...], "training_negatives": [...]}
    - CSV/TSV: columns drug_id, disease_id, label (1=positive, 0=negative)

    Args:
        training_pairs_file: Path to training pairs file

    Returns:
        tuple: (training_positives_set, training_negatives_set)
    """

    ext = os.path.splitext(training_pairs_file)[1].lower()

    if ext == '.json':
        with open(training_pairs_file, 'r') as f:
            data = json.load(f)

        training_positives = set(tuple(pair) for pair in data['training_positives'])
        training_negatives = set(tuple(pair) for pair in data['training_negatives'])

    elif ext in ['.csv', '.tsv']:
        delimiter = ',' if ext == '.csv' else '\t'
        df = pd.read_csv(training_pairs_file, delimiter=delimiter)

        required_cols = ['drug_id', 'disease_id', 'label']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV/TSV must have columns: {required_cols}. Found: {df.columns.tolist()}")

        training_positives = set()
        training_negatives = set()
        for _, row in df.iterrows():
            pair = (row['drug_id'], row['disease_id'])
            if row['label'] == 1:
                training_positives.add(pair)
            else:
                training_negatives.add(pair)
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .json, .csv, or .tsv")

    return training_positives, training_negatives


def integrate_external_model(predictions_file, ground_truth_file, training_pairs_file,
                             output_dir=None, model_name=None, embedding_info=None):
    """Integrate external model into SimplePredictions framework.

    Args:
        predictions_file: Path to predictions (CSV/TSV/JSON)
        ground_truth_file: Path to SimplePredictions ground truth
        training_pairs_file: Path to training pairs (REQUIRED for data leakage prevention)
        output_dir: Output directory (defaults to graphs/{model_name}/...)
        model_name: Model name (defaults to predictions filename)
        embedding_info: Optional embedding metadata dict

    Returns:
        str: Path to generated evaluation_metrics.json
    """
    start_time = datetime.now()

    # Load predictions
    print("=== Loading External Predictions ===")
    predictions = load_predictions(predictions_file)
    print(f"Loaded {len(predictions):,} predictions")

    # Load ground truth
    print("\n=== Loading SimplePredictions Ground Truth ===")
    all_positive_pairs, gt_stats = load_ground_truth(ground_truth_file, embeddings=None)
    original_gt_count = gt_stats["clean_rows"]
    print(f"Ground truth indications: {original_gt_count}")

    all_drug_ids = set(pair[0] for pair in all_positive_pairs)
    all_disease_ids = set(pair[1] for pair in all_positive_pairs)
    print(f"Unique drugs: {len(all_drug_ids)}, Unique diseases: {len(all_disease_ids)}")

    # Load training pairs
    print("\n=== Loading Training Pairs ===")
    training_positives, training_negatives = load_training_pairs(training_pairs_file)
    print(f"Training positives: {len(training_positives)}")
    print(f"Training negatives: {len(training_negatives)}")

    training_pairs_to_exclude = training_positives | training_negatives
    all_positive_pairs_set = set(all_positive_pairs)
    evaluation_positives = all_positive_pairs_set - training_positives

    # Build evaluation set from predictions
    print("\n=== Building Evaluation Set ===")
    eval_pairs = []
    y_scores = []
    y_true = []

    for pair, score in predictions.items():
        # Skip training pairs
        if pair in training_pairs_to_exclude:
            continue

        # Only include if in ground truth universe
        if pair[0] not in all_drug_ids or pair[1] not in all_disease_ids:
            continue

        eval_pairs.append(pair)
        y_scores.append(score)
        y_true.append(1 if pair in evaluation_positives else 0)

    print(f"Evaluation pairs: {len(eval_pairs):,}")
    print(f"Evaluation positives: {sum(y_true)}")
    print(f"Evaluation negatives: {len(y_true) - sum(y_true)}")

    # Calculate metrics using SimplePredictions shared function
    print("\n=== Calculating Metrics ===")
    metrics = calculate_metrics_from_predictions(
        y_true=y_true,
        y_scores=y_scores,
        eval_pairs=eval_pairs,
        original_gt_count=original_gt_count,
        training_positives_count=len(training_positives)
    )

    # Print summary
    print("\n=== Metrics Summary ===")
    for k in [1, 5, 10, 20, 50, 100]:
        if k in metrics['precision_at_k']:
            p = metrics['precision_at_k'][k]
            r = metrics['recall_at_k'][k]
            tr = metrics['total_recall_at_k'][k]
            h = metrics['hits_at_k'][k]
            print(f"K={k:3d}: P={p:.4f}, R={r:.4f}, TR={tr:.4f}, H={h:.4f}")

    # Create output directory
    if output_dir is None:
        if model_name is None:
            model_name = os.path.splitext(os.path.basename(predictions_file))[0]
        output_dir = os.path.join("graphs", model_name, "embeddings", "embeddings_0", "models", "model_0")

    os.makedirs(output_dir, exist_ok=True)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Save evaluation_metrics.json (webapp format)
    evaluation_metrics = {
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration,
        "model_dir": output_dir,
        "embeddings_file": "external",
        "ground_truth_file": ground_truth_file,
        "evaluation_summary": {
            "all_combinations": len(predictions),
            "training_pairs_excluded": len(training_pairs_to_exclude),
            "evaluation_combinations": metrics['evaluation_combinations'],
            "evaluation_positives": metrics['evaluation_positives'],
            "evaluation_negatives": metrics['evaluation_negatives']
        },
        "precision_at_k": {str(k): float(v) for k, v in metrics['precision_at_k'].items()},
        "recall_at_k": {str(k): float(v) for k, v in metrics['recall_at_k'].items()},
        "total_recall_at_k": {str(k): float(v) for k, v in metrics['total_recall_at_k'].items()},
        "total_recall_max": float(metrics['total_recall_max']),
        "hits_at_k": {str(k): float(v) for k, v in metrics['hits_at_k'].items()},
        "k_values": [int(k) for k in metrics['k_values']]
    }

    metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_file}")

    # Save provenance.json (webapp metadata)
    provenance = {
        "timestamp": start_time.isoformat(),
        "script": "integrate_external_predictions.py",
        "model_type": "external",
        "model_parameters": {"negative_ratio": 1},
        "input_data": {
            "predictions_file": predictions_file,
            "ground_truth": {"ground_truth_file": ground_truth_file},
            "training_pairs_file": training_pairs_file,
            "negative_sampling_method": "external" if training_pairs_file else "unknown",
            "contraindications": None,
            "embedding_info": embedding_info or {
                "embedding_dim": "unknown",
                "embedding_provenance": {
                    "algorithm": "external",
                    "tool": "external",
                    "parameters": {}
                }
            }
        }
    }

    provenance_file = os.path.join(output_dir, "provenance.json")
    with open(provenance_file, 'w') as f:
        json.dump(provenance, f, indent=2)
    print(f"Provenance saved: {provenance_file}")

    # Save training pairs (always, since it's required)
    training_pairs_output = {
        "training_positives": [list(pair) for pair in training_positives],
        "training_negatives": [list(pair) for pair in training_negatives]
    }
    training_file = os.path.join(output_dir, "training_pairs.json")
    with open(training_file, 'w') as f:
        json.dump(training_pairs_output, f, indent=2)
    print(f"Training pairs saved: {training_file}")

    print(f"\n=== Integration Complete ===")
    print(f"Output: {output_dir}")
    print(f"This model will appear in the webapp at http://localhost:5001")

    return metrics_file


def main():
    parser = argparse.ArgumentParser(
        description="Integrate external model predictions into SimplePredictions framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic integration with TSV files (recommended)
  python src/modeling/integrate_external_predictions.py \\
      --predictions my_predictions.tsv \\
      --ground-truth "ground_truth/Indications List.csv" \\
      --training-pairs my_training_pairs.tsv \\
      --model-name "gnn_model_v1"

  # With embedding metadata
  python src/modeling/integrate_external_predictions.py \\
      --predictions predictions.tsv \\
      --ground-truth "ground_truth/Indications List.csv" \\
      --training-pairs training_pairs.tsv \\
      --model-name "transformer_model" \\
      --embedding-dim 768 \\
      --embedding-algorithm "graph_transformer"

File formats:

  Predictions (TSV - recommended):
    drug_id	disease_id	score
    CHEBI:123	MONDO:456	0.95

  Training pairs (TSV - recommended):
    drug_id	disease_id	label
    CHEBI:123	MONDO:456	1
    CHEBI:789	MONDO:012	0

  Also supports CSV and JSON (see examples/README_external_models.md)
        """
    )

    parser.add_argument("--predictions", required=True,
                       help="Path to predictions file (CSV/TSV/JSON)")
    parser.add_argument("--ground-truth", required=True,
                       help="Path to SimplePredictions ground truth")
    parser.add_argument("--training-pairs", required=True,
                       help="Path to training pairs file (REQUIRED for data leakage prevention)")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory (default: graphs/{model-name}/...)")
    parser.add_argument("--model-name", default=None,
                       help="Model name (default: predictions filename)")
    parser.add_argument("--embedding-dim", type=int, default=None,
                       help="Embedding dimension (for provenance)")
    parser.add_argument("--embedding-algorithm", default=None,
                       help="Embedding algorithm (for provenance)")

    args = parser.parse_args()

    # Build embedding info
    embedding_info = None
    if args.embedding_dim or args.embedding_algorithm:
        embedding_info = {
            "embedding_dim": args.embedding_dim or "unknown",
            "embedding_provenance": {
                "algorithm": args.embedding_algorithm or "external",
                "tool": "external",
                "parameters": {}
            }
        }

    integrate_external_model(
        predictions_file=args.predictions,
        ground_truth_file=args.ground_truth,
        training_pairs_file=args.training_pairs,
        output_dir=args.output_dir,
        model_name=args.model_name,
        embedding_info=embedding_info
    )


if __name__ == "__main__":
    main()
