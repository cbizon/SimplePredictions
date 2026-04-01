# Integrating External Models into SimplePredictions

This guide shows how to integrate predictions from external models into the SimplePredictions evaluation framework.

## Quick Start

If you have a model that makes predictions for drug-disease pairs, you can integrate it in one command:

```bash
python src/modeling/integrate_external_predictions.py \
    --predictions your_predictions.tsv \
    --ground-truth "ground_truth/Indications List.csv" \
    --training-pairs your_training_pairs.tsv \
    --model-name "my_model"
```

Your model will then:
1. Be evaluated using SimplePredictions' standard metrics
2. Appear in the webapp at http://localhost:5001
3. Be directly comparable to all other models

## Input Formats

### Predictions File (Required)

**TSV Format (recommended):**
```tsv
drug_id	disease_id	score
CHEBI:6801	MONDO:0005015	0.95
CHEBI:6801	MONDO:0011382	0.87
```

**CSV Format:**
```csv
drug_id,disease_id,score
CHEBI:6801,MONDO:0005015,0.95
CHEBI:6801,MONDO:0011382,0.87
```

**JSON Format:**
```json
{
  "predictions": [
    {"drug": "CHEBI:6801", "disease": "MONDO:0005015", "score": 0.95},
    {"drug": "CHEBI:6801", "disease": "MONDO:0011382", "score": 0.87}
  ]
}
```

### Training Pairs File (REQUIRED for Data Leakage Prevention)

You **must** provide the training pairs used by your model. This ensures they're excluded from evaluation and prevents data leakage:

**TSV Format (recommended):**
```tsv
drug_id	disease_id	label
CHEBI:6801	MONDO:0005015	1
CHEBI:1234	MONDO:0005678	0
```

**CSV Format:**
```csv
drug_id,disease_id,label
CHEBI:6801,MONDO:0005015,1
CHEBI:1234,MONDO:0005678,0
```

**JSON Format:**
```json
{
  "training_positives": [["CHEBI:6801", "MONDO:0005015"], ...],
  "training_negatives": [["CHEBI:1234", "MONDO:0005678"], ...]
}
```

## Examples

### Basic Integration

```bash
python src/modeling/integrate_external_predictions.py \
    --predictions my_predictions.tsv \
    --ground-truth "ground_truth/Indications List.csv" \
    --training-pairs my_training_pairs.tsv \
    --model-name "gnn_model_v1"
```

### With Embedding Metadata

```bash
python src/modeling/integrate_external_predictions.py \
    --predictions predictions.tsv \
    --ground-truth "ground_truth/Indications List.csv" \
    --training-pairs training_pairs.tsv \
    --model-name "my_model" \
    --embedding-dim 768 \
    --embedding-algorithm "graph_transformer"
```

## How It Works

1. **Loads your predictions** from CSV/TSV/JSON
2. **Maps to SimplePredictions ground truth universe** (same drug/disease IDs)
3. **Excludes training pairs** (required to prevent data leakage)
4. **Calculates standard metrics** using `calculate_metrics_from_predictions()`:
   - Precision@K
   - Recall@K
   - Total Recall@K
   - Hits@K
5. **Generates output files**:
   - `evaluation_metrics.json` (for webapp)
   - `provenance.json` (metadata)
   - `training_pairs.json` (from your input)

## Output Location

By default, creates: `graphs/{model_name}/embeddings/embeddings_0/models/model_0/`

Custom location:
```bash
python src/modeling/integrate_external_predictions.py \
    --predictions predictions.csv \
    --ground-truth "ground_truth/Indications List.csv" \
    --output-dir custom/path/to/model
```

## Viewing Results

After integration:

1. Start the webapp: `./run_app.sh`
2. Navigate to http://localhost:5001
3. Select your model alongside SimplePredictions models
4. Compare metrics side-by-side

## Requirements for Comparability

For results to be comparable across models:

1. ✅ **Same ground truth file** - All models use `ground_truth/Indications List.csv`
2. ✅ **Same evaluation methodology** - The script handles this automatically
3. ✅ **Same metrics** - Uses SimplePredictions' `calculate_metrics_from_predictions()`
4. ✅ **Training pair exclusion** - Training pairs are required to prevent data leakage

## What Can Differ

- Graph structure (different graph filtering)
- Embeddings (different methods, dimensions)
- Model architecture (GNN, transformer, RF, etc.)
- Training approach (different negatives, losses, etc.)

All will be evaluated on the same evaluation set and produce comparable metrics!
