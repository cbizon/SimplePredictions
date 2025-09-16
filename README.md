# SimplePredictions

A biomedical knowledge graph link prediction system for predicting Drug-treats-Disease relationships using node2vec embeddings and Random Forest models.

## Overview

This project implements an end-to-end machine learning pipeline for predicting therapeutic relationships between drugs and diseases. The system uses graph neural network embeddings (node2vec) combined with traditional machine learning (Random Forest) to make predictions while carefully avoiding data leakage.

### Key Features

- **Data Leakage Prevention**: Removes direct Chemical-Disease edges during training
- **Graph Filtering**: Creates specialized graph views (CCDD: Chemical-Chemical + Disease-Disease)  
- **Node2vec Embeddings**: Uses PecanPy for efficient node2vec implementation
- **Ranking-based Evaluation**: Precision@K, Recall@K, Hits@K metrics for drug discovery
- **Proper Train-Test Splitting**: Ensures evaluation on truly unseen data

## Quick Start

### Prerequisites

- Conda environment with Python 3.8+
- Required packages: `pecanpy`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `jsonlines`

### Setup Environment

```bash
# Create and activate conda environment
conda create -n simplepredictions python=3.8
conda activate simplepredictions

# Install dependencies
conda install scikit-learn pandas numpy matplotlib seaborn
pip install pecanpy jsonlines
```

### Running the Pipeline

1. **Create CCDD Graph and Generate Embeddings**:
   ```bash
   ./scripts/run_ccdd_analysis.sh
   ```

2. **Train Random Forest Model**:
   ```bash
   ./scripts/train_ccdd_model.sh
   ```

3. **Evaluate Model** (new simplified interface):
   ```bash
   python src/modeling/evaluate_model.py \
     --model-dir graphs/robokop_base_nonredundant_CCDD/embeddings/embeddings_0/models/model_0
   ```

4. **Run Full Pipeline** (create graphs, embeddings, models, and evaluations):
   ```bash
   ./regenerate_all.sh
   ```

## Project Structure

```
├── input_graphs/              # Original biomedical knowledge graph data
│   ├── robokop_base/          # Full dataset (23GB edges, 1.6GB nodes)
│   │   ├── robokop_base_edges.jsonl
│   │   └── robokop_base_nodes.jsonl
│   └── robokop_base_nonredundant/  # Deduplicated dataset (10GB edges, 1.6GB nodes) [DEFAULT]
│       ├── edges.jsonl
│       └── nodes.jsonl
├── ground_truth/              # Known drug-disease relationships
│   ├── Indications List.csv   # FDA/EMA approved indications
│   └── Contraindications List.csv  # Known contraindications (negative examples)
├── graphs/                    # Processed graph data
│   └── {graphname}/           # e.g., robokop_base_nonredundant_CCDD
│       ├── graph/             # Filtered edges and nodes
│       └── embeddings/        # Node2vec embeddings
│           └── embeddings_0/  # Versioned embedding runs
│               ├── embeddings.emb      # Node2vec embedding vectors
│               ├── provenance.json     # Embedding generation metadata
│               └── models/             # Models trained on these embeddings
│                   ├── model_0/        # Individual model runs
│                   │   ├── rf_model.pkl           # Trained Random Forest
│                   │   ├── provenance.json        # Complete model metadata
│                   │   ├── training_pairs.json    # Exact training data used
│                   │   ├── evaluation_metrics.json # Evaluation results
│                   │   └── results.json           # Training performance
│                   └── model_1/        # Additional model variants (e.g., with contraindications)
├── src/
│   ├── graph_modification/    # Graph filtering and preprocessing
│   ├── embedding/             # Node2vec embedding generation
│   ├── modeling/              # ML training and evaluation
│   └── ...
├── scripts/                   # Pipeline automation scripts
└── tests/                     # Unit tests
```

### New Hierarchical Organization

**Models are now nested under embeddings**: `/graphs/{graphname}/embeddings/{version}/models/{model_version}/`

This structure ensures that:
- Each model is directly associated with its specific embedding version
- Evaluation results are stored alongside the model that generated them
- The complete provenance chain is maintained: graph → embeddings → model → evaluation
- No separate `/evaluations/` directory is needed

### Input Datasets

The system supports multiple input graph datasets:

- **robokop_base_nonredundant** (Default): Smaller, deduplicated version with reduced redundancy
- **robokop_base**: Full original dataset with all edges and relationships

You can specify different datasets using command line arguments.

## How It Works

### 1. Graph Processing
- **Input**: Biomedical knowledge graph with nodes (drugs, diseases, genes) and edges (relationships)
- **Filtering**: Creates CCDD graph containing only Chemical-Chemical and Disease-Disease edges
- **Data Leakage Prevention**: Removes direct Drug-Disease edges to avoid training on target relationships

**Command Line Usage:**
```bash
# Using default nonredundant dataset
python src/graph_modification/create_robokop_input.py --style CCDD

# Using original full dataset  
python src/graph_modification/create_robokop_input.py \
  --style CCDD \
  --input-dir input_graphs/robokop_base \
  --nodes-filename robokop_base_nodes.jsonl \
  --edges-filename robokop_base_edges.jsonl
```

### 2. Embedding Generation
- Uses node2vec via PecanPy to generate 512-dimensional node embeddings
- Parameters: `--dimensions 512 --walk-length 30 --num-walks 10 --window-size 10 --p 1 --q 1`

### 3. Model Training
- **Features**: Concatenated drug and disease embeddings (1024-dim total)
- **Model**: Random Forest with balanced positive/negative sampling
- **Target**: Predict Drug-treats-Disease relationships

### 4. Evaluation
- **Metrics**: Precision@K, Recall@K, Hits@K for K = [1, 5, 10, 20, 50, 100, ...]
- **Test Set**: 20% held-out data, never seen during training
- **Visualization**: Ranking curves and precision-recall plots

## Key Results

Recent CCDD model performance on test set:
- **Perfect Precision@K** for top 100 predictions (100% accuracy on highest-confidence predictions)
- **Test Set Size**: 2,030 samples (1,015 positive, 1,015 negative)
- **Data Leakage**: Properly prevented through graph filtering and train-test splitting

## File Descriptions

### Core Scripts
- `src/graph_modification/create_robokop_input.py`: Graph filtering with data leakage prevention
- `src/embedding/generate_embeddings.py`: Node2vec embedding generation using PecanPy
- `src/modeling/train_model.py`: Random Forest training with balanced sampling and provenance
- `src/modeling/evaluate_model.py`: Simplified evaluation interface using model metadata (only requires `--model-dir`)

### Automation Scripts
- `scripts/run_ccdd_analysis.sh`: End-to-end graph processing and embedding generation
- `scripts/train_ccdd_model.sh`: Model training pipeline
- `regenerate_all.sh`: Complete pipeline including graph creation, embeddings, training, and evaluation

### Key Functions
- `has_cd_edge()`: Detects Chemical-Disease edges for data leakage prevention
- `keep_CCDD()`: Filters graph to Chemical-Chemical + Disease-Disease edges only
- `generate_negative_samples()`: Creates balanced negative examples for training

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

## Data Leakage Prevention

**Critical Feature**: The system prevents data leakage by:

1. **Graph Filtering**: Removes all direct Chemical-Disease edges during graph construction
2. **Feature Engineering**: Only uses indirect relationships (Chemical-Chemical, Disease-Disease)  
3. **Proper Evaluation**: Uses held-out test set that was never seen during training

This ensures the model learns to predict Drug-Disease relationships through indirect graph patterns rather than memorizing direct connections.

## Contributing

- Follow existing code patterns and naming conventions
- Maintain high test coverage with pytest
- Don't use mocks - test with real data
- Treat root causes of problems, not symptoms

## License

[Add license information]

## Citation

[Add citation information when published]