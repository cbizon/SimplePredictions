# SimplePredictions

A biomedical knowledge graph link prediction system for predicting Drug-treats-Disease relationships using node2vec embeddings and Random Forest models.

## Overview

This project implements an end-to-end machine learning pipeline for predicting therapeutic relationships between drugs and diseases. The system uses graph neural network embeddings (node2vec) combined with traditional machine learning (Random Forest) to make predictions while carefully avoiding data leakage.

### Key Features

- **Data Leakage Prevention**: Removes direct Chemical-Disease edges during training
- **Multiple Graph Styles**: CCDD, CGD, CFD (with synthetic fake genes) and subclass variants
- **CFD Upper Bounds**: Synthetic fake genes create perfect pathways for performance limits
- **Node2vec Embeddings**: Uses PecanPy for efficient node2vec implementation
- **Comprehensive Provenance**: Complete metadata tracking for reproducibility
- **Predicate Analysis**: Detailed statistics on relationship types in filtered graphs
- **Ranking-based Evaluation**: Precision@K, Recall@K, Hits@K metrics for drug discovery
- **Memory-Optimized Evaluation**: Batch processing for large-scale evaluation

## Quick Start

### Prerequisites

- Conda environment with Python 3.11+
- Access to biomedical knowledge graph data (RoboKOP format)

### Setup Environment

```bash
# Create and activate conda environment
conda create -n simplepredictions python=3.11
conda activate simplepredictions

# Install core dependencies
conda install scikit-learn pandas numpy matplotlib seaborn

# Install specialized packages
pip install pecanpy jsonlines flask
```

### Step-by-Step Pipeline

The complete pipeline involves four main steps:

#### 1. Graph Creation and Filtering
```bash
# Create CCDD graph (Chemical-Chemical + Disease-Disease edges only)
python src/graph_modification/create_robokop_input.py \
    --style CCDD \
    --input-dir input_graphs/robokop_base_nonredundant \
    --output-dir graphs

# Or create CFD graph with synthetic fake genes for upper bounds
python src/graph_modification/create_robokop_input.py \
    --style CFD \
    --input-dir input_graphs/robokop_base_nonredundant \
    --indications-file "ground_truth/Indications List.csv" \
    --output-dir graphs
```

#### 2. Generate Node2Vec Embeddings
```bash
# Generate 512-dimensional embeddings using PecanPy
python src/embedding/generate_embeddings.py \
    --graph-file graphs/robokop_base_nonredundant_CCDD/graph/edges.edg \
    --dimensions 512 \
    --walk-length 30 \
    --num-walks 10 \
    --window-size 10 \
    --p 1 \
    --q 1
```

#### 3. Train Random Forest Model
```bash
# Train with random negative sampling
python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CCDD \
    --ground-truth "ground_truth/Indications List.csv" \
    --embeddings-version embeddings_0 \
    --negative-ratio 1

# Or train with contraindications as negatives
python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CCDD \
    --ground-truth "ground_truth/Indications List.csv" \
    --contraindications "ground_truth/Contraindications List.csv" \
    --embeddings-version embeddings_0
```

#### 4. Evaluate Model Performance
```bash
# Comprehensive ranking-based evaluation
python src/modeling/evaluate_model.py \
    --model-dir graphs/robokop_base_nonredundant_CCDD/embeddings/embeddings_0/models/model_0
```

### Web Interface for Model Visualization

The project includes a Flask web application for interactive visualization and comparison of model evaluation results:

```bash
# Install Flask dependency
pip install flask

# Launch the web interface
./run_app.sh
# Or directly:
python app.py
```

Then navigate to **http://localhost:5000** in your browser.

#### Web Interface Features:
- **Hierarchical Model Browser**: Organized by graph type → embedding version → model version
- **Interactive Model Selection**: Choose multiple models for comparison
- **Comprehensive Visualization**: 4×2 grid of evaluation plots:
  - **Rows**: Precision@K, Recall@K, Total Recall@K, Hits@K
  - **Columns**: K range 1-1000 (zoomed) and 1-Max (full range)
- **Model Metadata Display**: Shows training parameters, embedding details, and provenance
- **Automatic Discovery**: Finds all models with evaluation_metrics.json files

The webapp automatically discovers all trained models in the graphs/ directory and provides an intuitive interface for comparing their performance across different metrics and K ranges.

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
│       │   ├── edges.edg      # PecanPy format edge list
│       │   ├── nodes.jsonl    # Node metadata (includes fake genes for CFD)
│       │   ├── provenance.json # Graph creation metadata
│       │   └── predicate_stats.json # Predicate type counts and statistics
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
├── app.py                     # Flask web application for model visualization
├── templates/
│   └── index.html             # Web interface for model comparison
├── run_app.sh                 # Script to launch webapp
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
- **Filtering**: Multiple graph styles available:
  - **CCDD**: Chemical-Chemical + Disease-Disease edges only
  - **CGD**: Chemical-Gene-Disease pathways (no direct CD edges)
  - **CFD**: CCDD + synthetic fake genes connecting known indications (upper bounds)
  - **Subclass variants**: Include biolink:subclass_of relationships
- **Data Leakage Prevention**: Removes direct Drug-Disease edges to avoid training on target relationships
- **Predicate Analysis**: Tracks and saves statistics on relationship types in filtered graphs

### 2. Synthetic Fake Genes (CFD Style)
The CFD graph style creates "ideal" synthetic pathways to establish performance upper bounds:
- Creates unique fake gene for each known drug-disease indication
- Connects Chemical→FakeGene→Disease with perfect synthetic pathways
- Only includes indications where both chemical and disease exist in original graph
- Provides theoretical maximum performance achievable with available knowledge

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