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

3. **Evaluate Model** (optional, automatic in training):
   ```bash
   python src/modeling/evaluate_predictions.py \
     --graph-dir graphs/CCDD \
     --ground-truth "ground_truth/Indications List.csv"
   ```

## Project Structure

```
├── input_graphs/              # Original biomedical knowledge graph data
├── ground_truth/              # Known drug-disease relationships
│   └── Indications List.csv   # FDA/EMA approved indications
├── graphs/                    # Processed graph data
│   └── CCDD/                  # Chemical-Chemical + Disease-Disease graph
│       ├── graph/             # Filtered edges and nodes
│       ├── embeddings/        # Node2vec embeddings
│       └── models/            # Trained models and results
├── src/
│   ├── graph_modification/    # Graph filtering and preprocessing
│   ├── modeling/              # ML training and evaluation
│   └── ...
├── scripts/                   # Pipeline automation scripts
└── tests/                     # Unit tests
```

## How It Works

### 1. Graph Processing
- **Input**: Biomedical knowledge graph with nodes (drugs, diseases, genes) and edges (relationships)
- **Filtering**: Creates CCDD graph containing only Chemical-Chemical and Disease-Disease edges
- **Data Leakage Prevention**: Removes direct Drug-Disease edges to avoid training on target relationships

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
- `src/modeling/train_rf_model.py`: Random Forest training with balanced sampling
- `src/modeling/evaluate_predictions.py`: Comprehensive evaluation with ranking metrics

### Automation Scripts
- `scripts/run_ccdd_analysis.sh`: End-to-end graph processing and embedding generation
- `scripts/train_ccdd_model.sh`: Model training pipeline

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