#!/bin/bash

set -e

# Activate conda environment
source /opt/anaconda3/bin/activate simplepredictions

# Create CCFD graph
python src/graph_modification/create_robokop_input.py \
    --style CCFD \
    --input-dir input_graphs/robokop_base_nonredundant \
    --indications-file "ground_truth/Indications List.csv" \
    --output-dir graphs

# Generate embeddings
python src/embedding/generate_embeddings.py \
    --graph-file graphs/robokop_base_nonredundant_CCFD/graph/edges.edg \
    --dimensions 512 \
    --walk-length 30 \
    --num-walks 10 \
    --window-size 10 \
    --p 1 \
    --q 1

# Train model
python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CCFD \
    --ground-truth "ground_truth/Indications List.csv" \
    --embeddings-version embeddings_0 \
    --negative-ratio 1 \
    --n-estimators 100 \
    --max-depth 10 \
    --random-state 42

# Evaluate model
python src/modeling/evaluate_model.py \
    --model-dir graphs/robokop_base_nonredundant_CCFD/embeddings/embeddings_0/models/model_0