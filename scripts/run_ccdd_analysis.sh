#!/bin/bash
set -e

# Script to create CCDD graph and run node2vec analysis
# Usage: ./scripts/run_ccdd_analysis.sh

echo "=== CCDD Graph Analysis Pipeline ==="

# Activate conda environment
echo "Activating simplepredictions environment..."
source /opt/anaconda3/bin/activate simplepredictions

# Create CCDD graph
echo "Creating CCDD filtered graph..."
/opt/anaconda3/envs/simplepredictions/bin/python src/graph_modification/create_robokop_input.py --style CCDD

# Check if the edg file was created
EDG_FILE="graphs/CCDD/graph/edges.edg"
if [ ! -f "$EDG_FILE" ]; then
    echo "Error: $EDG_FILE was not created successfully"
    exit 1
fi

echo "CCDD graph created successfully: $EDG_FILE"
echo "Graph stats:"
wc -l "$EDG_FILE"

# Create embeddings directory
EMBEDDINGS_DIR="graphs/CCDD/embeddings"
mkdir -p "$EMBEDDINGS_DIR"

# Run node2vec using pecanpy
echo "Running node2vec with PecanPy..."
pecanpy --input "$EDG_FILE" --output "$EMBEDDINGS_DIR/embeddings.emb" --dimensions 512 --walk-length 30 --num-walks 10 --window-size 10 --p 1 --q 1 --workers 4

# Check if embeddings were created
EMB_FILE="$EMBEDDINGS_DIR/embeddings.emb"
if [ ! -f "$EMB_FILE" ]; then
    echo "Error: $EMB_FILE was not created successfully"
    exit 1
fi

echo "Node2vec embeddings created successfully: $EMB_FILE"
echo "Embeddings stats:"
wc -l "$EMB_FILE"

echo "=== CCDD Analysis Pipeline Complete ==="
echo "Graph: $EDG_FILE"  
echo "Embeddings: $EMB_FILE"
