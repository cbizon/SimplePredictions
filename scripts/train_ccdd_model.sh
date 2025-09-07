#!/bin/bash
set -e

# Script to train Random Forest model on CCDD embeddings
# Usage: ./scripts/train_ccdd_model.sh

echo "=== CCDD Model Training Pipeline ==="

# Activate conda environment
echo "Activating simplepredictions environment..."
source /opt/anaconda3/bin/activate simplepredictions

# Check if CCDD embeddings exist
EMBEDDINGS_FILE="graphs/CCDD/embeddings/embeddings.emb"
if [ ! -f "$EMBEDDINGS_FILE" ]; then
    echo "Error: CCDD embeddings not found at $EMBEDDINGS_FILE"
    echo "Please run ./scripts/run_ccdd_analysis.sh first"
    exit 1
fi

# Check if ground truth file exists
GROUND_TRUTH_FILE="ground_truth/Indications List.csv"
if [ ! -f "$GROUND_TRUTH_FILE" ]; then
    echo "Error: Ground truth file not found at '$GROUND_TRUTH_FILE'"
    exit 1
fi

echo "CCDD embeddings found: $EMBEDDINGS_FILE"
echo "Ground truth file found: $GROUND_TRUTH_FILE"

# Train the model
echo "Training Random Forest model..."
/opt/anaconda3/envs/simplepredictions/bin/python src/modeling/train_rf_model.py \
    --graph-dir graphs/CCDD \
    --ground-truth "$GROUND_TRUTH_FILE" \
    --negative-ratio 1

# Check if model was created
MODEL_FILE="graphs/CCDD/models/rf_model.pkl"
RESULTS_FILE="graphs/CCDD/models/results.json"

if [ -f "$MODEL_FILE" ] && [ -f "$RESULTS_FILE" ]; then
    echo "=== Model Training Complete ==="
    echo "Model saved: $MODEL_FILE"
    echo "Results saved: $RESULTS_FILE"
    echo "Detailed report: graphs/CCDD/models/classification_report.txt"
    
    # Show quick results summary
    echo ""
    echo "Quick Results Summary:"
    cat "$RESULTS_FILE"
else
    echo "Error: Model training failed - output files not found"
    exit 1
fi