#!/bin/bash
set -e
export PATH="/opt/anaconda3/envs/simplepredictions/bin:$PATH"

# Treats-Only CD Edge Analysis Script
# This script creates graphs with only treats CD edges included to analyze
# the impact of partial data leakage (treating relationships only).

echo "=========================================="
echo "Treats-Only CD Edge Analysis Pipeline"
echo "=========================================="
echo ""
echo "This will create graphs with treats CD edges only:"
echo "  - CCDD_with_cd_treats"
echo "  - CCDD_with_subclass_with_cd_treats"
echo "  - CGD_with_cd_treats"
echo "  - CGD_with_subclass_with_cd_treats"
echo ""

# =============================================================================
# PHASE 1: Create Graphs
# =============================================================================
echo "PHASE 1: Creating graphs..."
echo "----------------------------"

python src/graph_modification/create_robokop_input.py --style CCDD_with_cd_treats
python src/graph_modification/create_robokop_input.py --style CCDD_with_subclass_with_cd_treats
python src/graph_modification/create_robokop_input.py --style CGD_with_cd_treats
python src/graph_modification/create_robokop_input.py --style CGD_with_subclass_with_cd_treats

echo "✓ Graphs created"
echo ""

# =============================================================================
# PHASE 2: Generate Embeddings
# =============================================================================
echo "PHASE 2: Generating embeddings..."
echo "----------------------------------"

# Standard embedding parameters for all graphs
EMBEDDING_PARAMS="--dimensions 512 --walk-length 30 --num-walks 10 --window-size 10 --p 1 --q 1"

python src/embedding/generate_embeddings.py --graph-file graphs/robokop_base_nonredundant_CCDD_with_cd_treats/graph/edges.edg $EMBEDDING_PARAMS
python src/embedding/generate_embeddings.py --graph-file graphs/robokop_base_nonredundant_CCDD_with_subclass_with_cd_treats/graph/edges.edg $EMBEDDING_PARAMS
python src/embedding/generate_embeddings.py --graph-file graphs/robokop_base_nonredundant_CGD_with_cd_treats/graph/edges.edg $EMBEDDING_PARAMS
python src/embedding/generate_embeddings.py --graph-file graphs/robokop_base_nonredundant_CGD_with_subclass_with_cd_treats/graph/edges.edg $EMBEDDING_PARAMS

echo "✓ Embeddings generated"
echo ""

# =============================================================================
# PHASE 3: Train Models
# =============================================================================
echo "PHASE 3: Training models..."
echo "---------------------------"

GROUND_TRUTH="ground_truth/Indications List.csv"
CONTRAINDICATIONS="ground_truth/Contraindications List.csv"
EMBEDDINGS_VERSION="embeddings_0"

# Use fixed random seed for reproducibility
RANDOM_SEED=42

echo "Training with random negatives..."
python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CCDD_with_cd_treats \
    --ground-truth "$GROUND_TRUTH" \
    --embeddings-version $EMBEDDINGS_VERSION \
    --negative-ratio 1 \
    --random-state $RANDOM_SEED &

python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CCDD_with_subclass_with_cd_treats \
    --ground-truth "$GROUND_TRUTH" \
    --embeddings-version $EMBEDDINGS_VERSION \
    --negative-ratio 1 \
    --random-state $RANDOM_SEED &

python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CGD_with_cd_treats \
    --ground-truth "$GROUND_TRUTH" \
    --embeddings-version $EMBEDDINGS_VERSION \
    --negative-ratio 1 \
    --random-state $RANDOM_SEED &

python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CGD_with_subclass_with_cd_treats \
    --ground-truth "$GROUND_TRUTH" \
    --embeddings-version $EMBEDDINGS_VERSION \
    --negative-ratio 1 \
    --random-state $RANDOM_SEED &

wait

echo "Training with contraindication negatives..."
python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CCDD_with_cd_treats \
    --ground-truth "$GROUND_TRUTH" \
    --contraindications "$CONTRAINDICATIONS" \
    --embeddings-version $EMBEDDINGS_VERSION \
    --random-state $RANDOM_SEED &

python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CCDD_with_subclass_with_cd_treats \
    --ground-truth "$GROUND_TRUTH" \
    --contraindications "$CONTRAINDICATIONS" \
    --embeddings-version $EMBEDDINGS_VERSION \
    --random-state $RANDOM_SEED &

python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CGD_with_cd_treats \
    --ground-truth "$GROUND_TRUTH" \
    --contraindications "$CONTRAINDICATIONS" \
    --embeddings-version $EMBEDDINGS_VERSION \
    --random-state $RANDOM_SEED &

python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CGD_with_subclass_with_cd_treats \
    --ground-truth "$GROUND_TRUTH" \
    --contraindications "$CONTRAINDICATIONS" \
    --embeddings-version $EMBEDDINGS_VERSION \
    --random-state $RANDOM_SEED &

wait

echo "✓ Models trained"
echo ""

# =============================================================================
# PHASE 4: Evaluate Models
# =============================================================================
echo "PHASE 4: Evaluating models..."
echo "------------------------------"

# Evaluate all treats-only variant models
echo "Evaluating CCDD_with_cd_treats models..."
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CCDD_with_cd_treats/embeddings/$EMBEDDINGS_VERSION/models/model_0
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CCDD_with_cd_treats/embeddings/$EMBEDDINGS_VERSION/models/model_1

echo "Evaluating CCDD_with_subclass_with_cd_treats models..."
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CCDD_with_subclass_with_cd_treats/embeddings/$EMBEDDINGS_VERSION/models/model_0
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CCDD_with_subclass_with_cd_treats/embeddings/$EMBEDDINGS_VERSION/models/model_1

echo "Evaluating CGD_with_cd_treats models..."
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CGD_with_cd_treats/embeddings/$EMBEDDINGS_VERSION/models/model_0
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CGD_with_cd_treats/embeddings/$EMBEDDINGS_VERSION/models/model_1

echo "Evaluating CGD_with_subclass_with_cd_treats models..."
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CGD_with_subclass_with_cd_treats/embeddings/$EMBEDDINGS_VERSION/models/model_0
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CGD_with_subclass_with_cd_treats/embeddings/$EMBEDDINGS_VERSION/models/model_1

echo "✓ Evaluation complete"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Results saved to evaluation_metrics.json in each model directory."
echo ""
echo "To visualize and compare results:"
echo "  1. Run: ./run_app.sh"
echo "  2. Navigate to: http://localhost:5000"
echo "  3. Compare baseline, _with_cd_treats, and _with_cd models"
echo ""
echo "Model comparisons to analyze:"
echo "  - CCDD vs CCDD_with_cd_treats vs CCDD_with_cd"
echo "  - CCDD_with_subclass vs CCDD_with_subclass_with_cd_treats vs CCDD_with_subclass_with_cd"
echo "  - CGD vs CGD_with_cd_treats vs CGD_with_cd"
echo "  - CGD_with_subclass vs CGD_with_subclass_with_cd_treats vs CGD_with_subclass_with_cd"
echo ""
echo "Expected observation: _with_cd_treats variants should show moderate"
echo "performance improvement, while _with_cd variants show dramatic improvement,"
echo "demonstrating the impact of different levels of data leakage."
echo ""
