#!/bin/bash
set -e
export PATH="/opt/anaconda3/envs/simplepredictions/bin:$PATH"

# Data Leakage Analysis Script
# This script compares baseline graphs (with CD edges filtered) against
# their _with_cd variants (with CD edges included) to quantify the
# performance impact of data leakage.

echo "=========================================="
echo "Data Leakage Analysis Pipeline"
echo "=========================================="
echo ""
echo "This will create matched pairs of graphs to measure leakage impact:"
echo "  - CCDD vs CCDD_with_cd"
echo "  - CCDD_with_subclass vs CCDD_with_subclass_with_cd"
echo "  - CGD vs CGD_with_cd"
echo "  - CGD_with_subclass vs CGD_with_subclass_with_cd"
echo ""

# =============================================================================
# PHASE 1: Create Graphs
# =============================================================================
echo "PHASE 1: Creating graphs..."
echo "----------------------------"

# Baseline graphs (already exist, but included for completeness)
#python src/graph_modification/create_robokop_input.py --style CCDD
#python src/graph_modification/create_robokop_input.py --style CCDD_with_subclass
#python src/graph_modification/create_robokop_input.py --style CGD
#python src/graph_modification/create_robokop_input.py --style CGD_with_subclass

# Leaky variants (new)
python src/graph_modification/create_robokop_input.py --style CCDD_with_cd
python src/graph_modification/create_robokop_input.py --style CCDD_with_subclass_with_cd
python src/graph_modification/create_robokop_input.py --style CGD_with_cd
python src/graph_modification/create_robokop_input.py --style CGD_with_subclass_with_cd
#
#echo "✓ Graphs created"
#echo ""

# =============================================================================
# PHASE 2: Generate Embeddings
# =============================================================================
echo "PHASE 2: Generating embeddings..."
echo "----------------------------------"

# Standard embedding parameters for all graphs
EMBEDDING_PARAMS="--dimensions 512 --walk-length 30 --num-walks 10 --window-size 10 --p 1 --q 1"

python src/embedding/generate_embeddings.py --graph-file graphs/robokop_base_nonredundant_CCDD_with_cd/graph/edges.edg $EMBEDDING_PARAMS 
python src/embedding/generate_embeddings.py --graph-file graphs/robokop_base_nonredundant_CCDD_with_subclass_with_cd/graph/edges.edg $EMBEDDING_PARAMS 
python src/embedding/generate_embeddings.py --graph-file graphs/robokop_base_nonredundant_CGD_with_cd/graph/edges.edg $EMBEDDING_PARAMS 
python src/embedding/generate_embeddings.py --graph-file graphs/robokop_base_nonredundant_CGD_with_subclass_with_cd/graph/edges.edg $EMBEDDING_PARAMS 

#wait

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

# Use fixed random seed for reproducibility across baseline/leaky comparisons
RANDOM_SEED=42

echo "Training with random negatives..."
python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CCDD_with_cd \
    --ground-truth "$GROUND_TRUTH" \
    --embeddings-version $EMBEDDINGS_VERSION \
    --negative-ratio 1 \
    --random-state $RANDOM_SEED &

python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CCDD_with_subclass_with_cd \
    --ground-truth "$GROUND_TRUTH" \
    --embeddings-version $EMBEDDINGS_VERSION \
    --negative-ratio 1 \
    --random-state $RANDOM_SEED &

python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CGD_with_cd \
    --ground-truth "$GROUND_TRUTH" \
    --embeddings-version $EMBEDDINGS_VERSION \
    --negative-ratio 1 \
    --random-state $RANDOM_SEED &

python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CGD_with_subclass_with_cd \
    --ground-truth "$GROUND_TRUTH" \
    --embeddings-version $EMBEDDINGS_VERSION \
    --negative-ratio 1 \
    --random-state $RANDOM_SEED &

wait

echo "Training with contraindication negatives..."
python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CCDD_with_cd \
    --ground-truth "$GROUND_TRUTH" \
    --contraindications "$CONTRAINDICATIONS" \
    --embeddings-version $EMBEDDINGS_VERSION \
    --random-state $RANDOM_SEED &

python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CCDD_with_subclass_with_cd \
    --ground-truth "$GROUND_TRUTH" \
    --contraindications "$CONTRAINDICATIONS" \
    --embeddings-version $EMBEDDINGS_VERSION \
    --random-state $RANDOM_SEED &

python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CGD_with_cd \
    --ground-truth "$GROUND_TRUTH" \
    --contraindications "$CONTRAINDICATIONS" \
    --embeddings-version $EMBEDDINGS_VERSION \
    --random-state $RANDOM_SEED &

python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CGD_with_subclass_with_cd \
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

# Evaluate all leaky variant models
echo "Evaluating CCDD_with_cd models..."
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CCDD_with_cd/embeddings/$EMBEDDINGS_VERSION/models/model_0
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CCDD_with_cd/embeddings/$EMBEDDINGS_VERSION/models/model_1

echo "Evaluating CCDD_with_subclass_with_cd models..."
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CCDD_with_subclass_with_cd/embeddings/$EMBEDDINGS_VERSION/models/model_0
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CCDD_with_subclass_with_cd/embeddings/$EMBEDDINGS_VERSION/models/model_1

echo "Evaluating CGD_with_cd models..."
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CGD_with_cd/embeddings/$EMBEDDINGS_VERSION/models/model_0
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CGD_with_cd/embeddings/$EMBEDDINGS_VERSION/models/model_1

echo "Evaluating CGD_with_subclass_with_cd models..."
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CGD_with_subclass_with_cd/embeddings/$EMBEDDINGS_VERSION/models/model_0
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CGD_with_subclass_with_cd/embeddings/$EMBEDDINGS_VERSION/models/model_1

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
echo "  3. Select baseline and _with_cd models to compare"
echo ""
echo "Model pairs to compare:"
echo "  - CCDD (model_0) vs CCDD_with_cd (model_0) - Random negatives"
echo "  - CCDD (model_1) vs CCDD_with_cd (model_1) - Contraindication negatives"
echo "  - CCDD_with_subclass (model_0) vs CCDD_with_subclass_with_cd (model_0)"
echo "  - CCDD_with_subclass (model_1) vs CCDD_with_subclass_with_cd (model_1)"
echo "  - CGD (model_0) vs CGD_with_cd (model_0)"
echo "  - CGD (model_1) vs CGD_with_cd (model_1)"
echo "  - CGD_with_subclass (model_0) vs CGD_with_subclass_with_cd (model_0)"
echo "  - CGD_with_subclass (model_1) vs CGD_with_subclass_with_cd (model_1)"
echo ""
echo "Expected observation: _with_cd variants should show dramatically"
echo "higher performance, quantifying the impact of data leakage."
echo ""
