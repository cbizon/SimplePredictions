#!/bin/bash
set -e
export PATH="/opt/anaconda3/envs/simplepredictions/bin:$PATH"

# Usage: ./generate_all.sh <filter_style>
# Example: ./generate_all.sh human_only
# Example: ./generate_all.sh no_text_mined
# Example: ./generate_all.sh human_only_no_text_mined

FILTER_STYLE=$1

# Validate filter style
VALID_FILTERS="multi_filter_1 no_filter original human_only no_text_mined human_only_no_text_mined_no_low_degree human_only_no_text_mined CGD CDD CCD CCDD CCGDD CGGD CCDD_with_subclass CGD_with_subclass CFD CFD_with_subclass CFGD CFGD_with_subclass CCFD CCFD_with_subclass CCDD_with_cd CCDD_with_subclass_with_cd CGD_with_cd CGD_with_subclass_with_cd CCDD_with_cd_treats CCDD_with_subclass_with_cd_treats CGD_with_cd_treats CGD_with_subclass_with_cd_treats"

if [[ ! " $VALID_FILTERS " =~ " $FILTER_STYLE " ]]; then
    echo "Error: Invalid filter style '$FILTER_STYLE'"
    echo "Valid options: $VALID_FILTERS"
    exit 1
fi

echo "=== Running pipeline with filter style: $FILTER_STYLE ==="

# Create graphs
echo "Step 1: Creating graph..."
#python src/graph_modification/create_robokop_input.py --style "$FILTER_STYLE" --input-dir input_graphs/robokop_base_nonredundant --output-dir graphs

# Generate embeddings
echo "Step 2: Generating embeddings..."
#python src/embedding/generate_embeddings.py --graph-file graphs/robokop_base_nonredundant_"$FILTER_STYLE"/graph/edges.edg

# Train models
echo "Step 3: Training model..."
#python src/modeling/train_model.py --graph-dir graphs/robokop_base_nonredundant_"$FILTER_STYLE" --ground-truth "ground_truth/Indications List.csv" --embeddings-version embeddings_2

# Evaluate individual models
echo "Step 4: Evaluating model..."
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_"$FILTER_STYLE"/embeddings/embeddings_1/models/model_0 --shap-top-k 0
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_"$FILTER_STYLE"/embeddings/embeddings_2/models/model_0 --shap-top-k 0

echo ""
echo "=== Pipeline complete for filter style: $FILTER_STYLE ==="
echo "Evaluation metrics saved to: graphs/robokop_base_nonredundant_${FILTER_STYLE}/embeddings/embeddings_0/models/model_0/evaluation_metrics.json"
echo ""
echo "Done!"
