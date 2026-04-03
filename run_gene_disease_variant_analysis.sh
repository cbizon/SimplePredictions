#!/bin/bash
set -euo pipefail

export PATH="/opt/anaconda3/envs/simplepredictions/bin:$PATH"

INPUT_DIR="${INPUT_DIR:-input_graphs/robokop_23e}"
OUTPUT_DIR="${OUTPUT_DIR:-graphs}"
GROUND_TRUTH_CSV="${GROUND_TRUTH_CSV:-ground_truth/robokop_23e_gene_disease_gold_standard.csv}"
GROUND_TRUTH_SUMMARY="${GROUND_TRUTH_SUMMARY:-analysis_results/robokop_23e_gene_disease_gold_standard_summary.json}"
EMBEDDINGS_VERSION="${EMBEDDINGS_VERSION:-embeddings_0}"
SEEDS_STRING="${SEEDS:-42}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-1}"
EMBEDDING_MODE="${EMBEDDING_MODE:-FirstOrderUnweighted}"
EMBEDDING_WORKERS="${EMBEDDING_WORKERS:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

GRAPH_STYLES=(
  "GD_no_variants"
  "GD_variant_gene_only"
  "GD_variant_disease_only"
  "GD_variant_full"
  "GD_variant_full_no_nearby"
  "GD_variant_full_with_gd"
)

EMBEDDING_PARAMS=(
  --dimensions 512
  --walk-length 30
  --num-walks 10
  --window-size 10
  --p 1
  --q 1
  --mode "$EMBEDDING_MODE"
  --workers "$EMBEDDING_WORKERS"
)

echo "=========================================="
echo "Gene-Disease Variant Analysis Pipeline"
echo "=========================================="
echo "Input graph: $INPUT_DIR"
echo "Gold standard CSV: $GROUND_TRUTH_CSV"
echo "Graph styles: ${GRAPH_STYLES[*]}"
echo "Seeds: $SEEDS_STRING"
echo "Skip existing outputs: $SKIP_EXISTING"
echo "Embedding mode: $EMBEDDING_MODE"
echo "Embedding workers: $EMBEDDING_WORKERS"
echo ""

echo "PHASE 1: Extracting gene-disease gold standard"
if [[ "$SKIP_EXISTING" == "1" && -f "$GROUND_TRUTH_CSV" && -f "$GROUND_TRUTH_SUMMARY" ]]; then
  echo "Skipping PHASE 1; outputs already exist."
else
  python src/analysis/extract_gene_disease_gold_standard.py \
    --input-dir "$INPUT_DIR" \
    --output-csv "$GROUND_TRUTH_CSV" \
    --summary-json "$GROUND_TRUTH_SUMMARY"
fi
echo ""

echo "PHASE 2: Creating filtered graph cohorts"
for style in "${GRAPH_STYLES[@]}"; do
  graph_file="$OUTPUT_DIR/$(basename "$INPUT_DIR")_${style}/graph/edges.edg"
  if [[ "$SKIP_EXISTING" == "1" && -f "$graph_file" ]]; then
    echo "Skipping graph style: $style"
    continue
  fi

  echo "Creating graph style: $style"
  python src/graph_modification/create_robokop_input.py \
    --style "$style" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR"
done
echo ""

echo "PHASE 3: Generating embeddings"
for style in "${GRAPH_STYLES[@]}"; do
  graph_file="$OUTPUT_DIR/$(basename "$INPUT_DIR")_${style}/graph/edges.edg"
  embeddings_file="$OUTPUT_DIR/$(basename "$INPUT_DIR")_${style}/embeddings/$EMBEDDINGS_VERSION/embeddings.npz"

  if [[ "$SKIP_EXISTING" == "1" && -f "$embeddings_file" ]]; then
    echo "Skipping embeddings for: $style"
    continue
  fi

  echo "Generating embeddings for: $style"
  python src/embedding/generate_embeddings.py \
    --graph-file "$graph_file" \
    --version-name "$EMBEDDINGS_VERSION" \
    "${EMBEDDING_PARAMS[@]}"
done
echo ""

echo "PHASE 4: Training models"
for style in "${GRAPH_STYLES[@]}"; do
  graph_dir="$OUTPUT_DIR/$(basename "$INPUT_DIR")_${style}"
  model_root="$graph_dir/embeddings/$EMBEDDINGS_VERSION/models"
  for seed in $SEEDS_STRING; do
    if [[ "$SKIP_EXISTING" == "1" && -d "$model_root" ]]; then
      existing_model="$(rg -l "\"random_state\": $seed" "$model_root"/model_*/provenance.json 2>/dev/null | head -n 1 || true)"
      if [[ -n "$existing_model" ]]; then
        echo "Skipping training $style seed $seed"
        continue
      fi
    fi

    echo "Training $style with seed $seed"
    python src/modeling/train_model.py \
      --graph-dir "$graph_dir" \
      --ground-truth "$GROUND_TRUTH_CSV" \
      --embeddings-version "$EMBEDDINGS_VERSION" \
      --negative-ratio "$NEGATIVE_RATIO" \
      --random-state "$seed" \
      --source-column gene_id \
      --target-column disease_id \
      --source-label gene \
      --target-label disease
  done
done
echo ""

echo "PHASE 5: Evaluating models"
for style in "${GRAPH_STYLES[@]}"; do
  model_root="$OUTPUT_DIR/$(basename "$INPUT_DIR")_${style}/embeddings/$EMBEDDINGS_VERSION/models"
  for model_dir in "$model_root"/model_*; do
    if [ -d "$model_dir" ]; then
      if [[ "$SKIP_EXISTING" == "1" && -f "$model_dir/evaluation_metrics.json" ]]; then
        echo "Skipping evaluation $(basename "$model_dir") for $style"
        continue
      fi

      echo "Evaluating $(basename "$model_dir") for $style"
      python src/modeling/evaluate_model.py --model-dir "$model_dir"
    fi
  done
done
echo ""

echo "Pipeline complete."
echo "Compare the following cohorts:"
echo "  GD_no_variants vs GD_variant_gene_only vs GD_variant_disease_only vs GD_variant_full"
echo "  GD_variant_full vs GD_variant_full_no_nearby"
echo "  GD_variant_full vs GD_variant_full_with_gd"
