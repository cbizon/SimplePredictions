# Gene-Disease Variant Experiments

This experiment suite tests whether sequence variants in `robokop_23e` help or hurt recovery of held-out direct gene-disease edges.

## Gold Standard

The gold standard is extracted from `robokop_23e` itself:

- keep direct `Gene`-`DiseaseOrPhenotypicFeature` edges
- drop `biolink:subclass_of`
- drop edges with `agent_type=text_mining_agent`
- collapse to unique `(gene_id, disease_id)` pairs

The extractor is:

```bash
python src/analysis/extract_gene_disease_gold_standard.py \
  --input-dir input_graphs/robokop_23e \
  --output-csv ground_truth/robokop_23e_gene_disease_gold_standard.csv \
  --summary-json analysis_results/robokop_23e_gene_disease_gold_standard_summary.json
```

## Graph Cohorts

All cohorts remove:

- `subclass_of`
- text-mined edges
- BindingDB `biolink:affects` edges with `affinity < 7`
- degree-1 nodes in a second pass

- `GD_no_variants`: also remove direct gene-disease edges and all variant edges
- `GD_variant_gene_only`: keep only SequenceVariant-Gene edges from the variant subgraph
- `GD_variant_disease_only`: keep only SequenceVariant-Disease edges from the variant subgraph
- `GD_variant_full`: keep SequenceVariant-Gene and SequenceVariant-Disease edges
- `GD_variant_full_no_nearby`: same as `GD_variant_full`, but remove `biolink:is_nearby_variant_of`
- `GD_variant_full_with_gd`: leakage sentinel; same as `GD_variant_full`, but keep direct gene-disease edges

## End-to-End Runner

```bash
./run_gene_disease_variant_analysis.sh
```

Useful environment variables:

```bash
INPUT_DIR=input_graphs/robokop_23e
GROUND_TRUTH_CSV=ground_truth/robokop_23e_gene_disease_gold_standard.csv
SEEDS="42 43 44"
NEGATIVE_RATIO=1
```

## Primary Comparisons

- `GD_no_variants` vs `GD_variant_full`
- `GD_variant_gene_only` vs `GD_variant_disease_only`
- `GD_variant_full` vs `GD_variant_full_no_nearby`
- `GD_variant_full` vs `GD_variant_full_with_gd`
