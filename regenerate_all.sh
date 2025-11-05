#!/bin/bash
set -e
export PATH="/opt/anaconda3/envs/simplepredictions/bin:$PATH"

# Clean up
#rm -rf graphs/robokop_base_nonredundant_*

# Create graphs
#python src/graph_modification/create_robokop_input.py --style CCDD
#python src/graph_modification/create_robokop_input.py --style CGD

# Generate embeddings
#python src/embedding/generate_embeddings.py --graph-file graphs/robokop_base_nonredundant_CCDD/graph/edges.edg &
#python src/embedding/generate_embeddings.py --graph-file graphs/robokop_base_nonredundant_CGD/graph/edges.edg &
#wait

# Train models
#python src/modeling/train_model.py --graph-dir graphs/robokop_base_nonredundant_CCDD --ground-truth "ground_truth/Indications List.csv" --embeddings-version embeddings_0 &
#python src/modeling/train_model.py --graph-dir graphs/robokop_base_nonredundant_CGD --ground-truth "ground_truth/Indications List.csv" --embeddings-version embeddings_0 &
#python src/modeling/train_model.py --graph-dir graphs/robokop_base_nonredundant_CCDD --ground-truth "ground_truth/Indications List.csv" --contraindications "ground_truth/Contraindications List.csv" --embeddings-version embeddings_0 &
#python src/modeling/train_model.py --graph-dir graphs/robokop_base_nonredundant_CGD --ground-truth "ground_truth/Indications List.csv" --contraindications "ground_truth/Contraindications List.csv" --embeddings-version embeddings_0 &
#python src/modeling/train_model.py --graph-dir graphs/robokop_base_nonredundant_CCDD_with_subclass --ground-truth "ground_truth/Indications List.csv" --embeddings-version embeddings_0
#python src/modeling/train_model.py --graph-dir graphs/robokop_base_nonredundant_CGD_with_subclass --ground-truth "ground_truth/Indications List.csv" --embeddings-version embeddings_0
#python src/modeling/train_model.py --graph-dir graphs/robokop_base_nonredundant_CCDD_with_subclass --ground-truth "ground_truth/Indications List.csv" --contraindications "ground_truth/Contraindications List.csv" --embeddings-version embeddings_0
#python src/modeling/train_model.py --graph-dir graphs/robokop_base_nonredundant_CGD_with_subclass --ground-truth "ground_truth/Indications List.csv" --contraindications "ground_truth/Contraindications List.csv" --embeddings-version embeddings_0

# Evaluate individual models (new single-model evaluation)
# Note: Models are now in embeddings/embeddings_0/models/ and save metrics directly to model directories

# Uncomment these lines after training models:
#python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CCDD/embeddings/embeddings_0/models/model_0
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CGD/embeddings/embeddings_0/models/model_0
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CCDD/embeddings/embeddings_0/models/model_1  
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CGD/embeddings/embeddings_0/models/model_1

python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CCDD_with_subclass/embeddings/embeddings_0/models/model_0
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CGD_with_subclass/embeddings/embeddings_0/models/model_0
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CCDD_with_subclass/embeddings/embeddings_0/models/model_1  
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base_nonredundant_CGD_with_subclass/embeddings/embeddings_0/models/model_1

echo "Evaluations now save metrics to individual model directories as evaluation_metrics.json"

echo "Done!"
