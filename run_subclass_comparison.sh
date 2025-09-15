#!/bin/bash
set -e
export PATH="/opt/anaconda3/envs/simplepredictions/bin:$PATH"

# Create new graphs with subclasses
#python src/graph_modification/create_robokop_input.py --style CCDD_with_subclass
#python src/graph_modification/create_robokop_input.py --style CGD_with_subclass

# Generate embeddings
#python src/embedding/generate_embeddings.py --graph-file graphs/robokop_base_nonredundant_CCDD_with_subclass/graph/edges.edg &
#python src/embedding/generate_embeddings.py --graph-file graphs/robokop_base_nonredundant_CGD_with_subclass/graph/edges.edg &
#wait

# Train models with inferred negatives
#python src/modeling/train_model.py --graph-dir graphs/robokop_base_nonredundant_CCDD_with_subclass --ground-truth "ground_truth/Indications List.csv" --embeddings-version embeddings_0 &
#python src/modeling/train_model.py --graph-dir graphs/robokop_base_nonredundant_CGD_with_subclass --ground-truth "ground_truth/Indications List.csv" --embeddings-version embeddings_0 &
#python src/modeling/train_model.py --graph-dir graphs/robokop_base_nonredundant_CCDD --ground-truth "ground_truth/Indications List.csv" --embeddings-version embeddings_0 &
#wait

# Train models with contraindications
#python src/modeling/train_model.py --graph-dir graphs/robokop_base_nonredundant_CCDD_with_subclass --ground-truth "ground_truth/Indications List.csv" --contraindications "ground_truth/Contraindications List.csv" --embeddings-version embeddings_0 &
#python src/modeling/train_model.py --graph-dir graphs/robokop_base_nonredundant_CGD_with_subclass --ground-truth "ground_truth/Indications List.csv" --contraindications "ground_truth/Contraindications List.csv" --embeddings-version embeddings_0 &
#python src/modeling/train_model.py --graph-dir graphs/robokop_base_nonredundant_CCDD --ground-truth "ground_truth/Indications List.csv" --contraindications "ground_truth/Contraindications List.csv" --embeddings-version embeddings_0 &
#wait

# Evaluate models - CCDD vs CCDD_with_subclass (inferred negatives)
python src/modeling/evaluate_model.py --model-dirs graphs/robokop_base_nonredundant_CCDD/models/model_1 graphs/robokop_base_nonredundant_CCDD_with_subclass/models/model_0 --model-labels CCDD CCDD_subclass

# Evaluate models - CGD vs CGD_with_subclass (inferred negatives)  
python src/modeling/evaluate_model.py --model-dirs graphs/robokop_base_nonredundant_CGD/models/model_0 graphs/robokop_base_nonredundant_CGD_with_subclass/models/model_0 --model-labels CGD CGD_subclass

# Evaluate models - CCDD vs CCDD_with_subclass (contraindications)
python src/modeling/evaluate_model.py --model-dirs graphs/robokop_base_nonredundant_CCDD/models/model_2 graphs/robokop_base_nonredundant_CCDD_with_subclass/models/model_1 --model-labels CCDD_contra CCDD_subclass_contra

# Evaluate models - CGD vs CGD_with_subclass (contraindications)
python src/modeling/evaluate_model.py --model-dirs graphs/robokop_base_nonredundant_CGD/models/model_1 graphs/robokop_base_nonredundant_CGD_with_subclass/models/model_1 --model-labels CGD_contra CGD_subclass_contra

echo "Done!"
