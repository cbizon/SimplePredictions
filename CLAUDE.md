# Simple Predictions

## Goal

We are working on biomedical knowledge graphs.  Our goal is to be able to make good link predictions.  In particular we are trying to predict a particular edge type, Drug-[treats]->DiseaseOrPhenotypicFeature.

## Basic Setup

* **github**: This project has a github repo at https://github.com/cbizon/SimplePredictions
* **conda**: we are using conda environment "simplepredictions" (located at /opt/anaconda3/envs/simplepredictions)
* **tests**: we are using pytest, and want to maintain high code coverage

## Environment Setup

```bash
# Create and activate conda environment
conda create -n simplepredictions python=3.11
conda activate simplepredictions

# Install core dependencies
conda install scikit-learn pandas numpy matplotlib seaborn

# Install specialized packages
pip install pecanpy jsonlines flask
```

## Key Dependencies

* **pecanpy**: for node2vec embeddings  
* **scikit-learn**: for Random Forest models
* **pandas, numpy**: data processing
* **matplotlib, seaborn**: plotting and visualization
* **jsonlines**: for JSONL file processing
* **flask**: for web application interface

## Basic Analysis Workflow

We are going to have codes that take the input graphs, modify them in some way to produce output graphs.  Then we will use node2vec to generate embeddings of the nodes, and a random forest model on the node embeddings.  We'll assess the performance of each updated graph.

## Input

The input data may never be changed. We have multiple input graph datasets in the input_graphs directory:

### Input Graph Datasets

**robokop_base_nonredundant/** (DEFAULT):
- `nodes.jsonl` (1.6GB) - Deduplicated nodes
- `edges.jsonl` (10GB) - Deduplicated edges  

**robokop_base/**:
- `robokop_base_nodes.jsonl` (1.6GB) - All nodes
- `robokop_base_edges.jsonl` (23GB) - All edges

An example node looks like this:
```json
{"id":"PUBCHEM.COMPOUND:3009304","name":"1H-1,3-Diazepine-1,3(2H)-dihexanoic acid, tetrahydro-5,6-dihydroxy-2-oxo-4,7-bis(phenylmethyl)-, (4R,5S,6S,7R)-","category":["biolink:SmallMolecule","biolink:MolecularEntity","biolink:ChemicalEntity","biolink:PhysicalEssence","biolink:ChemicalOrDrugOrTreatment","biolink:ChemicalEntityOrGeneOrGeneProduct","biolink:ChemicalEntityOrProteinOrPolypeptide","biolink:NamedThing","biolink:PhysicalEssenceOrOccurrent"],"equivalent_identifiers":["PUBCHEM.COMPOUND:3009304","CHEMBL.COMPOUND:CHEMBL29089","CAS:152928-75-1","INCHIKEY:XGEGDSLAQZJGCW-HHGOQMMWSA-N"]}
```

Note the identifier, which is the unambiguous id for the node.  Also note the node categories.  Nodes have multiple categories, which are hierarchical to a description.

An example edge looks like this:
```json
{"subject":"NCBITaxon:1661386","predicate":"biolink:subclass_of","object":"NCBITaxon:286","primary_knowledge_source":"infores:ubergraph","knowledge_level":"knowledge_assertion","agent_type":"manual_agent","original_subject":"NCBITaxon:1661386","original_object":"NCBITaxon:286"}
```

Note the subject/object/predicate structure. The subject and object are defined with identifiers that are in the nodes file.  If we want to e.g. filter edges where the subject and object have particular node types, we will need to use both files.


## Project structure
/input_graphs: where we have our initial inputs
/ground_truth/Indication\ List.csv: true positives for modeling
/ground_truth: definitions of our true edges.
/graphs: where we have the filtered or modified graphs
/graphs/{graphname}: individual modified graphs
/graphs/{graphname}/graph: node and edge files
/graphs/{graphname}/embeddings: node embedding files
/graphs/{graphname}/embeddings/{embeddings_subdirectories}: node embedding files
/graphs/{graphname}/embeddings/{embeddings_subdirectories}/models: models and results
/graphs/{graphname}/embeddings/{embeddings_subdirectories}/models/{model_subdirectories}: individual model runs and evaluations
/src
/src/graph_modification: code for modifying input_graphs to create analysis graphs
/src/embedding: codes for creating n2v embeddings from the analysis graphs
/src/modeling: codes for running RF or other models from the embeddings
/tests: pytests for checking the code.
/scripts: scripts for controlling the process

## Directory Structure and Provenance

**Hierarchical Organization**: Models are nested under embeddings at `/graphs/{graphname}/embeddings/{embeddings_version}/models/{model_version}/`

**Graph Directory** (`/graphs/{graphname}/graph/`):
- `edges.edg`: PecanPy format edge list 
- `nodes.jsonl`: Node metadata (includes fake genes for CFD styles)
- `provenance.json`: Graph creation metadata and parameters
- `predicate_stats.json`: Predicate type counts sorted by frequency

**Model Directory** (`/graphs/{graphname}/embeddings/{version}/models/{model_version}/`):
- `rf_model.pkl`: Trained Random Forest model
- `provenance.json`: Complete model metadata including parameters and training info
- `training_pairs.json`: Exact training pairs used (for data leakage prevention)
- `evaluation_metrics.json`: Comprehensive evaluation results
- `results.json`: Basic training performance metrics
- `classification_report.txt`: Detailed classification metrics

**Provenance Tracking**: Every step saves complete metadata for reproducibility:
- Graph filtering parameters and edge counts
- Embedding generation parameters and node counts  
- Model training parameters and data splits
- Evaluation parameters and comprehensive metrics

## Analysis

To perform node2vec we are going to use pecanpy, which is installed in our conda environment.
Information about pecanpy can be found at https://github.com/krishnanlab/PecanPy

Our base parameters for pecanpy are: --dimensions 512 --walk-length 30 --num-walks 10 --window-size 10 --p 1 --q 1
We may change them but this is the default.

## Data Leakage Prevention

**CRITICAL**: We want to be sure that the modeling does not suffer from data leakage. So in our graphs we want to remove all Chemical/DiseaseOrPhenotypicFeature edges during training. This is implemented in the `has_cd_edge()` function.

## Graph Types and Filtering Styles

**Base Styles:**
- **CCDD**: Chemical-Chemical + Disease-Disease edges only
- **CGD**: Chemical-Gene-Disease pathways (no direct CD edges)  
- **CCD**: Chemical-Chemical edges only
- **CDD**: Disease-Disease edges only

**Subclass Variants:**
- **CCDD_with_subclass**: CCDD + biolink:subclass_of relationships
- **CGD_with_subclass**: CGD + biolink:subclass_of relationships

**Synthetic Fake Gene Styles (Upper Bounds):**
- **CFD**: CCDD + synthetic fake genes connecting known indications
- **CFD_with_subclass**: CFD + biolink:subclass_of relationships
- **CFGD**: CGD + synthetic fake genes
- **CFGD_with_subclass**: CFGD + biolink:subclass_of relationships

**CFD Synthetic Pathways**: For each known drug-disease indication pair, creates:
1. Unique fake gene: `FAKE:gene_for_{drug}_{disease}_{index}`
2. CF edge: Chemical → Fake Gene (biolink:affects)
3. FD edge: Fake Gene → Disease (biolink:contributes_to)
4. Perfect synthetic pathway establishing performance upper bounds

## Ground Truth Data

For model building, we are using the data in `ground_truth/Indications List.csv`
It looks like this:

```
final normalized drug id,final normalized drug label,final normalized disease id,final normalized disease label,drug|disease,FDA,EMA,PMDA
CHEBI:8327,Polythiazide,MONDO:0005009,congestive heart failure,CHEBI:8327|MONDO:0005009,true,,
CHEBI:8327,Polythiazide,MONDO:0005155,liver cirrhosis,CHEBI:8327|MONDO:0005155,true,,
```

## Contraindications Data

We also have contraindications data in `ground_truth/Contraindications List.csv` which can be used as negative examples instead of generating random negatives. This file has the same format as the indications list:

```
final normalized drug id,final normalized drug label,final normalized disease id,final normalized disease label,drug|disease,...
CHEBI:8327,Polythiazide,MONDO:0002476,anuria,CHEBI:8327|MONDO:0002476,...
```

## Machine Learning Pipeline

1. **Graph Creation**: Filter input graphs to create analysis graphs (CCDD, CGD, etc.)
2. **Embeddings**: Generate node2vec embeddings using PecanPy (512 dimensions)
3. **Training**: Use Random Forest with 80/20 train/test split. Negative sampling can use either:
   - Random generation from drug/disease permutations (original approach)
   - Contraindications as true negatives (new approach)
4. **Training Pairs Storage**: Save exact training pairs to `training_pairs.json` in model directory
5. **Evaluation**: Ranking-based metrics on comprehensive drug-disease combination space

## Evaluation Methodology

**Data Leakage Prevention**: Training pairs are saved during model training and read during evaluation to ensure no overlap between training and evaluation sets.

**Evaluation Set Construction**:
- Generate ALL possible drug-disease combinations from ground truth universe
- Exclude training pairs (both positive and negative) 
- Use zero-padding for missing embeddings to ensure consistent evaluation across models
- Evaluate on ~1M combinations per model with logarithmic K sampling

**Metrics Calculated**:
- **Precision@K**: Accuracy of top K predictions  
- **Recall@K**: Fraction of test positives found in top K (max = 1.0)
- **Total Recall@K**: Fraction of all discoverable indications found (accounts for embedding coverage limits)
- **Hits@K**: Fraction of diseases with at least one hit in top K

**Total Recall Context**: 
- Denominator = Original indications - Training positives used
- Shows realistic performance bounds given embedding coverage constraints
- Theoretical maximum shown as dashed line on plots (e.g., ~0.11 for CCDD)

## Web Application for Model Visualization

The project includes a Flask web application (`app.py`) for interactive visualization and comparison of model evaluation results.

### Project Structure - Webapp Components
```
├── app.py                     # Flask web application
├── templates/
│   └── index.html            # Web interface HTML template  
├── run_app.sh                # Script to launch webapp
└── graphs/                   # Models discovered automatically
    └── **/evaluation_metrics.json  # Files webapp searches for
```

### Webapp Features

**Automatic Model Discovery**: Recursively finds all models with `evaluation_metrics.json` files in the graphs/ directory and organizes them hierarchically by:
- Graph type (e.g., robokop_base_nonredundant_CCDD)
- Embedding version (e.g., embeddings_0)  
- Model version (e.g., model_0)

**Interactive Visualization**: Generates 4×2 grid of evaluation plots:
- **Rows**: Precision@K, Recall@K, Total Recall@K, Hits@K
- **Columns**: K range 1-1000 (zoomed view) and 1-Max (full range)
- **Multiple model comparison** on same plots with different colors
- **Model metadata** extracted from provenance files

**Usage**:
```bash
# Launch webapp (use conda environment)
./run_app.sh
# Navigate to http://localhost:5000

# Or run directly:
python app.py
```

## Key Scripts

**Core Pipeline**:
- `src/graph_modification/create_robokop_input.py`: Graph filtering with data leakage prevention and predicate analysis
- `src/embedding/generate_embeddings.py`: Node2vec embedding generation using PecanPy
- `src/modeling/train_model.py`: Random Forest training with training pairs storage. Supports contraindications via `--contraindications` flag
- `src/modeling/evaluate_model.py`: Simplified evaluation interface using model metadata. Only requires `--model-dir` argument

**Web Interface**:
- `app.py`: Flask web application for interactive model visualization and comparison
- `run_app.sh`: Script to launch webapp with conda environment

## Step-by-Step Pipeline Usage

### 1. Graph Creation
```bash
# Create CCDD graph (baseline)
python src/graph_modification/create_robokop_input.py \
    --style CCDD \
    --input-dir input_graphs/robokop_base_nonredundant \
    --output-dir graphs

# Create CFD graph with synthetic fake genes (upper bounds)
python src/graph_modification/create_robokop_input.py \
    --style CFD \
    --input-dir input_graphs/robokop_base_nonredundant \
    --indications-file "ground_truth/Indications List.csv" \
    --output-dir graphs
```

### 2. Generate Embeddings
```bash
python src/embedding/generate_embeddings.py \
    --graph-file graphs/robokop_base_nonredundant_CCDD/graph/edges.edg \
    --dimensions 512 \
    --walk-length 30 \
    --num-walks 10 \
    --window-size 10 \
    --p 1 \
    --q 1
```

### 3. Train Models
```bash
# Train with random negatives
python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CCDD \
    --ground-truth "ground_truth/Indications List.csv" \
    --embeddings-version embeddings_0 \
    --negative-ratio 1

# Train with contraindications as negatives
python src/modeling/train_model.py \
    --graph-dir graphs/robokop_base_nonredundant_CCDD \
    --ground-truth "ground_truth/Indications List.csv" \
    --contraindications "ground_truth/Contraindications List.csv" \
    --embeddings-version embeddings_0
```

### 4. Evaluate Models
```bash
python src/modeling/evaluate_model.py \
    --model-dir graphs/robokop_base_nonredundant_CCDD/embeddings/embeddings_0/models/model_0
```


## ***RULES OF THE ROAD***

Don't use mocks. 

Ask clarifying questions

Do not implement bandaids - treat the root cause of problems

Once we have a test, do not delete it without explicit permission.  

Do not return made up results if an API fails.  Let it fail.

When changing code, don't make duplicate functions - just change the function. We can always roll back changes if needed.

Keep the directories clean, don't leave a bunch of junk laying around.

When making commit or PR messages, do not mention authorship. Do not advertise for yourself.
