# Simple Predictions

## Goal

We are working on biomedical knowledge graphs.  Our goal is to be able to make good link predictions.  In particular we are trying to predict a particular edge type, Drug-[treats]->DiseaseOrPhenotypicFeature.

## Basic Setup

* github: This project has a github repo at https://github.com/cbizon/SimplePredictions
* conda: we are using conda environment "simplepredictions" (located at /opt/anaconda3/envs/simplepredictions)
* tests: we are using pytest, and want to maintain high code coverage

## Key Dependencies

* pecanpy: for node2vec embeddings  
* scikit-learn: for Random Forest models
* pandas, numpy: data processing
* matplotlib, seaborn: plotting and visualization
* jsonlines: for JSONL file processing

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
/graphs/{graphname}/models: models and results
/graphs/{graphname}/models/{models subdirectories}: models and results
/evaluations/{evaluations subdirectories}
/src
/src/graph_modification: code for modifying input_graphs to create analysis graphs
/src/embedding: codes for creating n2v embeddings from the analysis graphs
/src/modeling: codes for running RF or other models from the embeddings
/tests: pytests for checking the code.
/scripts: scripts for controlling the process

Note the subdirectories for embeddings and models and evaluations.  These are directories related to each run of something. So for instance in embeddings, there might be embeddings\_0 holding the embeddings for the first embeddings calculated for that graph.  Those subdirectories also contain provenance files explaining where they came from.

## Analysis

To perform node2vec we are going to use pecanpy, which is installed in our conda environment.
Information about pecanpy can be found at https://github.com/krishnanlab/PecanPy

Our base parameters for pecanpy are: --dimensions 512 --walk-length 30 --num-walks 10 --window-size 10 --p 1 --q 1
We may change them but this is the default.

## Data Leakage Prevention

**CRITICAL**: We want to be sure that the modeling does not suffer from data leakage. So in our graphs we want to remove all Chemical/DiseaseOrPhenotypicFeature edges during training. This is implemented in the `has_cd_edge()` function.

Current graph types:
- **CCDD**: Chemical-Chemical + Disease-Disease edges only (implemented)
- **CGD**: Chemical-Gene-Disease edges (no direct CD) (planned)

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

## Key Scripts

- `scripts/run_ccdd_analysis.sh`: Create CCDD graph and generate embeddings
- `scripts/train_ccdd_model.sh`: Train Random Forest model  
- `src/graph_modification/create_robokop_input.py`: Graph filtering with data leakage prevention
- `src/modeling/train_model.py`: Random Forest training with training pairs storage. Supports contraindications via `--contraindications` flag
- `src/modeling/evaluate_model.py`: Simplified evaluation interface using model metadata. Only requires `--model-dir` argument

## Usage Examples

**Training with random negatives:**
```bash
python src/modeling/train_model.py --graph-dir graphs/robokop_base/CCDD --ground-truth "ground_truth/Indications List.csv"
```

**Training with contraindications as negatives:**
```bash
python src/modeling/train_model.py --graph-dir graphs/robokop_base/CCDD --ground-truth "ground_truth/Indications List.csv" --contraindications "ground_truth/Contraindications List.csv"
```

**Evaluation (simplified interface):**
```bash
python src/modeling/evaluate_model.py --model-dir graphs/robokop_base/CCDD/models/model_2
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
