#!/usr/bin/env python3
"""Filter biomedical knowledge graph edges and create PecanPy input files.

This script processes nodes and edges files to create filtered graphs
for link prediction analysis.
"""
import os
import argparse
import jsonlines
import json
from pathlib import Path
from datetime import datetime

def remove_subclass_and_cid(edge, typemap):
    if edge["predicate"] == "biolink:subclass_of":
        return True
    if edge["subject"].startswith("CAID"):
        return True
    return False

def has_cd_edge(edge, typemap, treats_only=False):
    """Check if edge is between Chemical and Disease (data leakage prevention).

    Args:
        edge: The edge to check
        typemap: Node type mapping
        treats_only: If True, only return True for CD edges with treats predicates

    Returns:
        True if edge should be filtered out, False otherwise
    """
    subj = edge["subject"]
    obj = edge["object"]
    subj_types = typemap.get(subj, set())
    obj_types = typemap.get(obj, set())

    # Check if it's a Chemical-Disease edge in either direction
    if ("biolink:ChemicalEntity" in subj_types and "biolink:DiseaseOrPhenotypicFeature" in obj_types) or \
       ("biolink:DiseaseOrPhenotypicFeature" in subj_types and "biolink:ChemicalEntity" in obj_types):

        # If treats_only, check if it's a treats predicate
        if treats_only:
            treats_predicates = {
                "biolink:treats",
                "biolink:applied_to_treat"
            }
            # Return True (filter) if it's NOT a treats predicate
            return edge["predicate"] not in treats_predicates

        return True
    return False


def check_accepted(edge, typemap, accepted, cd_mode="filter_all"):
    """Check if an edge should be filtered out based on accepted patterns.

    Args:
        edge: The edge to check
        typemap: Node type mapping
        accepted: List of accepted (type1, type2) patterns
        cd_mode: How to handle CD edges - "filter_all" (default), "allow_all", or "allow_treats_only"

    Returns:
        True if edge should be filtered out, False if it should be kept
    """
    # Check for data leakage based on CD mode
    if cd_mode == "filter_all":
        if has_cd_edge(edge, typemap, treats_only=False):
            return True  # Filter out all CD edges
    elif cd_mode == "allow_treats_only":
        if has_cd_edge(edge, typemap, treats_only=True):
            return True  # Filter out non-treats CD edges
    # elif cd_mode == "allow_all": don't filter any CD edges

    subj = edge["subject"]
    obj = edge["object"]
    subj_types = typemap.get(subj, set())
    obj_types = typemap.get(obj, set())
    for acc in accepted:
        if acc[0] in subj_types and acc[1] in obj_types:
            return False
        if acc[1] in subj_types and acc[0] in obj_types:
            return False
    return True

def keep_CD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature") ]
    return check_accepted(edge, typemap, accepted)


def keep_CGD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep CC, DD, CG, GD edges but NOT CD edges (data leakage prevention)
    # Keep: Chemical-Chemical, Disease-Disease, Chemical-Gene, Gene-Disease
    # Remove: Chemical-Disease (handled by check_accepted via has_cd_edge)
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)


def keep_CDD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CCD(edge, typemap):
    # return True if you want to filter this edge out
    # Chemical-Chemical edges only (CD filtered out by check_accepted)
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity")]
    return check_accepted(edge, typemap, accepted)


def keep_CCDD(edge, typemap):
    # return True if you want to filter this edge out  
    # Chemical-Chemical and Disease-Disease edges only (CD filtered out by check_accepted)
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CCGDD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CGGD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:Gene", "biolink:Gene"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CCDD_with_subclass(edge, typemap):
    # return True if you want to filter this edge out  
    # Chemical-Chemical, Disease-Disease edges AND subclass_of edges (CD filtered out by check_accepted)
    if edge["predicate"] == "biolink:subclass_of":
        return False  # Keep subclass edges
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CGD_with_subclass(edge, typemap):
    # return True if you want to filter this edge out
    # Keep: Chemical-Chemical, Disease-Disease, Chemical-Gene, Gene-Disease AND subclass_of edges
    # Remove: Chemical-Disease (handled by check_accepted via has_cd_edge)
    if edge["predicate"] == "biolink:subclass_of":
        return False  # Keep subclass edges
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CCFD(edge, typemap):
    # return True if you want to filter this edge out  
    # Chemical-Chemical edges only, plus CF and FD fake gene edges (NO DD edges)
    # This is for use with fake genes that provide the C-F-D pathways
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity")]
    return check_accepted(edge, typemap, accepted)

def keep_CCFD_with_subclass(edge, typemap):
    # return True if you want to filter this edge out  
    # Chemical-Chemical edges with subclass relationships, plus CF and FD fake gene edges (NO DD edges)
    # This is for use with fake genes that provide the C-F-D pathways
    if edge["predicate"] == "biolink:subclass_of":
        return False  # Keep subclass edges
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity")]
    return check_accepted(edge, typemap, accepted)

def no_filter(edge, typemap):
    # return True if you want to filter this edge out
    # Keep all edges - no filtering applied
    return False


def keep_CCDD_with_cd(edge, typemap):
    """CCDD + CD edges (data leakage analysis).

    Chemical-Chemical, Disease-Disease, AND Chemical-Disease edges.
    Subclass edges are filtered out.
    """
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted, cd_mode="allow_all")


def keep_CCDD_with_subclass_with_cd(edge, typemap):
    """CCDD with subclass + CD edges (data leakage analysis).

    Chemical-Chemical, Disease-Disease, Chemical-Disease, AND subclass_of edges.
    """
    if edge["predicate"] == "biolink:subclass_of":
        return False  # Keep subclass edges
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted, cd_mode="allow_all")


def keep_CGD_with_cd(edge, typemap):
    """CGD + CD edges (data leakage analysis).

    Chemical-Chemical, Disease-Disease, Chemical-Gene, Gene-Disease, AND Chemical-Disease edges.
    Subclass edges are filtered out.
    """
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted, cd_mode="allow_all")


def keep_CGD_with_subclass_with_cd(edge, typemap):
    """CGD with subclass + CD edges (data leakage analysis).

    Chemical-Chemical, Disease-Disease, Chemical-Gene, Gene-Disease, Chemical-Disease, AND subclass_of edges.
    """
    if edge["predicate"] == "biolink:subclass_of":
        return False  # Keep subclass edges
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted, cd_mode="allow_all")


def keep_CCDD_with_cd_treats(edge, typemap):
    """CCDD + CD treats edges only (partial leakage analysis).

    Chemical-Chemical, Disease-Disease, AND Chemical-Disease treats edges.
    Non-treats CD edges are filtered out. Subclass edges are filtered out.
    """
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted, cd_mode="allow_treats_only")


def keep_CCDD_with_subclass_with_cd_treats(edge, typemap):
    """CCDD with subclass + CD treats edges only (partial leakage analysis).

    Chemical-Chemical, Disease-Disease, Chemical-Disease treats edges, AND subclass_of edges.
    Non-treats CD edges are filtered out.
    """
    if edge["predicate"] == "biolink:subclass_of":
        return False  # Keep subclass edges
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted, cd_mode="allow_treats_only")


def keep_CGD_with_cd_treats(edge, typemap):
    """CGD + CD treats edges only (partial leakage analysis).

    Chemical-Chemical, Disease-Disease, Chemical-Gene, Gene-Disease, AND Chemical-Disease treats edges.
    Non-treats CD edges are filtered out. Subclass edges are filtered out.
    """
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted, cd_mode="allow_treats_only")


def keep_CGD_with_subclass_with_cd_treats(edge, typemap):
    """CGD with subclass + CD treats edges only (partial leakage analysis).

    Chemical-Chemical, Disease-Disease, Chemical-Gene, Gene-Disease, Chemical-Disease treats edges, AND subclass_of edges.
    Non-treats CD edges are filtered out.
    """
    if edge["predicate"] == "biolink:subclass_of":
        return False  # Keep subclass edges
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted, cd_mode="allow_treats_only")


def load_indication_pairs(indications_file):
    """Load Chemical-Disease indication pairs from CSV file.
    
    Args:
        indications_file: Path to indications CSV file
        
    Returns:
        list: List of (drug_id, disease_id) tuples
    """
    import pandas as pd
    
    if not os.path.exists(indications_file):
        print(f"Warning: Indications file not found: {indications_file}")
        return []
    
    df = pd.read_csv(indications_file)
    drug_col = 'final normalized drug id'
    disease_col = 'final normalized disease id'
    
    if drug_col not in df.columns or disease_col not in df.columns:
        print(f"Warning: Expected columns '{drug_col}' and '{disease_col}' not found in {indications_file}")
        return []
    
    # Remove rows with missing values
    df_clean = df[[drug_col, disease_col]].dropna()
    indication_pairs = list(zip(df_clean[drug_col], df_clean[disease_col]))
    
    print(f"Loaded {len(indication_pairs)} indication pairs from {indications_file}")
    return indication_pairs


def create_fake_gene_edges(indication_pairs, typemap=None):
    """Create fake gene nodes and CF/FD edges for indication pairs.
    
    Args:
        indication_pairs: List of (drug_id, disease_id) tuples
        typemap: Node type mapping to validate that nodes exist in original graph (optional)
        
    Returns:
        tuple: (fake_nodes, fake_edges) where fake_nodes is list of node dicts
               and fake_edges is list of edge dicts
    """
    fake_nodes = []
    fake_edges = []
    filtered_count = 0
    
    for i, (drug_id, disease_id) in enumerate(indication_pairs):
        # If typemap provided, only create fake genes for nodes that exist in original graph
        if typemap is not None:
            if drug_id not in typemap or disease_id not in typemap:
                filtered_count += 1
                continue
        
        # Create unique fake gene ID
        fake_gene_id = f"FAKE:gene_for_{drug_id.replace(':', '_')}_{disease_id.replace(':', '_')}_{i}"
        
        # Create fake gene node
        fake_node = {
            "id": fake_gene_id,
            "name": f"Synthetic gene for {drug_id} -> {disease_id}",
            "category": ["biolink:Gene", "biolink:NamedThing"],
            "equivalent_identifiers": [fake_gene_id]
        }
        fake_nodes.append(fake_node)
        
        # Create CF edge (Chemical -> Fake gene)
        cf_edge = {
            "subject": drug_id,
            "predicate": "biolink:affects",
            "object": fake_gene_id,
            "primary_knowledge_source": "infores:synthetic",
            "knowledge_level": "knowledge_assertion",
            "agent_type": "computational_model",
            "original_subject": drug_id,
            "original_object": fake_gene_id
        }
        fake_edges.append(cf_edge)
        
        # Create FD edge (Fake gene -> Disease)
        fd_edge = {
            "subject": fake_gene_id,
            "predicate": "biolink:contributes_to",
            "object": disease_id,
            "primary_knowledge_source": "infores:synthetic",
            "knowledge_level": "knowledge_assertion",
            "agent_type": "computational_model",
            "original_subject": fake_gene_id,
            "original_object": disease_id
        }
        fake_edges.append(fd_edge)
    
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} indication pairs with nodes not in RoboKOP")
    print(f"Created {len(fake_nodes)} fake genes and {len(fake_edges)} synthetic edges")
    return fake_nodes, fake_edges

# Graph style descriptions for UI display
GRAPH_DESCRIPTIONS = {
    "no_filter": "Complete graph with all edges (no filtering applied).",
    "original": "Original graph with subclass_of and CAID edges removed.",
    "CCDD": "Chemical-Chemical and Disease-Disease edges (subclass_of edges removed).",
    "CCDD_with_subclass": "Chemical-Chemical and Disease-Disease edges with subclass_of relationships.",
    "CGD": "Chemical-Chemical, Disease-Disease, Chemical-Gene, and Gene-Disease edges (subclass_of edges removed).",
    "CGD_with_subclass": "Chemical-Chemical, Disease-Disease, Chemical-Gene, and Gene-Disease edges with subclass_of relationships.",
    "CFD": "Chemical-Chemical and Disease-Disease edges with synthetic fake genes connecting indication pairs (CCDD + fake genes).",
    "CFD_with_subclass": "Chemical-Chemical and Disease-Disease edges with synthetic fake genes and subclass relationships (CCDD_with_subclass + fake genes).",
    "CFGD": "Chemical-Gene-Disease graph with synthetic fake genes for indication pairs (CGD + fake genes).",
    "CFGD_with_subclass": "Chemical-Gene-Disease graph with synthetic fake genes and subclass relationships (CGD_with_subclass + fake genes).",
    "CCFD": "Chemical-Chemical edges with synthetic fake genes connecting indication pairs (no Disease-Disease edges).",
    "CCFD_with_subclass": "Chemical-Chemical edges with synthetic fake genes and subclass relationships (no Disease-Disease edges).",
    "CCDD_with_cd": "CCDD with Chemical-Disease edges included (data leakage analysis).",
    "CCDD_with_subclass_with_cd": "CCDD with subclass and Chemical-Disease edges included (data leakage analysis).",
    "CGD_with_cd": "CGD with Chemical-Disease edges included (data leakage analysis).",
    "CGD_with_subclass_with_cd": "CGD with subclass and Chemical-Disease edges included (data leakage analysis).",
    "CCDD_with_cd_treats": "CCDD with Chemical-Disease treats edges only (partial leakage analysis).",
    "CCDD_with_subclass_with_cd_treats": "CCDD with subclass and Chemical-Disease treats edges only (partial leakage analysis).",
    "CGD_with_cd_treats": "CGD with Chemical-Disease treats edges only (partial leakage analysis).",
    "CGD_with_subclass_with_cd_treats": "CGD with subclass and Chemical-Disease treats edges only (partial leakage analysis)."
}


def pred_trans(edge, edge_map):
    edge_key = {"predicate": edge["predicate"]}
    edge_key["subject_aspect_qualifier"] = edge.get("subject_aspect_qualifier", "")
    edge_key["object_aspect_qualifier"] = edge.get("object_aspect_qualifier", "")
    edge_key["subject_direction_qualifier"] = edge.get("subject_direction_qualifier", "")
    edge_key["object_direction_qualifier"] = edge.get("object_direction_qualifier", "")
    edge_key_string = json.dumps(edge_key, sort_keys=True)
    if edge_key_string not in edge_map:
        edge_map[edge_key_string] = f"predicate:{len(edge_map)}"
    return edge_map[edge_key_string]


def create_robokop_input(input_base_dir, 
                        nodes_filename,
                        edges_filename,
                        style="CCDD", 
                        output_dir="graphs",
                        indications_file=None):
    """Create filtered graph for analysis.
    
    Args:
        input_base_dir: Base directory containing input graph files
        nodes_filename: Filename for nodes file (e.g., 'nodes.jsonl')
        edges_filename: Filename for edges file (e.g., 'edges.jsonl')
        style: Graph style (CD, CGD, etc.)
        output_dir: Base output directory
        indications_file: Path to indications CSV file (required for fake gene styles)
    """
    # Construct full paths
    node_file = os.path.join(input_base_dir, nodes_filename)
    edges_file = os.path.join(input_base_dir, edges_filename)
    
    # Verify files exist
    if not os.path.exists(node_file):
        raise FileNotFoundError(f"Nodes file not found: {node_file}")
    if not os.path.exists(edges_file):
        raise FileNotFoundError(f"Edges file not found: {edges_file}")
    outdir = f"{output_dir}/{os.path.basename(input_base_dir.rstrip('/'))}_{style}"
    graph_dir = f"{outdir}/graph"
    
    # Determine if this is a fake gene style and what base style to use
    use_fake_genes = False
    base_filter = None
    
    if style.startswith("CF") or style.startswith("CCFD"):
        use_fake_genes = True
        if style == "CFD":
            base_filter = keep_CCDD
        elif style == "CFD_with_subclass":
            base_filter = keep_CCDD_with_subclass
        elif style == "CFGD":
            base_filter = keep_CGD
        elif style == "CFGD_with_subclass":
            base_filter = keep_CGD_with_subclass
        elif style == "CCFD":
            base_filter = keep_CCFD
        elif style == "CCFD_with_subclass":
            base_filter = keep_CCFD_with_subclass
        else:
            raise ValueError(f"Unknown fake gene style '{style}'. Available fake styles: CFD, CFD_with_subclass, CFGD, CFGD_with_subclass, CCFD, CCFD_with_subclass")
    else:
        # Regular filtering styles (existing logic)
        if style == "no_filter":
            base_filter = no_filter
        elif style == "original":
            base_filter = remove_subclass_and_cid
        elif style == "CGD":
            base_filter = keep_CGD
        elif style == "CDD":
            base_filter = keep_CDD
        elif style == "CCD":
            base_filter = keep_CCD
        elif style == "CCDD":
            base_filter = keep_CCDD
        elif style == "CCGDD":
            base_filter = keep_CCGDD
        elif style == "CGGD":
            base_filter = keep_CGGD
        elif style == "CCDD_with_subclass":
            base_filter = keep_CCDD_with_subclass
        elif style == "CGD_with_subclass":
            base_filter = keep_CGD_with_subclass
        elif style == "CCDD_with_cd":
            base_filter = keep_CCDD_with_cd
        elif style == "CCDD_with_subclass_with_cd":
            base_filter = keep_CCDD_with_subclass_with_cd
        elif style == "CGD_with_cd":
            base_filter = keep_CGD_with_cd
        elif style == "CGD_with_subclass_with_cd":
            base_filter = keep_CGD_with_subclass_with_cd
        elif style == "CCDD_with_cd_treats":
            base_filter = keep_CCDD_with_cd_treats
        elif style == "CCDD_with_subclass_with_cd_treats":
            base_filter = keep_CCDD_with_subclass_with_cd_treats
        elif style == "CGD_with_cd_treats":
            base_filter = keep_CGD_with_cd_treats
        elif style == "CGD_with_subclass_with_cd_treats":
            base_filter = keep_CGD_with_subclass_with_cd_treats
        else:
            if style not in GRAPH_DESCRIPTIONS:
                raise ValueError(f"Unknown graph style '{style}'. Available styles: {list(GRAPH_DESCRIPTIONS.keys())}")
            print("I don't know what you mean")
            return
    
    # Check if indications file is required and provided
    if use_fake_genes:
        if not indications_file:
            raise ValueError(f"Indications file is required for fake gene style '{style}'")
        if not os.path.exists(indications_file):
            raise FileNotFoundError(f"Indications file not found: {indications_file}")
    
    remove_edge = base_filter
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    
    # Load type map from nodes
    typemap = load_typemap(node_file)
    
    # Write the filtered graph
    if use_fake_genes:
        # Load indication pairs and create fake genes
        indication_pairs = load_indication_pairs(indications_file)
        fake_nodes, fake_edges = create_fake_gene_edges(indication_pairs, typemap)
        
        # Add fake genes to typemap for edge filtering
        for fake_node in fake_nodes:
            typemap[fake_node["id"]] = set(fake_node["category"])
        
        # Write the filtered graph with fake gene augmentation
        edge_count, predicate_stats = write_pecanpy_input_with_fake_genes(edges_file, graph_dir, remove_edge, typemap, 
                                                                        node_file, fake_nodes, fake_edges)
    else:
        # Standard graph filtering
        edge_count, predicate_stats = write_pecanpy_input(edges_file, graph_dir, remove_edge, typemap)
    
    # Get description for this style
    if style not in GRAPH_DESCRIPTIONS:
        raise ValueError(f"No description available for graph style '{style}'. Please add to GRAPH_DESCRIPTIONS.")
    
    # Save provenance metadata
    provenance = {
        "timestamp": datetime.now().isoformat(),
        "script": "create_robokop_input.py",
        "input_base_dir": input_base_dir,
        "nodes_filename": nodes_filename,
        "edges_filename": edges_filename,
        "input_nodes_file": node_file,
        "input_edges_file": edges_file,
        "style": style,
        "output_dir": output_dir,
        "edge_count": edge_count,
        "filter_function": remove_edge.__name__ if hasattr(remove_edge, '__name__') else str(remove_edge),
        "description": GRAPH_DESCRIPTIONS[style]
    }
    
    # Add fake gene info to provenance if applicable
    if use_fake_genes:
        provenance["indications_file"] = indications_file
        provenance["fake_nodes_created"] = len(fake_nodes)
        provenance["fake_edges_created"] = len(fake_edges)
    
    # Add predicate statistics to provenance
    provenance["predicate_counts"] = predicate_stats
    provenance["unique_predicates"] = len(predicate_stats)
    
    # Save predicate statistics to separate file for easy analysis
    predicate_stats_file = os.path.join(graph_dir, "predicate_stats.json")
    # Sort predicates by count (descending) for easier reading
    sorted_predicates = dict(sorted(predicate_stats.items(), key=lambda x: x[1], reverse=True))
    with open(predicate_stats_file, 'w') as f:
        json.dump(sorted_predicates, f, indent=2)
    
    provenance_file = os.path.join(graph_dir, "provenance.json")
    with open(provenance_file, 'w') as f:
        json.dump(provenance, f, indent=2)
    
    print(f"Predicate statistics saved: {predicate_stats_file}")
    print(f"Provenance saved: {provenance_file}")
    print(f"Graph contains {len(predicate_stats)} unique predicates")

def load_typemap(node_file):
    """Load node type mappings from nodes file.
    
    Args:
        node_file: Path to nodes.jsonl file
        
    Returns:
        dict: Mapping from node id to set of categories
    """
    typemap = {}
    with jsonlines.open(node_file) as reader:
        for node in reader:
            typemap[node["id"]] = set(node.get("category", []))
    return typemap


def write_pecanpy_input(edges_file, output_dir, filter_func, typemap):
    """Write filtered edges in PecanPy .edg format.
    
    From the pecanpy docs:
    Input format is an edgelist .edg file (node id could be int or string):
    node1_id node2_id <weight_float, optional>
    
    We use tab delimiters and CURIE node identifiers.
    
    Args:
        edges_file: Path to edges.jsonl file
        output_dir: Directory to write output files
        filter_func: Function to determine if edge should be filtered out
        typemap: Node type mapping
    
    Returns:
        tuple: (edge_count, predicate_stats) where predicate_stats is dict of predicate counts
    """
    edge_output = os.path.join(output_dir, "edges.edg")
    edge_count = 0
    predicate_stats = {}
    
    with jsonlines.open(edges_file) as reader, open(edge_output, 'w') as outf:
        for edge in reader:
            if not filter_func(edge, typemap):
                # Write in PecanPy format: subject\tobject\n
                outf.write(f"{edge['subject']}\t{edge['object']}\n")
                edge_count += 1
                
                # Count predicates
                predicate = edge.get('predicate', 'unknown')
                predicate_stats[predicate] = predicate_stats.get(predicate, 0) + 1
    
    return edge_count, predicate_stats


def write_pecanpy_input_with_fake_genes(edges_file, output_dir, filter_func, typemap, 
                                       node_file, fake_nodes, fake_edges):
    """Write filtered edges plus fake gene edges in PecanPy format.
    
    Args:
        edges_file: Path to edges.jsonl file
        output_dir: Directory to write output files
        filter_func: Function to determine if edge should be filtered out
        typemap: Node type mapping (includes fake genes)
        node_file: Path to original nodes.jsonl file
        fake_nodes: List of fake gene node dictionaries
        fake_edges: List of fake CF/FD edge dictionaries
    
    Returns:
        tuple: (edge_count, predicate_stats) where predicate_stats is dict of predicate counts
    """
    edge_output = os.path.join(output_dir, "edges.edg")
    node_output = os.path.join(output_dir, "nodes.jsonl")
    edge_count = 0
    predicate_stats = {}
    
    # Write edges file (filtered original edges + fake edges)
    with jsonlines.open(edges_file) as reader, open(edge_output, 'w') as outf:
        # First, write filtered original edges
        for edge in reader:
            if not filter_func(edge, typemap):
                outf.write(f"{edge['subject']}\t{edge['object']}\n")
                edge_count += 1
                
                # Count original predicates
                predicate = edge.get('predicate', 'unknown')
                predicate_stats[predicate] = predicate_stats.get(predicate, 0) + 1
        
        # Then, write fake gene edges
        for fake_edge in fake_edges:
            outf.write(f"{fake_edge['subject']}\t{fake_edge['object']}\n")
            edge_count += 1
            
            # Count fake predicates
            predicate = fake_edge.get('predicate', 'unknown')
            predicate_stats[predicate] = predicate_stats.get(predicate, 0) + 1
    
    # Write nodes file (original nodes + fake gene nodes)
    with jsonlines.open(node_file) as reader, jsonlines.open(node_output, 'w') as writer:
        # First, copy original nodes
        for node in reader:
            writer.write(node)
        
        # Then, write fake gene nodes
        for fake_node in fake_nodes:
            writer.write(fake_node)
    
    print(f"Written {edge_count} total edges ({edge_count - len(fake_edges)} original + {len(fake_edges)} fake)")
    print(f"Written {len(fake_nodes)} fake gene nodes")
    
    return edge_count, predicate_stats

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Create filtered graph for link prediction")
    parser.add_argument("--style", default="CCDD",
                       choices=["no_filter", "original", "CGD", "CDD", "CCD", "CCDD", "CCGDD", "CGGD",
                               "CCDD_with_subclass", "CGD_with_subclass", "CFD", "CFD_with_subclass",
                               "CFGD", "CFGD_with_subclass", "CCFD", "CCFD_with_subclass",
                               "CCDD_with_cd", "CCDD_with_subclass_with_cd", "CGD_with_cd", "CGD_with_subclass_with_cd",
                               "CCDD_with_cd_treats", "CCDD_with_subclass_with_cd_treats",
                               "CGD_with_cd_treats", "CGD_with_subclass_with_cd_treats"],
                       help="Graph filtering style (CD removed to prevent data leakage, except _with_cd variants)")
    parser.add_argument("--input-dir", default="input_graphs/robokop_base_nonredundant",
                       help="Base directory containing input graph files")
    parser.add_argument("--nodes-filename", default="nodes.jsonl",
                       help="Filename for nodes file")
    parser.add_argument("--edges-filename", default="edges.jsonl", 
                       help="Filename for edges file")
    parser.add_argument("--output-dir", default="graphs",
                       help="Base output directory")
    parser.add_argument("--indications-file", default="ground_truth/Indications List.csv",
                       help="Path to indications CSV file (required for fake gene styles CF*)")
    
    args = parser.parse_args()
    
    create_robokop_input(
        input_base_dir=args.input_dir,
        nodes_filename=args.nodes_filename,
        edges_filename=args.edges_filename,
        style=args.style,
        output_dir=args.output_dir,
        indications_file=args.indications_file
    )
    
    print(f"Graph '{args.style}' created in {args.output_dir}/{os.path.basename(args.input_dir)}_{args.style}/graph/")


if __name__ == "__main__":
    main()
