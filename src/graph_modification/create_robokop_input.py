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


def keep_human_only(edge, typemap, nonhuman_ids=None, organism_taxon_ids=None, nonhuman_reactome_ids=None, nonhuman_disease_ids=None):
    """Filter to keep only human content.

    Filters out:
    1. Any edge with OrganismTaxon as subject or object
    2. Any edge with non-human Gene or Protein as subject or object
    3. Any edge with non-human Reactome pathway as subject or object
    4. Any edge with non-human animal disease as subject or object

    Args:
        edge: The edge to check
        typemap: Node type mapping
        nonhuman_ids: Set of non-human gene/protein IDs to filter out
        organism_taxon_ids: Set of OrganismTaxon node IDs to filter out
        nonhuman_reactome_ids: Set of non-human Reactome pathway IDs to filter out
        nonhuman_disease_ids: Set of non-human animal disease IDs to filter out

    Returns:
        True if edge should be filtered out, False otherwise
    """
    if nonhuman_ids is None:
        nonhuman_ids = set()
    if organism_taxon_ids is None:
        organism_taxon_ids = set()
    if nonhuman_reactome_ids is None:
        nonhuman_reactome_ids = set()
    if nonhuman_disease_ids is None:
        nonhuman_disease_ids = set()

    subj = edge["subject"]
    obj = edge["object"]

    # Filter out edges involving OrganismTaxon nodes
    if subj in organism_taxon_ids or obj in organism_taxon_ids:
        return True

    # Filter out edges involving non-human genes/proteins
    if subj in nonhuman_ids or obj in nonhuman_ids:
        return True

    # Filter out edges involving non-human Reactome pathways
    if subj in nonhuman_reactome_ids or obj in nonhuman_reactome_ids:
        return True

    # Filter out edges involving non-human animal diseases
    if subj in nonhuman_disease_ids or obj in nonhuman_disease_ids:
        return True

    return False


def keep_no_text_mined(edge, typemap):
    """Filter out text-mined edges.

    Filters out any edge with agent_type="text_mining_agent"

    Args:
        edge: The edge to check
        typemap: Node type mapping (not used but required for consistency)

    Returns:
        True if edge should be filtered out, False otherwise
    """
    agent_type = edge.get("agent_type", "")
    if agent_type == "text_mining_agent":
        return True
    return False


def keep_human_only_no_text_mined(edge, typemap, nonhuman_ids=None, organism_taxon_ids=None, nonhuman_reactome_ids=None, nonhuman_disease_ids=None):
    """Filter to keep only human content without text-mined edges.

    Combines human_only and no_text_mined filters.

    Filters out:
    1. Any edge with OrganismTaxon as subject or object
    2. Any edge with non-human genes/proteins as subject or object
    3. Any edge with non-human Reactome pathways as subject or object
    4. Any edge with non-human animal diseases as subject or object
    5. Any edge with agent_type="text_mining_agent"

    Args:
        edge: The edge to check
        typemap: Node type mapping
        nonhuman_ids: Set of non-human gene/protein IDs to filter out
        organism_taxon_ids: Set of OrganismTaxon node IDs to filter out
        nonhuman_reactome_ids: Set of non-human Reactome pathway IDs to filter out
        nonhuman_disease_ids: Set of non-human animal disease IDs to filter out

    Returns:
        True if edge should be filtered out, False otherwise
    """
    # Check human-only filters first
    if keep_human_only(edge, typemap, nonhuman_ids, organism_taxon_ids, nonhuman_reactome_ids, nonhuman_disease_ids):
        return True

    # Check text mining filter
    if keep_no_text_mined(edge, typemap):
        return True

    return False


def keep_human_only_no_text_mined_no_low_degree(edge, typemap, nonhuman_ids=None, organism_taxon_ids=None, nonhuman_reactome_ids=None, nonhuman_disease_ids=None, low_degree_nodes=None):
    """Filter to keep only human content without text-mined edges and without low-degree nodes.

    This is a wrapper that applies human_only and no_text_mined filters.
    The low-degree filtering happens in a second pass after initial filtering.

    Args:
        edge: The edge to check
        typemap: Node type mapping
        nonhuman_ids: Set of non-human gene/protein IDs to filter out
        organism_taxon_ids: Set of OrganismTaxon node IDs to filter out
        nonhuman_reactome_ids: Set of non-human Reactome pathway IDs to filter out
        nonhuman_disease_ids: Set of non-human animal disease IDs to filter out
        low_degree_nodes: Set of nodes with degree <= 2 to filter out (used in second pass)

    Returns:
        True if edge should be filtered out, False otherwise
    """
    # First pass: apply human_only and no_text_mined filters
    if keep_human_only_no_text_mined(edge, typemap, nonhuman_ids, organism_taxon_ids, nonhuman_reactome_ids, nonhuman_disease_ids):
        return True

    # Second pass: filter low-degree nodes (if provided)
    if low_degree_nodes is not None:
        subj = edge["subject"]
        obj = edge["object"]
        if subj in low_degree_nodes or obj in low_degree_nodes:
            return True

    return False


def multi_filter_1(edge, typemap, nonhuman_ids=None, organism_taxon_ids=None, nonhuman_reactome_ids=None, nonhuman_disease_ids=None, low_degree_nodes=None):
    """Multi-filter combining human_only, no_text_mined, low-degree removal, BindingDB affinity filtering, and NCIT subclass filtering.

    First pass filters:
    1. Non-human content (OrganismTaxon, non-human genes/proteins, non-human Reactome pathways, non-human animal diseases)
    2. Text-mined edges
    3. BindingDB affects edges with affinity < 7
    4. NCIT-to-NCIT subclass_of relationships

    Second pass filters:
    5. Nodes with degree <= 2

    Args:
        edge: The edge to check
        typemap: Node type mapping
        nonhuman_ids: Set of non-human gene/protein IDs to filter out
        organism_taxon_ids: Set of OrganismTaxon node IDs to filter out
        nonhuman_reactome_ids: Set of non-human Reactome pathway IDs to filter out
        nonhuman_disease_ids: Set of non-human animal disease IDs to filter out
        low_degree_nodes: Set of nodes with degree <= 2 to filter out (used in second pass)

    Returns:
        True if edge should be filtered out, False otherwise
    """
    # First pass filters
    # 1. Apply human_only and no_text_mined filters
    if keep_human_only_no_text_mined(edge, typemap, nonhuman_ids, organism_taxon_ids, nonhuman_reactome_ids, nonhuman_disease_ids):
        return True

    # 2. Filter BindingDB affects edges with affinity < 7
    if edge.get("predicate") == "biolink:affects":
        # Check if this is from BindingDB
        primary_ks = edge.get("primary_knowledge_source", "")
        if "bindingdb" in primary_ks.lower():
            # Check for affinity attribute
            affinity = edge.get("affinity")
            if affinity is not None:
                try:
                    affinity_value = float(affinity)
                    if affinity_value < 7:
                        return True  # Filter out low affinity
                except (ValueError, TypeError):
                    pass

    # 3. Filter subclass_of edges between two NCIT nodes
    if edge.get("predicate") == "biolink:subclass_of":
        subj = edge.get("subject", "")
        obj = edge.get("object", "")
        if subj.startswith("NCIT:") and obj.startswith("NCIT:"):
            return True  # Filter out NCIT-to-NCIT subclass relationships

    # Second pass: filter low-degree nodes (if provided)
    if low_degree_nodes is not None:
        subj = edge["subject"]
        obj = edge["object"]
        if subj in low_degree_nodes or obj in low_degree_nodes:
            return True

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
    "human_only": "Human-only graph with OrganismTaxon edges and non-human genes/proteins filtered out.",
    "no_text_mined": "Graph with text-mined edges (agent_type=text_mining_agent) removed.",
    "human_only_no_text_mined": "Human-only graph with text-mined edges removed (combines human_only + no_text_mined).",
    "human_only_no_text_mined_no_low_degree": "Human-only graph with text-mined edges removed, plus removal of degree 1-2 nodes (two-pass filtering).",
    "multi_filter_1": "Multi-filter: human-only + no text-mining + BindingDB affinity<7 removed + NCIT-to-NCIT subclass_of removed + degree 1-2 nodes removed (two-pass filtering).",
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
    "CGD_with_subclass_with_cd_treats": "CGD with subclass and Chemical-Disease treats edges only (partial leakage analysis).",
    "multi_filter_1_hgt": "Multi-filter: human-only + no text-mining + BindingDB affinity<7 removed + NCIT-to-NCIT subclass_of removed + degree 1-2 nodes removed + CD edges kept. TSV output for DGL with pseudo-predicates from qualifiers."
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
                        indications_file=None,
                        output_format="pecanpy"):
    """Create filtered graph for analysis.

    Args:
        input_base_dir: Base directory containing input graph files
        nodes_filename: Filename for nodes file (e.g., 'nodes.jsonl')
        edges_filename: Filename for edges file (e.g., 'edges.jsonl')
        style: Graph style (CD, CGD, etc.)
        output_dir: Base output directory
        indications_file: Path to indications CSV file (required for fake gene styles)
        output_format: Output format - 'pecanpy', 'hgt', or 'kgx' (default: 'pecanpy')
    """
    # Construct full paths
    node_file = os.path.join(input_base_dir, nodes_filename)
    edges_file = os.path.join(input_base_dir, edges_filename)

    # Verify files exist
    if not os.path.exists(node_file):
        raise FileNotFoundError(f"Nodes file not found: {node_file}")
    if not os.path.exists(edges_file):
        raise FileNotFoundError(f"Edges file not found: {edges_file}")

    # Determine output directory based on format
    # KGX format writes to input_graphs directory, others write to graphs directory
    if output_format == "kgx":
        base_output_dir = os.path.dirname(input_base_dir.rstrip('/'))  # Get parent of input_base_dir
        outdir = f"{base_output_dir}/{os.path.basename(input_base_dir.rstrip('/'))}_{style}"
        graph_dir = outdir  # KGX writes directly to output dir, no /graph subdirectory
    else:
        outdir = f"{output_dir}/{os.path.basename(input_base_dir.rstrip('/'))}_{style}"
        graph_dir = f"{outdir}/graph"
    
    # Determine if this is a fake gene style and what base style to use
    use_fake_genes = False
    use_human_filter = False
    use_two_pass = False
    use_hgt_output = (output_format == "hgt")
    use_kgx_output = (output_format == "kgx")
    base_filter = None
    filter_kwargs = {}

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
        elif style == "human_only":
            base_filter = keep_human_only
            use_human_filter = True
        elif style == "no_text_mined":
            base_filter = keep_no_text_mined
        elif style == "human_only_no_text_mined":
            base_filter = keep_human_only_no_text_mined
            use_human_filter = True
        elif style == "human_only_no_text_mined_no_low_degree":
            base_filter = keep_human_only_no_text_mined_no_low_degree
            use_human_filter = True
            use_two_pass = True  # Flag for two-pass filtering
        elif style == "multi_filter_1":
            base_filter = multi_filter_1
            use_human_filter = True
            use_two_pass = True  # Flag for two-pass filtering
        elif style == "multi_filter_1_hgt":
            base_filter = multi_filter_1_with_cd
            use_human_filter = True
            use_two_pass = True
            use_hgt_output = True
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

    # Handle human-only filtering if needed
    if use_human_filter:
        try:
            from src.graph_modification.name_resolver import identify_nonhuman_genes_proteins
        except ModuleNotFoundError:
            # Try relative import when running as script
            import sys
            from pathlib import Path
            # Add project root to path
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from src.graph_modification.name_resolver import identify_nonhuman_genes_proteins

        print("\n=== Human-only filtering ===")
        print("Step 1: Collecting OrganismTaxon nodes...")
        organism_taxon_ids = collect_organism_taxon_nodes(node_file)

        print("\nStep 2: Identifying non-human genes and proteins...")
        nonhuman_ids = identify_nonhuman_genes_proteins(node_file)

        print("\nStep 3: Identifying non-human Reactome pathways...")
        nonhuman_reactome_ids = collect_nonhuman_reactome_pathways(node_file)

        print("\nStep 4: Identifying non-human animal diseases...")
        nonhuman_disease_ids = collect_nonhuman_animal_diseases(node_file)

        filter_kwargs["organism_taxon_ids"] = organism_taxon_ids
        filter_kwargs["nonhuman_ids"] = nonhuman_ids
        filter_kwargs["nonhuman_reactome_ids"] = nonhuman_reactome_ids
        filter_kwargs["nonhuman_disease_ids"] = nonhuman_disease_ids
        print(f"\nTotal nodes to be filtered: {len(organism_taxon_ids) + len(nonhuman_ids) + len(nonhuman_reactome_ids) + len(nonhuman_disease_ids)}")

    # Write the filtered graph
    if use_fake_genes:
        # Load indication pairs and create fake genes
        indication_pairs = load_indication_pairs(indications_file)
        fake_nodes, fake_edges = create_fake_gene_edges(indication_pairs, typemap)

        # Add fake genes to typemap for edge filtering
        for fake_node in fake_nodes:
            typemap[fake_node["id"]] = set(fake_node["category"])

        # Write the filtered graph with fake gene augmentation
        # Note: Fake genes currently only support PecanPy format
        if use_kgx_output or use_hgt_output:
            raise ValueError(f"Fake gene styles (CF*, CCFD*) are not yet supported with output_format='{output_format}'. Use 'pecanpy' format.")
        edge_count, predicate_stats = write_pecanpy_input_with_fake_genes(edges_file, graph_dir, remove_edge, typemap,
                                                                        node_file, fake_nodes, fake_edges)
    else:
        # Standard graph filtering
        if use_kgx_output:
            edge_count, predicate_stats = write_kgx_output(edges_file, node_file, graph_dir, remove_edge, typemap, filter_kwargs)
        elif use_hgt_output:
            edge_count, predicate_stats = write_hgt_input(edges_file, node_file, graph_dir, remove_edge, typemap, filter_kwargs)
        else:
            edge_count, predicate_stats = write_pecanpy_input(edges_file, graph_dir, remove_edge, typemap, filter_kwargs)

    # Two-pass filtering for low-degree node removal
    if use_two_pass:
        print("\n=== Two-pass filtering: removing low-degree nodes ===")
        from collections import defaultdict

        # Read the just-written edges file to calculate degrees
        # Different formats use different extensions: .jsonl (KGX), .tsv (HGT), .edg (PecanPy)
        if use_kgx_output:
            edges_temp = os.path.join(graph_dir, "edges.jsonl")
        elif use_hgt_output:
            edges_temp = os.path.join(graph_dir, "edges.tsv")
        else:
            edges_temp = os.path.join(graph_dir, "edges.edg")

        print(f"Pass 1 complete. Edges: {edge_count}")
        print("Calculating node degrees from Pass 1 output...")

        node_degrees = defaultdict(int)

        if use_kgx_output:
            # Read JSONL format
            with jsonlines.open(edges_temp) as reader:
                for edge in reader:
                    node_degrees[edge['subject']] += 1
                    node_degrees[edge['object']] += 1
        else:
            # Read TSV or tab-delimited format
            with open(edges_temp, 'r') as f:
                # Skip header line if HGT output (TSV format)
                if use_hgt_output:
                    next(f)  # Skip header

                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        node_degrees[parts[0]] += 1
                        node_degrees[parts[1]] += 1

        # Find low-degree nodes (degree 1 or 2)
        low_degree_nodes = {node for node, degree in node_degrees.items() if degree <= 2}
        print(f"Found {len(low_degree_nodes):,} nodes with degree <= 2")

        # Add low_degree_nodes to filter_kwargs for second pass
        filter_kwargs["low_degree_nodes"] = low_degree_nodes

        # Second pass: re-filter with low-degree node filtering
        print("Pass 2: Re-filtering with low-degree nodes removed...")
        if use_kgx_output:
            edge_count, predicate_stats = write_kgx_output(edges_file, node_file, graph_dir, remove_edge, typemap, filter_kwargs)
        elif use_hgt_output:
            edge_count, predicate_stats = write_hgt_input(edges_file, node_file, graph_dir, remove_edge, typemap, filter_kwargs)
        else:
            edge_count, predicate_stats = write_pecanpy_input(edges_file, graph_dir, remove_edge, typemap, filter_kwargs)
        print(f"Pass 2 complete. Final edges: {edge_count}")

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


def collect_organism_taxon_nodes(node_file):
    """Collect all OrganismTaxon node IDs from nodes file.

    Args:
        node_file: Path to nodes.jsonl file

    Returns:
        set: Set of node IDs that are OrganismTaxon nodes
    """
    organism_taxon_ids = set()
    with jsonlines.open(node_file) as reader:
        for node in reader:
            categories = set(node.get("category", []))
            if "biolink:OrganismTaxon" in categories:
                organism_taxon_ids.add(node["id"])

    print(f"Found {len(organism_taxon_ids)} OrganismTaxon nodes")
    return organism_taxon_ids


def collect_nonhuman_reactome_pathways(node_file):
    """Collect non-human Reactome pathway IDs from nodes file.

    Reactome pathways (IDs starting with REACT) are species-specific,
    with species indicated in the name in parentheses at the end.
    We want to filter out all except those ending with "(Homo sapiens)".

    Args:
        node_file: Path to nodes.jsonl file

    Returns:
        set: Set of non-human Reactome pathway node IDs to filter out
    """
    nonhuman_reactome_ids = set()
    with jsonlines.open(node_file) as reader:
        for node in reader:
            node_id = node.get("id", "")
            # Check if this is a Reactome pathway
            if node_id.startswith("REACT"):
                name = node.get("name", "")
                # Keep only if it ends with (Homo sapiens)
                if not name.endswith("(Homo sapiens)"):
                    nonhuman_reactome_ids.add(node_id)

    print(f"Found {len(nonhuman_reactome_ids)} non-human Reactome pathways")
    return nonhuman_reactome_ids


def collect_nonhuman_animal_diseases(node_file):
    """Collect non-human animal disease IDs from nodes file.

    Diseases with MONDO_SUPERCLASS_non-human_animal_disease: true
    are animal-specific diseases that should be filtered out.

    Args:
        node_file: Path to nodes.jsonl file

    Returns:
        set: Set of non-human animal disease node IDs to filter out
    """
    nonhuman_disease_ids = set()
    with jsonlines.open(node_file) as reader:
        for node in reader:
            # Check if this node has the non-human animal disease property
            if node.get("MONDO_SUPERCLASS_non-human_animal_disease") is True:
                nonhuman_disease_ids.add(node["id"])

    print(f"Found {len(nonhuman_disease_ids)} non-human animal diseases")
    return nonhuman_disease_ids


def write_pecanpy_input(edges_file, output_dir, filter_func, typemap, filter_kwargs=None):
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
        filter_kwargs: Additional keyword arguments to pass to filter_func

    Returns:
        tuple: (edge_count, predicate_stats) where predicate_stats is dict of predicate counts
    """
    if filter_kwargs is None:
        filter_kwargs = {}

    edge_output = os.path.join(output_dir, "edges.edg")
    edge_count = 0
    predicate_stats = {}

    with jsonlines.open(edges_file) as reader, open(edge_output, 'w') as outf:
        for edge in reader:
            if not filter_func(edge, typemap, **filter_kwargs):
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


def multi_filter_1_with_cd(edge, typemap, nonhuman_ids=None, organism_taxon_ids=None, nonhuman_reactome_ids=None, nonhuman_disease_ids=None, low_degree_nodes=None):
    """Multi-filter like multi_filter_1 but KEEPS Chemical-Disease edges (no CD filtering).

    Filters applied:
    1. Non-human content (OrganismTaxon, non-human genes/proteins, non-human Reactome pathways, non-human animal diseases)
    2. Text-mined edges
    3. BindingDB affects edges with affinity < 7
    4. NCIT-to-NCIT subclass_of relationships
    5. Nodes with degree <= 2 (second pass)

    DOES NOT filter Chemical-Disease edges (unlike multi_filter_1).

    Args:
        edge: The edge to check
        typemap: Node type mapping
        nonhuman_ids: Set of non-human gene/protein IDs to filter out
        organism_taxon_ids: Set of OrganismTaxon node IDs to filter out
        nonhuman_reactome_ids: Set of non-human Reactome pathway IDs to filter out
        nonhuman_disease_ids: Set of non-human animal disease IDs to filter out
        low_degree_nodes: Set of nodes with degree <= 2 to filter out (used in second pass)

    Returns:
        True if edge should be filtered out, False otherwise
    """
    # 1. Apply human_only and no_text_mined filters (but not CD filtering)
    # We need to do this manually instead of calling keep_human_only_no_text_mined

    # Check human-only filters
    subj = edge.get("subject")
    obj = edge.get("object")

    # Filter OrganismTaxon nodes
    if organism_taxon_ids:
        if subj in organism_taxon_ids or obj in organism_taxon_ids:
            return True

    # Filter non-human genes/proteins
    if nonhuman_ids:
        if subj in nonhuman_ids or obj in nonhuman_ids:
            return True

    # Filter non-human Reactome pathways
    if nonhuman_reactome_ids:
        if subj in nonhuman_reactome_ids or obj in nonhuman_reactome_ids:
            return True

    # Filter non-human animal diseases
    if nonhuman_disease_ids:
        if subj in nonhuman_disease_ids or obj in nonhuman_disease_ids:
            return True

    # Check text mining filter
    agent_type = edge.get("agent_type", "")
    if agent_type == "text_mining_agent":
        return True

    # 2. Filter BindingDB affects edges with affinity < 7
    if edge.get("predicate") == "biolink:affects":
        primary_ks = edge.get("primary_knowledge_source", "")
        if "bindingdb" in primary_ks.lower():
            affinity = edge.get("affinity")
            if affinity is not None:
                try:
                    affinity_value = float(affinity)
                    if affinity_value < 7:
                        return True
                except (ValueError, TypeError):
                    pass

    # 3. Filter subclass_of edges between two NCIT nodes
    if edge.get("predicate") == "biolink:subclass_of":
        if subj.startswith("NCIT:") and obj.startswith("NCIT:"):
            return True

    # 4. Second pass: filter low-degree nodes (if provided)
    if low_degree_nodes is not None:
        if subj in low_degree_nodes or obj in low_degree_nodes:
            return True

    return False


def write_kgx_output(edges_file, nodes_file, output_dir, filter_func, typemap, filter_kwargs=None):
    """Write filtered edges in KGX JSONL format (original format preserved).

    Output format:
    - nodes.jsonl: Original node format, filtered to only nodes appearing in edges
    - edges.jsonl: Original edge format, filtered based on filter_func

    Args:
        edges_file: Path to edges.jsonl file
        nodes_file: Path to nodes.jsonl file
        output_dir: Directory to write output files
        filter_func: Function to determine if edge should be filtered out
        typemap: Node type mapping
        filter_kwargs: Additional keyword arguments to pass to filter_func

    Returns:
        tuple: (edge_count, predicate_stats)
    """
    if filter_kwargs is None:
        filter_kwargs = {}

    edges_output = os.path.join(output_dir, "edges.jsonl")
    nodes_output = os.path.join(output_dir, "nodes.jsonl")

    edge_count = 0
    predicate_stats = {}
    nodes_in_edges = set()

    # Process edges - write filtered edges in original JSONL format
    with jsonlines.open(edges_file) as reader, jsonlines.open(edges_output, 'w') as writer:
        for edge in reader:
            if not filter_func(edge, typemap, **filter_kwargs):
                # Write edge in original format
                writer.write(edge)
                edge_count += 1

                # Track nodes
                nodes_in_edges.add(edge['subject'])
                nodes_in_edges.add(edge['object'])

                # Count predicates
                predicate = edge.get('predicate', 'unknown')
                predicate_stats[predicate] = predicate_stats.get(predicate, 0) + 1

    # Write nodes file - only nodes that appear in filtered edges
    with jsonlines.open(nodes_file) as reader, jsonlines.open(nodes_output, 'w') as writer:
        for node in reader:
            node_id = node.get("id")
            if node_id in nodes_in_edges:
                # Write node in original format
                writer.write(node)

    print(f"Written {edge_count} edges to {edges_output}")
    print(f"Written {len(nodes_in_edges)} nodes to {nodes_output}")

    return edge_count, predicate_stats


def write_hgt_input(edges_file, nodes_file, output_dir, filter_func, typemap, filter_kwargs=None):
    """Write filtered edges in HGT TSV format with pseudo-predicates from qualifiers.

    Output format:
    - nodes.tsv: id, type (most specific biolink type)
    - edges.tsv: subject, predicate, object

    Pseudo-predicates are created from qualifiers:
    - {predicate}-{direction}-{aspect} when both qualifiers present
    - {predicate}-{direction} when only direction qualifier
    - {predicate}-{aspect} when only aspect qualifier
    - {predicate} when no qualifiers

    Args:
        edges_file: Path to edges.jsonl file
        nodes_file: Path to nodes.jsonl file
        output_dir: Directory to write output files
        filter_func: Function to determine if edge should be filtered out
        typemap: Node type mapping
        filter_kwargs: Additional keyword arguments to pass to filter_func

    Returns:
        tuple: (edge_count, predicate_stats)
    """
    if filter_kwargs is None:
        filter_kwargs = {}

    # Import type_utils for get_most_specific_type
    try:
        from src.graph_modification.type_utils import get_most_specific_type
    except ModuleNotFoundError:
        import sys
        from pathlib import Path as PathLib
        project_root = PathLib(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.graph_modification.type_utils import get_most_specific_type

    edges_output = os.path.join(output_dir, "edges.tsv")
    nodes_output = os.path.join(output_dir, "nodes.tsv")

    edge_count = 0
    predicate_stats = {}
    nodes_in_edges = set()

    # Process edges
    with jsonlines.open(edges_file) as reader, open(edges_output, 'w') as outf:
        # Write header
        outf.write("subject\tpredicate\tobject\n")

        for edge in reader:
            if not filter_func(edge, typemap, **filter_kwargs):
                subj = edge['subject']
                obj = edge['object']

                # Track nodes
                nodes_in_edges.add(subj)
                nodes_in_edges.add(obj)

                # Get base predicate (strip biolink: prefix)
                predicate = edge.get('predicate', 'related_to')
                predicate_base = predicate.split(":")[-1] if ":" in predicate else predicate

                # Get qualifiers (direct fields, not in array)
                direction = edge.get("object_direction_qualifier")
                aspect = edge.get("object_aspect_qualifier")

                # Strip prefixes from qualifier values
                if direction and ":" in direction:
                    direction = direction.split(":")[-1]
                if aspect and ":" in aspect:
                    aspect = aspect.split(":")[-1]

                # Build pseudo-predicate
                if direction and aspect:
                    pseudo_predicate = f"{predicate_base}-{direction}-{aspect}"
                elif direction:
                    pseudo_predicate = f"{predicate_base}-{direction}"
                elif aspect:
                    pseudo_predicate = f"{predicate_base}-{aspect}"
                else:
                    pseudo_predicate = predicate_base

                # Write edge
                outf.write(f"{subj}\t{pseudo_predicate}\t{obj}\n")
                edge_count += 1

                # Count predicates
                predicate_stats[pseudo_predicate] = predicate_stats.get(pseudo_predicate, 0) + 1

    # Write nodes file
    with jsonlines.open(nodes_file) as reader, open(nodes_output, 'w') as outf:
        # Write header
        outf.write("id\ttype\n")

        for node in reader:
            node_id = node.get("id")
            if node_id in nodes_in_edges:
                # Get most specific type
                categories = node.get("category", [])
                node_type = get_most_specific_type(categories) if categories else "Unknown"
                outf.write(f"{node_id}\t{node_type}\n")

    print(f"Written {edge_count} edges to {edges_output}")
    print(f"Written {len(nodes_in_edges)} nodes to {nodes_output}")

    return edge_count, predicate_stats


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Create filtered graph for link prediction")
    parser.add_argument("--style", default="CCDD",
                       choices=["no_filter", "original", "human_only", "no_text_mined", "human_only_no_text_mined",
                               "human_only_no_text_mined_no_low_degree", "multi_filter_1", "multi_filter_1_hgt",
                               "CGD", "CDD", "CCD", "CCDD", "CCGDD", "CGGD",
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
                       help="Base output directory (ignored for KGX format, which writes to input_graphs)")
    parser.add_argument("--output-format", default="pecanpy",
                       choices=["pecanpy", "hgt", "kgx"],
                       help="Output format: 'pecanpy' (.edg), 'hgt' (.tsv), or 'kgx' (.jsonl)")
    parser.add_argument("--indications-file", default="ground_truth/Indications List.csv",
                       help="Path to indications CSV file (required for fake gene styles CF*)")

    args = parser.parse_args()

    create_robokop_input(
        input_base_dir=args.input_dir,
        nodes_filename=args.nodes_filename,
        edges_filename=args.edges_filename,
        style=args.style,
        output_dir=args.output_dir,
        indications_file=args.indications_file,
        output_format=args.output_format
    )

    # Print output location based on format
    if args.output_format == "kgx":
        base_output = os.path.dirname(args.input_dir.rstrip('/'))
        output_location = f"{base_output}/{os.path.basename(args.input_dir)}_{args.style}/"
    else:
        output_location = f"{args.output_dir}/{os.path.basename(args.input_dir)}_{args.style}/graph/"

    print(f"Graph '{args.style}' created in {output_location}")


if __name__ == "__main__":
    main()

