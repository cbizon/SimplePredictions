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

def has_cd_edge(edge, typemap):
    """Check if edge is between Chemical and Disease (data leakage prevention)."""
    subj = edge["subject"]
    obj = edge["object"]
    subj_types = typemap.get(subj, set())
    obj_types = typemap.get(obj, set())
    
    # Check if it's a Chemical-Disease edge in either direction
    if ("biolink:ChemicalEntity" in subj_types and "biolink:DiseaseOrPhenotypicFeature" in obj_types) or \
       ("biolink:DiseaseOrPhenotypicFeature" in subj_types and "biolink:ChemicalEntity" in obj_types):
        return True
    return False


def check_accepted(edge, typemap, accepted):
    # First check for data leakage - always filter out CD edges
    if has_cd_edge(edge, typemap):
        return True  # Filter out CD edges to prevent data leakage
        
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
                        output_dir="graphs"):
    """Create filtered graph for analysis.
    
    Args:
        input_base_dir: Base directory containing input graph files
        nodes_filename: Filename for nodes file (e.g., 'nodes.jsonl')
        edges_filename: Filename for edges file (e.g., 'edges.jsonl')
        style: Graph style (CD, CGD, etc.)
        output_dir: Base output directory
    """
    # Construct full paths
    node_file = os.path.join(input_base_dir, nodes_filename)
    edges_file = os.path.join(input_base_dir, edges_filename)
    
    # Verify files exist
    if not os.path.exists(node_file):
        raise FileNotFoundError(f"Nodes file not found: {node_file}")
    if not os.path.exists(edges_file):
        raise FileNotFoundError(f"Edges file not found: {edges_file}")
    outdir = f"{output_dir}/{os.path.basename(input_base_dir)}_{style}"
    graph_dir = f"{outdir}/graph"
    if style == "original":
        # This filters the edges by
        # 1) removing all subclass_of and
        # 2) removing all edges with a subject that starts with "CAID"
        remove_edge = remove_subclass_and_cid
    elif style == "CGD":
        # This keeps any edges between
        #   chemicals and genes
        #   genes and diseases
        #   chemicals and diseases
        # removes subclass edges
        remove_edge = keep_CGD
    elif style == "CDD":
        # No subclasses, no CD edges (data leakage prevention)
        # only disease/disease edges
        remove_edge = keep_CDD
    elif style == "CCD":
        # No subclasses, no CD edges (data leakage prevention)
        # only chemical/chemical edges
        remove_edge = keep_CCD
    elif style == "CCDD":
        # No subclasses, no CD edges (data leakage prevention)
        # chemical/chemical and disease/disease edges
        remove_edge = keep_CCDD
    elif style == "CCGDD":
        # No subclasses, no CD edges (data leakage prevention)
        # chemical/chemical, gene/gene, and disease/disease edges
        remove_edge = keep_CCGDD
    elif style == "CGGD":
        # No subclasses
        # only chemical/disease edges and disease/disease edges
        remove_edge = keep_CGGD
    else:
        print("I don't know what you mean")
        return
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    
    # Load type map from nodes
    typemap = load_typemap(node_file)
    
    # Write the filtered graph
    edge_count = write_pecanpy_input(edges_file, graph_dir, remove_edge, typemap)
    
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
        "description": f"Filtered {style} graph with data leakage prevention (CD edges removed)"
    }
    
    provenance_file = os.path.join(graph_dir, "provenance.json")
    with open(provenance_file, 'w') as f:
        json.dump(provenance, f, indent=2)
    
    print(f"Provenance saved: {provenance_file}")

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
        int: Number of edges written to output file
    """
    edge_output = os.path.join(output_dir, "edges.edg")
    edge_count = 0
    
    with jsonlines.open(edges_file) as reader, open(edge_output, 'w') as outf:
        for edge in reader:
            if not filter_func(edge, typemap):
                # Write in PecanPy format: subject\tobject\n
                outf.write(f"{edge['subject']}\t{edge['object']}\n")
                edge_count += 1
    
    return edge_count

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Create filtered graph for link prediction")
    parser.add_argument("--style", default="CCDD", 
                       choices=["original", "CGD", "CDD", "CCD", "CCDD", "CCGDD", "CGGD"],
                       help="Graph filtering style (CD removed to prevent data leakage)")
    parser.add_argument("--input-dir", default="input_graphs/robokop_base_nonredundant",
                       help="Base directory containing input graph files")
    parser.add_argument("--nodes-filename", default="nodes.jsonl",
                       help="Filename for nodes file")
    parser.add_argument("--edges-filename", default="edges.jsonl", 
                       help="Filename for edges file")
    parser.add_argument("--output-dir", default="graphs",
                       help="Base output directory")
    
    args = parser.parse_args()
    
    create_robokop_input(
        input_base_dir=args.input_dir,
        nodes_filename=args.nodes_filename,
        edges_filename=args.edges_filename,
        style=args.style,
        output_dir=args.output_dir
    )
    
    print(f"Graph '{args.style}' created in {args.output_dir}/{os.path.basename(args.input_dir)}/{args.style}/")


if __name__ == "__main__":
    main()
