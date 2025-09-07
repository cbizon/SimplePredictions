#!/usr/bin/env python3
"""Tests for create_robokop_input module."""
import os
import tempfile
import pytest
import json
from pathlib import Path

from src.graph_modification.create_robokop_input import (
    keep_CCDD, keep_CGD, load_typemap, write_pecanpy_input, 
    check_accepted, remove_subclass_and_cid, has_cd_edge
)


@pytest.fixture
def sample_nodes():
    """Sample nodes with different categories."""
    return [
        {
            "id": "CHEBI:123",
            "name": "Test Chemical",
            "category": ["biolink:ChemicalEntity", "biolink:NamedThing"]
        },
        {
            "id": "MONDO:456", 
            "name": "Test Disease",
            "category": ["biolink:DiseaseOrPhenotypicFeature", "biolink:NamedThing"]
        },
        {
            "id": "HGNC:789",
            "name": "Test Gene", 
            "category": ["biolink:Gene", "biolink:NamedThing"]
        }
    ]


@pytest.fixture
def sample_edges():
    """Sample edges with different predicates and types."""
    return [
        {
            "subject": "CHEBI:123",
            "predicate": "biolink:treats", 
            "object": "MONDO:456"
        },
        {
            "subject": "CHEBI:123",
            "predicate": "biolink:affects",
            "object": "HGNC:789" 
        },
        {
            "subject": "MONDO:456",
            "predicate": "biolink:subclass_of",
            "object": "MONDO:999"
        },
        {
            "subject": "CAID:123",
            "predicate": "biolink:affects", 
            "object": "MONDO:456"
        }
    ]


@pytest.fixture
def temp_nodes_file(sample_nodes):
    """Create temporary nodes file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for node in sample_nodes:
            f.write(json.dumps(node) + '\n')
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_edges_file(sample_edges):
    """Create temporary edges file.""" 
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for edge in sample_edges:
            f.write(json.dumps(edge) + '\n')
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


def test_load_typemap(temp_nodes_file):
    """Test loading type mappings from nodes file."""
    typemap = load_typemap(temp_nodes_file)
    
    assert "CHEBI:123" in typemap
    assert "biolink:ChemicalEntity" in typemap["CHEBI:123"]
    assert "biolink:NamedThing" in typemap["CHEBI:123"]
    
    assert "MONDO:456" in typemap
    assert "biolink:DiseaseOrPhenotypicFeature" in typemap["MONDO:456"]
    
    assert "HGNC:789" in typemap  
    assert "biolink:Gene" in typemap["HGNC:789"]


def test_remove_subclass_and_cid():
    """Test filtering of subclass and CAID edges."""
    typemap = {}
    
    # Should remove subclass_of edges
    subclass_edge = {"predicate": "biolink:subclass_of", "subject": "A", "object": "B"}
    assert remove_subclass_and_cid(subclass_edge, typemap) == True
    
    # Should remove CAID edges
    caid_edge = {"predicate": "biolink:affects", "subject": "CAID:123", "object": "B"}  
    assert remove_subclass_and_cid(caid_edge, typemap) == True
    
    # Should keep normal edges
    normal_edge = {"predicate": "biolink:treats", "subject": "CHEBI:123", "object": "MONDO:456"}
    assert remove_subclass_and_cid(normal_edge, typemap) == False


def test_has_cd_edge():
    """Test the CD edge detection function."""
    typemap = {
        "CHEBI:123": {"biolink:ChemicalEntity"},
        "MONDO:456": {"biolink:DiseaseOrPhenotypicFeature"},
        "HGNC:789": {"biolink:Gene"}
    }
    
    # Should detect chemical-disease edge
    cd_edge = {"subject": "CHEBI:123", "object": "MONDO:456"}
    assert has_cd_edge(cd_edge, typemap) == True
    
    # Should detect disease-chemical edge
    dc_edge = {"subject": "MONDO:456", "object": "CHEBI:123"}
    assert has_cd_edge(dc_edge, typemap) == True
    
    # Should not detect chemical-gene edge
    cg_edge = {"subject": "CHEBI:123", "object": "HGNC:789"}
    assert has_cd_edge(cg_edge, typemap) == False


def test_check_accepted():
    """Test the check_accepted function with data leakage prevention."""
    typemap = {
        "CHEBI:123": {"biolink:ChemicalEntity"},
        "MONDO:456": {"biolink:DiseaseOrPhenotypicFeature"},
        "HGNC:789": {"biolink:Gene"}
    }
    
    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity")]
    
    # Should filter CD edges (data leakage prevention)
    cd_edge = {"subject": "CHEBI:123", "object": "MONDO:456"}
    assert check_accepted(cd_edge, typemap, accepted) == True
    
    # Should keep chemical-chemical edge
    cc_edge = {"subject": "CHEBI:123", "object": "CHEBI:456"}
    typemap["CHEBI:456"] = {"biolink:ChemicalEntity"}
    assert check_accepted(cc_edge, typemap, accepted) == False
    
    # Should filter chemical-gene edge (not in accepted list)
    cg_edge = {"subject": "CHEBI:123", "object": "HGNC:789"}
    assert check_accepted(cg_edge, typemap, accepted) == True


def test_keep_CCDD():
    """Test CCDD filtering (Chemical-Chemical and Disease-Disease only)."""
    typemap = {
        "CHEBI:123": {"biolink:ChemicalEntity"},
        "CHEBI:456": {"biolink:ChemicalEntity"},
        "MONDO:456": {"biolink:DiseaseOrPhenotypicFeature"},
        "MONDO:789": {"biolink:DiseaseOrPhenotypicFeature"},
        "HGNC:789": {"biolink:Gene"}
    }
    
    # Should filter subclass edges
    subclass_edge = {"predicate": "biolink:subclass_of", "subject": "A", "object": "B"}
    assert keep_CCDD(subclass_edge, typemap) == True
    
    # Should filter chemical-disease edges (data leakage prevention)
    cd_edge = {"predicate": "biolink:treats", "subject": "CHEBI:123", "object": "MONDO:456"}
    assert keep_CCDD(cd_edge, typemap) == True
    
    # Should keep chemical-chemical edges
    cc_edge = {"predicate": "biolink:interacts_with", "subject": "CHEBI:123", "object": "CHEBI:456"}
    assert keep_CCDD(cc_edge, typemap) == False
    
    # Should keep disease-disease edges
    dd_edge = {"predicate": "biolink:associated_with", "subject": "MONDO:456", "object": "MONDO:789"}
    assert keep_CCDD(dd_edge, typemap) == False
    
    # Should filter chemical-gene edges
    cg_edge = {"predicate": "biolink:affects", "subject": "CHEBI:123", "object": "HGNC:789"}
    assert keep_CCDD(cg_edge, typemap) == True


def test_keep_CGD():
    """Test CGD filtering (Chemical-Gene-Disease, but no CD due to data leakage prevention)."""
    typemap = {
        "CHEBI:123": {"biolink:ChemicalEntity"},
        "MONDO:456": {"biolink:DiseaseOrPhenotypicFeature"},
        "HGNC:789": {"biolink:Gene"}
    }
    
    # Should filter subclass edges  
    subclass_edge = {"predicate": "biolink:subclass_of", "subject": "A", "object": "B"}
    assert keep_CGD(subclass_edge, typemap) == True
    
    # Should filter chemical-disease edges (data leakage prevention)
    cd_edge = {"predicate": "biolink:treats", "subject": "CHEBI:123", "object": "MONDO:456"}
    assert keep_CGD(cd_edge, typemap) == True
    
    # Should keep chemical-gene edges  
    cg_edge = {"predicate": "biolink:affects", "subject": "CHEBI:123", "object": "HGNC:789"}
    assert keep_CGD(cg_edge, typemap) == False
    
    # Should keep gene-disease edges
    gd_edge = {"predicate": "biolink:associated_with", "subject": "HGNC:789", "object": "MONDO:456"} 
    assert keep_CGD(gd_edge, typemap) == False


def test_write_pecanpy_input(temp_edges_file, sample_nodes):
    """Test writing PecanPy format output."""
    # Create typemap
    typemap = {}
    for node in sample_nodes:
        typemap[node["id"]] = set(node["category"])
    
    with tempfile.TemporaryDirectory() as temp_dir:
        write_pecanpy_input(temp_edges_file, temp_dir, keep_CCDD, typemap)
        
        output_file = os.path.join(temp_dir, "edges.edg")
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        # Should have no edges since sample_edges doesn't have CC or DD edges
        # and CD edges are filtered out for data leakage prevention
        assert len(lines) == 0


def test_write_pecanpy_input_file_format():
    """Test that output file has correct PecanPy format with CC edges."""
    # Create test data with chemical-chemical edges
    sample_nodes = [
        {"id": "CHEBI:123", "category": ["biolink:ChemicalEntity"]},
        {"id": "CHEBI:456", "category": ["biolink:ChemicalEntity"]}
    ]
    sample_edges = [
        {"subject": "CHEBI:123", "predicate": "biolink:interacts_with", "object": "CHEBI:456"}
    ]
    
    typemap = {}
    for node in sample_nodes:
        typemap[node["id"]] = set(node["category"])
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for edge in sample_edges:
            f.write(json.dumps(edge) + '\n')
        temp_edges_file = f.name
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_pecanpy_input(temp_edges_file, temp_dir, keep_CCDD, typemap)
            
            output_file = os.path.join(temp_dir, "edges.edg")
            
            with open(output_file, 'r') as f:
                for line in f:
                    # Each line should be tab-separated with 2 fields
                    parts = line.strip().split('\t')
                    assert len(parts) == 2
                    # Both parts should be node identifiers (CURIEs)
                    assert ':' in parts[0] and ':' in parts[1]
    finally:
        os.unlink(temp_edges_file)