#!/usr/bin/env python3
"""Tests for create_robokop_input module."""
import os
import tempfile
import pytest
import json
import jsonlines
from pathlib import Path

from src.graph_modification.create_robokop_input import (
    keep_CCDD, keep_CGD, load_typemap, write_pecanpy_input,
    check_accepted, remove_subclass_and_cid, has_cd_edge, create_robokop_input,
    keep_CCDD_with_cd, keep_CCDD_with_subclass_with_cd,
    keep_CGD_with_cd, keep_CGD_with_subclass_with_cd,
    keep_human_only, collect_organism_taxon_nodes,
    keep_no_text_mined, keep_human_only_no_text_mined,
    multi_filter_1_with_cd, write_hgt_input, write_kgx_output
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
        edge_count, predicate_stats = write_pecanpy_input(temp_edges_file, temp_dir, keep_CCDD, typemap)

        output_file = os.path.join(temp_dir, "edges.edg")
        assert os.path.exists(output_file)

        with open(output_file, 'r') as f:
            lines = f.readlines()

        # Should have no edges since sample_edges doesn't have CC or DD edges
        # and CD edges are filtered out for data leakage prevention
        assert len(lines) == 0
        assert edge_count == 0
        assert predicate_stats == {}


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
            edge_count, predicate_stats = write_pecanpy_input(temp_edges_file, temp_dir, keep_CCDD, typemap)

            output_file = os.path.join(temp_dir, "edges.edg")

            with open(output_file, 'r') as f:
                lines = list(f)
                assert edge_count == len(lines), f"Edge count {edge_count} doesn't match file lines {len(lines)}"
                assert predicate_stats == {"biolink:interacts_with": 1}
                for line in lines:
                    # Each line should be tab-separated with 2 fields
                    parts = line.strip().split('\t')
                    assert len(parts) == 2
                    # Both parts should be node identifiers (CURIEs)
                    assert ':' in parts[0] and ':' in parts[1]
    finally:
        os.unlink(temp_edges_file)


def test_create_robokop_input_new_interface():
    """Test the updated create_robokop_input function with new interface."""
    # Create temporary input directory structure
    with tempfile.TemporaryDirectory() as temp_base:
        input_dir = os.path.join(temp_base, "input")
        os.makedirs(input_dir)
        
        # Create test nodes and edges files
        sample_nodes = [
            {"id": "CHEBI:123", "category": ["biolink:ChemicalEntity"]},
            {"id": "CHEBI:456", "category": ["biolink:ChemicalEntity"]}
        ]
        sample_edges = [
            {"subject": "CHEBI:123", "predicate": "biolink:interacts_with", "object": "CHEBI:456"}
        ]
        
        # Write test files
        nodes_file = os.path.join(input_dir, "test_nodes.jsonl")
        edges_file = os.path.join(input_dir, "test_edges.jsonl")
        
        with open(nodes_file, 'w') as f:
            for node in sample_nodes:
                f.write(json.dumps(node) + '\n')
                
        with open(edges_file, 'w') as f:
            for edge in sample_edges:
                f.write(json.dumps(edge) + '\n')
        
        # Test the new interface
        output_dir = os.path.join(temp_base, "output")
        create_robokop_input(
            input_base_dir=input_dir,
            nodes_filename="test_nodes.jsonl",
            edges_filename="test_edges.jsonl", 
            style="CCDD",
            output_dir=output_dir
        )
        
        # Verify output was created (check actual output path from function)
        expected_output = os.path.join(output_dir, "input_CCDD", "graph", "edges.edg")
        assert os.path.exists(expected_output), f"Output file not found: {expected_output}"
        
        # Verify content
        with open(expected_output, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1, "Should have exactly one CC edge"
            assert "CHEBI:123\tCHEBI:456" in lines[0], "Should contain the CC edge"
        
        # Verify provenance file was created
        provenance_file = os.path.join(output_dir, "input_CCDD", "graph", "provenance.json")
        assert os.path.exists(provenance_file), f"Provenance file not found: {provenance_file}"
        
        with open(provenance_file, 'r') as f:
            provenance = json.load(f)
            assert provenance["style"] == "CCDD"
            assert provenance["edge_count"] == 1
            assert provenance["nodes_filename"] == "test_nodes.jsonl"
            assert provenance["edges_filename"] == "test_edges.jsonl"
            assert "timestamp" in provenance
            assert "script" in provenance


def test_check_accepted_with_cd_mode():
    """Test check_accepted with cd_mode='allow_all' allows CD edges (data leakage analysis)."""
    typemap = {
        "CHEBI:123": {"biolink:ChemicalEntity"},
        "MONDO:456": {"biolink:DiseaseOrPhenotypicFeature"},
        "CHEBI:456": {"biolink:ChemicalEntity"}
    }

    accepted = [("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature")]

    # Should KEEP CD edges with cd_mode='allow_all' (no leakage prevention)
    cd_edge = {"subject": "CHEBI:123", "object": "MONDO:456"}
    assert check_accepted(cd_edge, typemap, accepted, cd_mode="allow_all") == False

    # Should keep chemical-chemical edge
    cc_edge = {"subject": "CHEBI:123", "object": "CHEBI:456"}
    assert check_accepted(cc_edge, typemap, accepted, cd_mode="allow_all") == False

    # Should filter chemical-gene edge (not in accepted list)
    typemap["HGNC:789"] = {"biolink:Gene"}
    cg_edge = {"subject": "CHEBI:123", "object": "HGNC:789"}
    assert check_accepted(cg_edge, typemap, accepted, cd_mode="allow_all") == True


def test_keep_CCDD_with_cd():
    """Test CCDD_with_cd filtering includes CD edges."""
    typemap = {
        "CHEBI:123": {"biolink:ChemicalEntity"},
        "CHEBI:456": {"biolink:ChemicalEntity"},
        "MONDO:456": {"biolink:DiseaseOrPhenotypicFeature"},
        "MONDO:789": {"biolink:DiseaseOrPhenotypicFeature"},
        "HGNC:789": {"biolink:Gene"}
    }

    # Should filter subclass edges
    subclass_edge = {"predicate": "biolink:subclass_of", "subject": "A", "object": "B"}
    assert keep_CCDD_with_cd(subclass_edge, typemap) == True

    # Should KEEP chemical-disease edges (data leakage variant)
    cd_edge = {"predicate": "biolink:treats", "subject": "CHEBI:123", "object": "MONDO:456"}
    assert keep_CCDD_with_cd(cd_edge, typemap) == False

    # Should keep chemical-chemical edges
    cc_edge = {"predicate": "biolink:interacts_with", "subject": "CHEBI:123", "object": "CHEBI:456"}
    assert keep_CCDD_with_cd(cc_edge, typemap) == False

    # Should keep disease-disease edges
    dd_edge = {"predicate": "biolink:associated_with", "subject": "MONDO:456", "object": "MONDO:789"}
    assert keep_CCDD_with_cd(dd_edge, typemap) == False

    # Should filter chemical-gene edges
    cg_edge = {"predicate": "biolink:affects", "subject": "CHEBI:123", "object": "HGNC:789"}
    assert keep_CCDD_with_cd(cg_edge, typemap) == True


def test_keep_CCDD_with_subclass_with_cd():
    """Test CCDD_with_subclass_with_cd includes both subclass and CD edges."""
    typemap = {
        "CHEBI:123": {"biolink:ChemicalEntity"},
        "MONDO:456": {"biolink:DiseaseOrPhenotypicFeature"},
        "MONDO:789": {"biolink:DiseaseOrPhenotypicFeature"}
    }

    # Should KEEP subclass edges
    subclass_edge = {"predicate": "biolink:subclass_of", "subject": "MONDO:456", "object": "MONDO:789"}
    assert keep_CCDD_with_subclass_with_cd(subclass_edge, typemap) == False

    # Should KEEP chemical-disease edges (data leakage variant)
    cd_edge = {"predicate": "biolink:treats", "subject": "CHEBI:123", "object": "MONDO:456"}
    assert keep_CCDD_with_subclass_with_cd(cd_edge, typemap) == False


def test_keep_CGD_with_cd():
    """Test CGD_with_cd filtering includes CD edges."""
    typemap = {
        "CHEBI:123": {"biolink:ChemicalEntity"},
        "MONDO:456": {"biolink:DiseaseOrPhenotypicFeature"},
        "HGNC:789": {"biolink:Gene"}
    }

    # Should filter subclass edges
    subclass_edge = {"predicate": "biolink:subclass_of", "subject": "A", "object": "B"}
    assert keep_CGD_with_cd(subclass_edge, typemap) == True

    # Should KEEP chemical-disease edges (data leakage variant)
    cd_edge = {"predicate": "biolink:treats", "subject": "CHEBI:123", "object": "MONDO:456"}
    assert keep_CGD_with_cd(cd_edge, typemap) == False

    # Should keep chemical-gene edges
    cg_edge = {"predicate": "biolink:affects", "subject": "CHEBI:123", "object": "HGNC:789"}
    assert keep_CGD_with_cd(cg_edge, typemap) == False

    # Should keep gene-disease edges
    gd_edge = {"predicate": "biolink:associated_with", "subject": "HGNC:789", "object": "MONDO:456"}
    assert keep_CGD_with_cd(gd_edge, typemap) == False


def test_keep_CGD_with_subclass_with_cd():
    """Test CGD_with_subclass_with_cd includes subclass and CD edges."""
    typemap = {
        "CHEBI:123": {"biolink:ChemicalEntity"},
        "MONDO:456": {"biolink:DiseaseOrPhenotypicFeature"},
        "HGNC:789": {"biolink:Gene"}
    }

    # Should KEEP subclass edges
    subclass_edge = {"predicate": "biolink:subclass_of", "subject": "A", "object": "B"}
    assert keep_CGD_with_subclass_with_cd(subclass_edge, typemap) == False

    # Should KEEP chemical-disease edges (data leakage variant)
    cd_edge = {"predicate": "biolink:treats", "subject": "CHEBI:123", "object": "MONDO:456"}
    assert keep_CGD_with_subclass_with_cd(cd_edge, typemap) == False


def test_collect_organism_taxon_nodes():
    """Test collecting OrganismTaxon nodes from nodes file."""
    import jsonlines

    sample_nodes = [
        {"id": "NCBITaxon:9606", "name": "Homo sapiens", "category": ["biolink:OrganismTaxon", "biolink:NamedThing"]},
        {"id": "NCBITaxon:10090", "name": "Mus musculus", "category": ["biolink:OrganismTaxon", "biolink:NamedThing"]},
        {"id": "HGNC:5", "name": "Human Gene", "category": ["biolink:Gene", "biolink:NamedThing"]},
        {"id": "CHEBI:123", "name": "Test Chemical", "category": ["biolink:ChemicalEntity", "biolink:NamedThing"]}
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        with jsonlines.Writer(f) as writer:
            for node in sample_nodes:
                writer.write(node)
        temp_path = f.name

    try:
        organism_taxon_ids = collect_organism_taxon_nodes(temp_path)

        # Should collect OrganismTaxon nodes
        assert "NCBITaxon:9606" in organism_taxon_ids
        assert "NCBITaxon:10090" in organism_taxon_ids

        # Should not collect other types
        assert "HGNC:5" not in organism_taxon_ids
        assert "CHEBI:123" not in organism_taxon_ids

        assert len(organism_taxon_ids) == 2
    finally:
        os.unlink(temp_path)


def test_keep_human_only_filters_organism_taxon():
    """Test that human_only filter removes OrganismTaxon edges."""
    typemap = {
        "NCBITaxon:9606": {"biolink:OrganismTaxon"},
        "HGNC:5": {"biolink:Gene"},
        "CHEBI:123": {"biolink:ChemicalEntity"}
    }

    organism_taxon_ids = {"NCBITaxon:9606"}
    nonhuman_ids = set()

    # Edge with OrganismTaxon subject should be filtered
    taxon_edge = {"subject": "NCBITaxon:9606", "object": "HGNC:5", "predicate": "biolink:related_to"}
    assert keep_human_only(taxon_edge, typemap, nonhuman_ids, organism_taxon_ids) == True

    # Edge with OrganismTaxon object should be filtered
    taxon_edge2 = {"subject": "HGNC:5", "object": "NCBITaxon:9606", "predicate": "biolink:in_taxon"}
    assert keep_human_only(taxon_edge2, typemap, nonhuman_ids, organism_taxon_ids) == True

    # Edge without OrganismTaxon should pass
    normal_edge = {"subject": "CHEBI:123", "object": "HGNC:5", "predicate": "biolink:affects"}
    assert keep_human_only(normal_edge, typemap, nonhuman_ids, organism_taxon_ids) == False


def test_keep_human_only_filters_nonhuman_genes():
    """Test that human_only filter removes non-human gene edges."""
    typemap = {
        "HGNC:5": {"biolink:Gene"},
        "MGI:95523": {"biolink:Gene"},
        "CHEBI:123": {"biolink:ChemicalEntity"}
    }

    organism_taxon_ids = set()
    nonhuman_ids = {"MGI:95523"}  # Mouse gene

    # Edge with non-human gene subject should be filtered
    mouse_edge = {"subject": "MGI:95523", "object": "CHEBI:123", "predicate": "biolink:affected_by"}
    assert keep_human_only(mouse_edge, typemap, nonhuman_ids, organism_taxon_ids) == True

    # Edge with non-human gene object should be filtered
    mouse_edge2 = {"subject": "CHEBI:123", "object": "MGI:95523", "predicate": "biolink:affects"}
    assert keep_human_only(mouse_edge2, typemap, nonhuman_ids, organism_taxon_ids) == True

    # Edge with human gene should pass
    human_edge = {"subject": "HGNC:5", "object": "CHEBI:123", "predicate": "biolink:affected_by"}
    assert keep_human_only(human_edge, typemap, nonhuman_ids, organism_taxon_ids) == False


def test_keep_human_only_combined_filters():
    """Test human_only filter with both OrganismTaxon and non-human genes."""
    typemap = {
        "NCBITaxon:9606": {"biolink:OrganismTaxon"},
        "HGNC:5": {"biolink:Gene"},
        "MGI:95523": {"biolink:Gene"},
        "CHEBI:123": {"biolink:ChemicalEntity"}
    }

    organism_taxon_ids = {"NCBITaxon:9606"}
    nonhuman_ids = {"MGI:95523"}

    # OrganismTaxon edge should be filtered
    taxon_edge = {"subject": "NCBITaxon:9606", "object": "HGNC:5", "predicate": "biolink:related_to"}
    assert keep_human_only(taxon_edge, typemap, nonhuman_ids, organism_taxon_ids) == True

    # Non-human gene edge should be filtered
    mouse_edge = {"subject": "MGI:95523", "object": "CHEBI:123", "predicate": "biolink:affects"}
    assert keep_human_only(mouse_edge, typemap, nonhuman_ids, organism_taxon_ids) == True

    # Human gene edge should pass
    human_edge = {"subject": "HGNC:5", "object": "CHEBI:123", "predicate": "biolink:affects"}
    assert keep_human_only(human_edge, typemap, nonhuman_ids, organism_taxon_ids) == False


def test_keep_human_only_default_parameters():
    """Test human_only filter with default None parameters."""
    typemap = {
        "HGNC:5": {"biolink:Gene"},
        "CHEBI:123": {"biolink:ChemicalEntity"}
    }

    # With None parameters, should not filter anything
    edge = {"subject": "HGNC:5", "object": "CHEBI:123", "predicate": "biolink:affects"}
    assert keep_human_only(edge, typemap) == False


def test_keep_no_text_mined_filters_text_mining():
    """Test that no_text_mined filter removes text-mined edges."""
    typemap = {}

    # Edge with text_mining_agent should be filtered
    text_mined_edge = {
        "subject": "CHEBI:123",
        "object": "NCBIGene:5770",
        "predicate": "biolink:affects",
        "agent_type": "text_mining_agent"
    }
    assert keep_no_text_mined(text_mined_edge, typemap) == True

    # Edge with manual_agent should pass
    manual_edge = {
        "subject": "CHEBI:123",
        "object": "NCBIGene:5770",
        "predicate": "biolink:affects",
        "agent_type": "manual_agent"
    }
    assert keep_no_text_mined(manual_edge, typemap) == False

    # Edge without agent_type should pass
    no_agent_edge = {
        "subject": "CHEBI:123",
        "object": "NCBIGene:5770",
        "predicate": "biolink:affects"
    }
    assert keep_no_text_mined(no_agent_edge, typemap) == False


def test_keep_human_only_no_text_mined_combines_filters():
    """Test that human_only_no_text_mined combines both filters."""
    typemap = {
        "NCBITaxon:9606": {"biolink:OrganismTaxon"},
        "MGI:95523": {"biolink:Gene"},
        "HGNC:5": {"biolink:Gene"},
        "CHEBI:123": {"biolink:ChemicalEntity"}
    }

    organism_taxon_ids = {"NCBITaxon:9606"}
    nonhuman_ids = {"MGI:95523"}

    # Edge with OrganismTaxon should be filtered (human filter)
    taxon_edge = {
        "subject": "NCBITaxon:9606",
        "object": "HGNC:5",
        "predicate": "biolink:related_to",
        "agent_type": "manual_agent"
    }
    assert keep_human_only_no_text_mined(taxon_edge, typemap, nonhuman_ids, organism_taxon_ids) == True

    # Edge with non-human gene should be filtered (human filter)
    mouse_edge = {
        "subject": "MGI:95523",
        "object": "CHEBI:123",
        "predicate": "biolink:affects",
        "agent_type": "manual_agent"
    }
    assert keep_human_only_no_text_mined(mouse_edge, typemap, nonhuman_ids, organism_taxon_ids) == True

    # Edge with text_mining_agent should be filtered (text mining filter)
    text_mined_edge = {
        "subject": "HGNC:5",
        "object": "CHEBI:123",
        "predicate": "biolink:affects",
        "agent_type": "text_mining_agent"
    }
    assert keep_human_only_no_text_mined(text_mined_edge, typemap, nonhuman_ids, organism_taxon_ids) == True

    # Valid human edge with manual_agent should pass
    valid_edge = {
        "subject": "HGNC:5",
        "object": "CHEBI:123",
        "predicate": "biolink:affects",
        "agent_type": "manual_agent"
    }
    assert keep_human_only_no_text_mined(valid_edge, typemap, nonhuman_ids, organism_taxon_ids) == False


def test_multi_filter_1_with_cd_keeps_cd_edges():
    """Test that multi_filter_1_with_cd does NOT filter Chemical-Disease edges."""
    typemap = {
        "CHEBI:123": {"biolink:ChemicalEntity"},
        "MONDO:456": {"biolink:DiseaseOrPhenotypicFeature"}
    }

    # Chemical-Disease edge should NOT be filtered (unlike multi_filter_1)
    cd_edge = {
        "subject": "CHEBI:123",
        "object": "MONDO:456",
        "predicate": "biolink:treats"
    }
    assert multi_filter_1_with_cd(cd_edge, typemap) == False


def test_multi_filter_1_with_cd_filters_text_mined():
    """Test that multi_filter_1_with_cd filters text-mined edges."""
    typemap = {
        "HGNC:5": {"biolink:Gene"},
        "CHEBI:123": {"biolink:ChemicalEntity"}
    }

    # Text-mined edge should be filtered
    text_mined_edge = {
        "subject": "HGNC:5",
        "object": "CHEBI:123",
        "predicate": "biolink:affects",
        "agent_type": "text_mining_agent"
    }
    assert multi_filter_1_with_cd(text_mined_edge, typemap) == True


def test_multi_filter_1_with_cd_filters_ncit_subclass():
    """Test that multi_filter_1_with_cd filters NCIT-to-NCIT subclass_of edges."""
    typemap = {}

    # NCIT-to-NCIT subclass edge should be filtered
    ncit_subclass_edge = {
        "subject": "NCIT:C123",
        "object": "NCIT:C456",
        "predicate": "biolink:subclass_of"
    }
    assert multi_filter_1_with_cd(ncit_subclass_edge, typemap) == True

    # Non-NCIT subclass edge should NOT be filtered
    mondo_subclass_edge = {
        "subject": "MONDO:123",
        "object": "MONDO:456",
        "predicate": "biolink:subclass_of"
    }
    assert multi_filter_1_with_cd(mondo_subclass_edge, typemap) == False


def test_write_hgt_input_format(temp_edges_file, temp_nodes_file):
    """Test that write_hgt_input creates correct TSV format."""
    import tempfile
    import jsonlines

    typemap = load_typemap(temp_nodes_file)

    # Use a no-op filter to keep all edges
    def no_filter(edge, typemap):
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        edge_count, predicate_stats = write_hgt_input(
            temp_edges_file, temp_nodes_file, tmpdir, no_filter, typemap
        )

        # Check edges.tsv exists and has correct format
        edges_tsv = os.path.join(tmpdir, "edges.tsv")
        assert os.path.exists(edges_tsv)

        with open(edges_tsv, 'r') as f:
            lines = f.readlines()
            # Should have header + edges
            assert len(lines) > 0
            assert lines[0].strip() == "subject\tpredicate\tobject"

        # Check nodes.tsv exists and has correct format
        nodes_tsv = os.path.join(tmpdir, "nodes.tsv")
        assert os.path.exists(nodes_tsv)

        with open(nodes_tsv, 'r') as f:
            lines = f.readlines()
            # Should have header + nodes
            assert len(lines) > 0
            assert lines[0].strip() == "id\ttype"


def test_write_hgt_input_pseudo_predicates():
    """Test that write_hgt_input creates pseudo-predicates from qualifiers."""
    import tempfile
    import jsonlines

    # Create test data with qualifiers
    edges_data = [
        {
            "subject": "CHEBI:123",
            "object": "HGNC:456",
            "predicate": "biolink:affects",
            "object_direction_qualifier": "biolink:increased",
            "object_aspect_qualifier": "biolink:activity"
        },
        {
            "subject": "CHEBI:789",
            "object": "HGNC:999",
            "predicate": "biolink:regulates",
            "object_direction_qualifier": "biolink:decreased"
        },
        {
            "subject": "CHEBI:111",
            "object": "HGNC:222",
            "predicate": "biolink:targets"
        }
    ]

    nodes_data = [
        {"id": "CHEBI:123", "category": ["biolink:ChemicalEntity"]},
        {"id": "CHEBI:789", "category": ["biolink:ChemicalEntity"]},
        {"id": "CHEBI:111", "category": ["biolink:ChemicalEntity"]},
        {"id": "HGNC:456", "category": ["biolink:Gene"]},
        {"id": "HGNC:999", "category": ["biolink:Gene"]},
        {"id": "HGNC:222", "category": ["biolink:Gene"]}
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as edges_f:
        for edge in edges_data:
            edges_f.write(json.dumps(edge) + '\n')
        edges_file = edges_f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as nodes_f:
        for node in nodes_data:
            nodes_f.write(json.dumps(node) + '\n')
        nodes_file = nodes_f.name

    try:
        typemap = load_typemap(nodes_file)

        def no_filter(edge, typemap):
            return False

        with tempfile.TemporaryDirectory() as tmpdir:
            edge_count, predicate_stats = write_hgt_input(
                edges_file, nodes_file, tmpdir, no_filter, typemap
            )

            # Read edges.tsv and check pseudo-predicates
            edges_tsv = os.path.join(tmpdir, "edges.tsv")
            with open(edges_tsv, 'r') as f:
                lines = f.readlines()

                # Check that we have the expected predicates
                assert "affects-increased-activity" in predicate_stats
                assert "regulates-decreased" in predicate_stats
                assert "targets" in predicate_stats

    finally:
        os.unlink(edges_file)
        os.unlink(nodes_file)


def test_write_kgx_output_format():
    """Test that write_kgx_output produces valid KGX JSONL files."""
    # Create test nodes
    nodes_data = [
        {"id": "CHEBI:1", "name": "Drug A", "category": ["biolink:ChemicalEntity"]},
        {"id": "CHEBI:2", "name": "Drug B", "category": ["biolink:ChemicalEntity"]},
        {"id": "MONDO:1", "name": "Disease A", "category": ["biolink:DiseaseOrPhenotypicFeature"]},
        {"id": "UNUSED:1", "name": "Unused Node", "category": ["biolink:NamedThing"]},
    ]

    # Create test edges
    edges_data = [
        {"subject": "CHEBI:1", "predicate": "biolink:similar_to", "object": "CHEBI:2"},
        {"subject": "CHEBI:2", "predicate": "biolink:similar_to", "object": "CHEBI:1"},
        {"subject": "CHEBI:1", "predicate": "biolink:treats", "object": "MONDO:1"},  # Should be filtered
    ]

    # Write test files
    edges_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    nodes_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)

    try:
        with jsonlines.open(edges_file.name, 'w') as writer:
            for edge in edges_data:
                writer.write(edge)

        with jsonlines.open(nodes_file.name, 'w') as writer:
            for node in nodes_data:
                writer.write(node)

        edges_file.close()
        nodes_file.close()

        # Load typemap
        typemap = load_typemap(nodes_file.name)

        # Use keep_CCDD filter (should filter CD edges)
        with tempfile.TemporaryDirectory() as tmpdir:
            edge_count, predicate_stats = write_kgx_output(
                edges_file.name, nodes_file.name, tmpdir, keep_CCDD, typemap
            )

            # Check edge count (CD edge filtered out)
            assert edge_count == 2

            # Check predicate stats
            assert predicate_stats == {"biolink:similar_to": 2}

            # Read edges.jsonl and verify format
            edges_jsonl = os.path.join(tmpdir, "edges.jsonl")
            assert os.path.exists(edges_jsonl)

            with jsonlines.open(edges_jsonl) as reader:
                edges = list(reader)

            # Should have 2 edges (CD filtered)
            assert len(edges) == 2

            # Check that edges preserve original format
            assert edges[0]["subject"] == "CHEBI:1"
            assert edges[0]["predicate"] == "biolink:similar_to"
            assert edges[0]["object"] == "CHEBI:2"

            # Read nodes.jsonl and verify format
            nodes_jsonl = os.path.join(tmpdir, "nodes.jsonl")
            assert os.path.exists(nodes_jsonl)

            with jsonlines.open(nodes_jsonl) as reader:
                nodes = list(reader)

            # Should have 2 nodes (only nodes in filtered edges)
            assert len(nodes) == 2

            # Check that nodes preserve original format
            node_ids = {node["id"] for node in nodes}
            assert "CHEBI:1" in node_ids
            assert "CHEBI:2" in node_ids
            assert "MONDO:1" not in node_ids  # Filtered because CD edge was removed
            assert "UNUSED:1" not in node_ids  # Filtered because not in any edge

    finally:
        os.unlink(edges_file.name)
        os.unlink(nodes_file.name)