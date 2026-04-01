#!/usr/bin/env python3
"""Tests for type_utils module."""
import pytest
from src.graph_modification.type_utils import get_most_specific_type


def test_get_most_specific_type_chemical():
    """Test that most specific chemical type is returned."""
    categories = ["biolink:NamedThing", "biolink:ChemicalEntity", "biolink:SmallMolecule"]
    result = get_most_specific_type(categories)
    assert result == "SmallMolecule"


def test_get_most_specific_type_gene():
    """Test that most specific gene type is returned."""
    categories = ["biolink:NamedThing", "biolink:GeneOrGeneProduct", "biolink:Gene"]
    result = get_most_specific_type(categories)
    assert result == "Gene"


def test_get_most_specific_type_disease():
    """Test that most specific disease type is returned."""
    categories = ["biolink:NamedThing", "biolink:DiseaseOrPhenotypicFeature", "biolink:Disease"]
    result = get_most_specific_type(categories)
    assert result == "Disease"


def test_get_most_specific_type_with_prefix():
    """Test that biolink prefix is stripped correctly."""
    categories = ["biolink:NamedThing", "biolink:ChemicalEntity", "biolink:Drug"]
    result = get_most_specific_type(categories)
    assert result == "Drug"
    assert not result.startswith("biolink:")


def test_get_most_specific_type_unknown():
    """Test handling of unknown types."""
    categories = ["biolink:SomeNewType", "biolink:AnotherNewType"]
    result = get_most_specific_type(categories)
    # Should return one of the types (default mid-level)
    assert result in ["SomeNewType", "AnotherNewType"]


def test_get_most_specific_type_empty():
    """Test handling of empty categories list."""
    categories = []
    result = get_most_specific_type(categories)
    assert result == "Unknown"


def test_get_most_specific_type_none():
    """Test handling of None."""
    result = get_most_specific_type(None)
    assert result == "Unknown"


def test_get_most_specific_type_mixed_prefixes():
    """Test handling of mixed prefix/no-prefix categories."""
    categories = ["NamedThing", "biolink:ChemicalEntity", "biolink:SmallMolecule"]
    result = get_most_specific_type(categories)
    assert result == "SmallMolecule"


def test_get_most_specific_type_protein():
    """Test that most specific protein type is returned."""
    categories = ["biolink:NamedThing", "biolink:GeneOrGeneProduct", "biolink:Protein"]
    result = get_most_specific_type(categories)
    assert result == "Protein"


def test_get_most_specific_type_organism_taxon():
    """Test that OrganismTaxon is recognized as specific."""
    categories = ["biolink:NamedThing", "biolink:OrganismTaxon"]
    result = get_most_specific_type(categories)
    assert result == "OrganismTaxon"


def test_get_most_specific_type_pathway():
    """Test that most specific pathway type is returned."""
    categories = ["biolink:NamedThing", "biolink:BiologicalProcess", "biolink:Pathway"]
    result = get_most_specific_type(categories)
    assert result == "Pathway"


def test_get_most_specific_type_avoids_abstract():
    """Test that abstract types like PhysicalEssenceOrOccurrent are avoided."""
    categories = ["biolink:PhysicalEssenceOrOccurrent", "biolink:BiologicalProcess"]
    result = get_most_specific_type(categories)
    assert result == "BiologicalProcess"
    assert result != "PhysicalEssenceOrOccurrent"


def test_get_most_specific_type_avoids_occurrent():
    """Test that very abstract types like Occurrent are avoided when more specific types exist."""
    categories = ["biolink:Occurrent", "biolink:PhysicalEssence", "biolink:BiologicalEntity", "biolink:ChemicalEntity"]
    result = get_most_specific_type(categories)
    assert result == "ChemicalEntity"


def test_get_most_specific_type_real_node_example():
    """Test with a real node example that had problems."""
    # A realistic example node with many categories
    categories = [
        "biolink:SmallMolecule",
        "biolink:MolecularEntity",
        "biolink:ChemicalEntity",
        "biolink:PhysicalEssence",
        "biolink:ChemicalOrDrugOrTreatment",
        "biolink:ChemicalEntityOrGeneOrGeneProduct",
        "biolink:ChemicalEntityOrProteinOrPolypeptide",
        "biolink:NamedThing",
        "biolink:PhysicalEssenceOrOccurrent"
    ]
    result = get_most_specific_type(categories)
    assert result == "SmallMolecule"


def test_get_most_specific_type_bmt_validation():
    """Test that BMT properly identifies more specific types using ancestor count."""
    # Disease is more specific than DiseaseOrPhenotypicFeature
    categories = ["biolink:DiseaseOrPhenotypicFeature", "biolink:Disease"]
    result = get_most_specific_type(categories)
    assert result == "Disease"

    # SmallMolecule is more specific than ChemicalEntity
    categories = ["biolink:ChemicalEntity", "biolink:SmallMolecule"]
    result = get_most_specific_type(categories)
    assert result == "SmallMolecule"

    # Gene is more specific than GeneOrGeneProduct
    categories = ["biolink:GeneOrGeneProduct", "biolink:Gene"]
    result = get_most_specific_type(categories)
    assert result == "Gene"
