#!/usr/bin/env python3
"""Tests for generate_embeddings module."""
import os
import tempfile
import pytest
import json
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.embedding.generate_embeddings import (
    get_next_embedding_version, count_edges_and_nodes, generate_embeddings
)


@pytest.fixture
def temp_embeddings_dir():
    """Create temporary embeddings directory with existing versions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        embeddings_dir = os.path.join(temp_dir, "embeddings")
        os.makedirs(embeddings_dir)
        
        # Create some existing embedding versions
        os.makedirs(os.path.join(embeddings_dir, "embeddings_0"))
        os.makedirs(os.path.join(embeddings_dir, "embeddings_1"))
        os.makedirs(os.path.join(embeddings_dir, "embeddings_4"))
        
        yield embeddings_dir


@pytest.fixture
def sample_graph_file():
    """Create a sample .edg graph file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.edg', delete=False) as f:
        # Write tab-separated edges
        f.write("CHEBI:123\tCHEBI:456\n")
        f.write("CHEBI:456\tCHEBI:789\n")
        f.write("MONDO:123\tMONDO:456\n")
        f.write("CHEBI:123\tCHEBI:789\n")  # CHEBI:123 appears again
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_graph_file_malformed():
    """Create a malformed graph file for testing edge cases."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.edg', delete=False) as f:
        # Write some good and bad lines
        f.write("CHEBI:123\tCHEBI:456\n")
        f.write("SINGLE_NODE\n")  # Only one node - should be skipped
        f.write("TOO\tMANY\tCOLUMNS\n")  # Too many columns - should still work
        f.write("\t\n")  # Empty nodes - should be skipped
        f.write("GOOD:1\tGOOD:2\n")
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


def test_get_next_embedding_version_existing_versions(temp_embeddings_dir):
    """Test version numbering with existing embedding versions."""
    version = get_next_embedding_version(temp_embeddings_dir)
    assert version == 5  # Should be max(0, 1, 4) + 1


def test_get_next_embedding_version_empty_dir():
    """Test version numbering with empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        embeddings_dir = os.path.join(temp_dir, "embeddings")
        os.makedirs(embeddings_dir)
        
        version = get_next_embedding_version(embeddings_dir)
        assert version == 0


def test_get_next_embedding_version_nonexistent_dir():
    """Test version numbering with non-existent directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        nonexistent_dir = os.path.join(temp_dir, "nonexistent")
        
        version = get_next_embedding_version(nonexistent_dir)
        assert version == 0


def test_get_next_embedding_version_invalid_names(temp_embeddings_dir):
    """Test version numbering ignores invalid directory names."""
    # Add some directories with invalid names
    os.makedirs(os.path.join(temp_embeddings_dir, "not_embeddings_1"))
    os.makedirs(os.path.join(temp_embeddings_dir, "embeddings_"))
    os.makedirs(os.path.join(temp_embeddings_dir, "embeddings_abc"))
    
    version = get_next_embedding_version(temp_embeddings_dir)
    assert version == 5  # Should still be max(0, 1, 4) + 1, ignoring invalid names


def test_count_edges_and_nodes(sample_graph_file):
    """Test counting edges and unique nodes."""
    edge_count, node_count = count_edges_and_nodes(sample_graph_file)
    
    assert edge_count == 4  # 4 lines in the file
    assert node_count == 5  # Unique nodes: CHEBI:123, CHEBI:456, CHEBI:789, MONDO:123, MONDO:456


def test_count_edges_and_nodes_malformed(sample_graph_file_malformed):
    """Test counting with malformed graph file."""
    edge_count, node_count = count_edges_and_nodes(sample_graph_file_malformed)
    
    # Should handle malformed lines gracefully
    assert edge_count == 3  # Only lines with >= 2 columns count as edges
    assert node_count == 4  # CHEBI:123, CHEBI:456, GOOD:1, GOOD:2


def test_count_edges_and_nodes_empty_file():
    """Test counting with empty file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.edg', delete=False) as f:
        temp_path = f.name
    
    try:
        edge_count, node_count = count_edges_and_nodes(temp_path)
        assert edge_count == 0
        assert node_count == 0
    finally:
        os.unlink(temp_path)


def test_count_edges_and_nodes_nonexistent_file():
    """Test counting with non-existent file."""
    with pytest.raises(FileNotFoundError):
        count_edges_and_nodes("/nonexistent/file.edg")


@patch('subprocess.run')
def test_generate_embeddings_success(mock_run, sample_graph_file):
    """Test successful embedding generation."""
    # Mock successful subprocess run
    mock_result = MagicMock()
    mock_result.stdout = "Embedding generation completed"
    mock_result.stderr = ""
    mock_run.return_value = mock_result
    
    # Create a temporary base directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up directory structure: base_dir/graph/edges.edg
        graph_dir = os.path.join(temp_dir, "graph")
        os.makedirs(graph_dir)
        
        # Copy sample graph file to the expected location
        graph_file = os.path.join(graph_dir, "edges.edg")
        with open(sample_graph_file, 'r') as src:
            with open(graph_file, 'w') as dst:
                dst.write(src.read())
        
        # Generate embeddings
        version_dir = generate_embeddings(
            graph_file=graph_file,
            dimensions=128,
            walk_length=10,
            num_walks=5
        )
        
        # Verify results
        assert os.path.exists(version_dir)
        assert os.path.basename(version_dir) == "embeddings_0"
        
        # Check that provenance file was created
        provenance_file = os.path.join(version_dir, "provenance.json")
        assert os.path.exists(provenance_file)
        
        # Verify provenance content
        with open(provenance_file, 'r') as f:
            provenance = json.load(f)
        
        assert provenance["algorithm"] == "node2vec"
        assert provenance["tool"] == "pecanpy"
        assert provenance["parameters"]["dimensions"] == 128
        assert provenance["parameters"]["walk_length"] == 10
        assert provenance["parameters"]["num_walks"] == 5
        assert provenance["graph_info"]["edge_count"] == 4
        assert provenance["graph_info"]["node_count"] == 5
        
        # Verify subprocess was called correctly
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]  # First positional argument (the command)
        
        assert call_args[0] == "pecanpy"
        assert "--input" in call_args
        assert "--output" in call_args
        assert "--dimensions" in call_args and "128" in call_args
        assert "--walk-length" in call_args and "10" in call_args
        assert "--num-walks" in call_args and "5" in call_args


@patch('subprocess.run')
def test_generate_embeddings_subprocess_failure(mock_run, sample_graph_file):
    """Test embedding generation with subprocess failure."""
    # Mock failed subprocess run
    mock_run.side_effect = subprocess.CalledProcessError(
        1, "pecanpy"
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        graph_dir = os.path.join(temp_dir, "graph")
        os.makedirs(graph_dir)
        
        graph_file = os.path.join(graph_dir, "edges.edg")
        with open(sample_graph_file, 'r') as src:
            with open(graph_file, 'w') as dst:
                dst.write(src.read())
        
        # Should raise the subprocess error
        with pytest.raises(subprocess.CalledProcessError):
            generate_embeddings(graph_file=graph_file)


def test_generate_embeddings_nonexistent_file():
    """Test embedding generation with non-existent input file."""
    with pytest.raises(FileNotFoundError, match="Graph file not found"):
        generate_embeddings("/nonexistent/file.edg")


@patch('subprocess.run')
def test_generate_embeddings_directory_creation(mock_run, sample_graph_file):
    """Test that embedding generation creates necessary directories."""
    mock_result = MagicMock()
    mock_result.stdout = "Success"
    mock_result.stderr = ""
    mock_run.return_value = mock_result
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create base directory but not embeddings directory
        graph_dir = os.path.join(temp_dir, "graph")
        os.makedirs(graph_dir)
        
        graph_file = os.path.join(graph_dir, "edges.edg")
        with open(sample_graph_file, 'r') as src:
            with open(graph_file, 'w') as dst:
                dst.write(src.read())
        
        # Generate embeddings
        version_dir = generate_embeddings(graph_file=graph_file)
        
        # Check that all necessary directories were created
        embeddings_dir = os.path.join(temp_dir, "embeddings")
        assert os.path.exists(embeddings_dir)
        assert os.path.exists(version_dir)
        assert os.path.basename(version_dir) == "embeddings_0"


@patch('subprocess.run')
def test_generate_embeddings_multiple_versions(mock_run, sample_graph_file):
    """Test that multiple embedding generations create different versions."""
    mock_result = MagicMock()
    mock_result.stdout = "Success"
    mock_result.stderr = ""
    mock_run.return_value = mock_result
    
    with tempfile.TemporaryDirectory() as temp_dir:
        graph_dir = os.path.join(temp_dir, "graph")
        os.makedirs(graph_dir)
        
        graph_file = os.path.join(graph_dir, "edges.edg")
        with open(sample_graph_file, 'r') as src:
            with open(graph_file, 'w') as dst:
                dst.write(src.read())
        
        # Generate first embedding
        version_dir_1 = generate_embeddings(graph_file=graph_file)
        assert os.path.basename(version_dir_1) == "embeddings_0"
        
        # Generate second embedding
        version_dir_2 = generate_embeddings(graph_file=graph_file)
        assert os.path.basename(version_dir_2) == "embeddings_1"
        
        # Both should exist
        assert os.path.exists(version_dir_1)
        assert os.path.exists(version_dir_2)
        assert version_dir_1 != version_dir_2


@patch('subprocess.run')
def test_generate_embeddings_default_parameters(mock_run, sample_graph_file):
    """Test embedding generation with default parameters."""
    mock_result = MagicMock()
    mock_result.stdout = "Success"
    mock_result.stderr = ""
    mock_run.return_value = mock_result
    
    with tempfile.TemporaryDirectory() as temp_dir:
        graph_dir = os.path.join(temp_dir, "graph")
        os.makedirs(graph_dir)
        
        graph_file = os.path.join(graph_dir, "edges.edg")
        with open(sample_graph_file, 'r') as src:
            with open(graph_file, 'w') as dst:
                dst.write(src.read())
        
        # Generate with defaults
        version_dir = generate_embeddings(graph_file=graph_file)
        
        # Check provenance has default values
        provenance_file = os.path.join(version_dir, "provenance.json")
        with open(provenance_file, 'r') as f:
            provenance = json.load(f)
        
        # Verify default parameters
        params = provenance["parameters"]
        assert params["dimensions"] == 512
        assert params["walk_length"] == 30
        assert params["num_walks"] == 10
        assert params["window_size"] == 10
        assert params["p"] == 1
        assert params["q"] == 1
        assert params["workers"] == 4


@patch('subprocess.run')
def test_generate_embeddings_custom_parameters(mock_run, sample_graph_file):
    """Test embedding generation with custom parameters."""
    mock_result = MagicMock()
    mock_result.stdout = "Custom run"
    mock_result.stderr = "Custom stderr"
    mock_run.return_value = mock_result
    
    with tempfile.TemporaryDirectory() as temp_dir:
        graph_dir = os.path.join(temp_dir, "graph")
        os.makedirs(graph_dir)
        
        graph_file = os.path.join(graph_dir, "edges.edg")
        with open(sample_graph_file, 'r') as src:
            with open(graph_file, 'w') as dst:
                dst.write(src.read())
        
        # Generate with custom parameters
        version_dir = generate_embeddings(
            graph_file=graph_file,
            dimensions=256,
            walk_length=40,
            num_walks=15,
            window_size=8,
            p=0.5,
            q=2.0,
            workers=8
        )
        
        # Check provenance has custom values
        provenance_file = os.path.join(version_dir, "provenance.json")
        with open(provenance_file, 'r') as f:
            provenance = json.load(f)
        
        # Verify custom parameters
        params = provenance["parameters"]
        assert params["dimensions"] == 256
        assert params["walk_length"] == 40
        assert params["num_walks"] == 15
        assert params["window_size"] == 8
        assert params["p"] == 0.5
        assert params["q"] == 2.0
        assert params["workers"] == 8
        
        # Verify pecanpy output is captured
        assert provenance["pecanpy_output"]["stdout"] == "Custom run"
        assert provenance["pecanpy_output"]["stderr"] == "Custom stderr"


@patch('subprocess.run')
def test_generate_embeddings_provenance_metadata(mock_run, sample_graph_file):
    """Test that all expected provenance metadata is captured."""
    mock_result = MagicMock()
    mock_result.stdout = "Success"
    mock_result.stderr = ""
    mock_run.return_value = mock_result
    
    with tempfile.TemporaryDirectory() as temp_dir:
        graph_dir = os.path.join(temp_dir, "graph")
        os.makedirs(graph_dir)
        
        graph_file = os.path.join(graph_dir, "edges.edg")
        with open(sample_graph_file, 'r') as src:
            with open(graph_file, 'w') as dst:
                dst.write(src.read())
        
        version_dir = generate_embeddings(graph_file=graph_file)
        
        provenance_file = os.path.join(version_dir, "provenance.json")
        with open(provenance_file, 'r') as f:
            provenance = json.load(f)
        
        # Check all required metadata fields
        assert "timestamp" in provenance
        assert "duration_seconds" in provenance
        assert provenance["script"] == "generate_embeddings.py"
        assert provenance["version"] == "embeddings_0"
        assert provenance["algorithm"] == "node2vec"
        assert provenance["tool"] == "pecanpy"
        assert "input_graph_file" in provenance
        assert "output_embeddings_file" in provenance
        assert "parameters" in provenance
        assert "graph_info" in provenance
        assert "pecanpy_output" in provenance
        assert "description" in provenance
        
        # Check that duration is reasonable (should be very small for mocked run)
        assert isinstance(provenance["duration_seconds"], float)
        assert provenance["duration_seconds"] >= 0