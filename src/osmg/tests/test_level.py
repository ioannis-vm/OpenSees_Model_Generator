"""Unit tests for the Level class."""

#
#   _|_|      _|_|_|  _|      _|    _|_|_|
# _|    _|  _|        _|_|  _|_|  _|
# _|    _|    _|_|    _|  _|  _|  _|  _|_|
# _|    _|        _|  _|      _|  _|    _|
#   _|_|    _|_|_|    _|      _|    _|_|_|
#
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

import pytest
from unittest.mock import MagicMock
from osmg.core.osmg_collections import NodeCollection, Collection
from osmg.core.level import Level
from osmg.core.component_assemblies import ComponentAssembly


class TestLevel:
    """Tests for the Level class."""

    def setup_method(self):
        """Set up mock objects for testing."""
        self.mock_node_collection = MagicMock(spec=NodeCollection)
        self.mock_collection = MagicMock(spec=Collection)
        self.mock_node_collection.__len__.return_value = 3
        self.mock_collection.__len__.return_value = 2
        self.mock_node_collection.__getitem__.side_effect = lambda x: [f"Node({i})" for i in range(3)][x]
        self.mock_collection.__getitem__.side_effect = lambda x: [f"Component({i})" for i in range(2)][x]

        # Patch the NodeCollection and Collection to use mocks
        Level.__post_init__ = lambda self: None  # Disable __post_init__ temporarily
        Level.nodes = self.mock_node_collection
        Level.components = self.mock_collection

    def test_initialization(self):
        """Test that the Level object initializes with the correct attributes."""
        level = Level(uid=1, elevation=10.0)

        assert level.uid == 1
        assert level.elevation == 10.0
        assert isinstance(level.nodes, NodeCollection)
        assert isinstance(level.components, Collection)

    def test_node_and_component_counts(self):
        """Test that the node and component counts are correct."""
        level = Level(uid=2, elevation=20.0)

        assert len(level.nodes) == 3
        assert len(level.components) == 2

    def test_repr_no_nodes_or_components(self):
        """Test the string representation with no nodes or components."""
        self.mock_node_collection.__len__.return_value = 0
        self.mock_collection.__len__.return_value = 0

        level = Level(uid=3, elevation=30.0)
        expected_repr = (
            "Level Object\n"
            "  UID: 3\n"
            "  Elevation: 30.0 units\n"
            "  Number of Nodes: 0\n"
            "  Number of Components: 0\n"
        )

        assert repr(level) == expected_repr

    def test_repr_with_nodes_and_components(self):
        """Test the string representation with nodes and components."""
        level = Level(uid=4, elevation=40.0)
        expected_repr = (
            "Level Object\n"
            "  UID: 4\n"
            "  Elevation: 40.0 units\n"
            "  Number of Nodes: 3\n"
            "  Number of Components: 2\n"
            "  Nodes: ['Node(0)', 'Node(1)', 'Node(2)']\n"
            "  Components: ['Component(0)', 'Component(1)']\n"
        )

        assert repr(level) == expected_repr

    def test_repr_truncated_preview(self):
        """Test the string representation when nodes or components are truncated."""
        self.mock_node_collection.__len__.return_value = 6
        self.mock_node_collection.__getitem__.side_effect = lambda x: [f"Node({i})" for i in range(6)][x]

        level = Level(uid=5, elevation=50.0)
        expected_repr = (
            "Level Object\n"
            "  UID: 5\n"
            "  Elevation: 50.0 units\n"
            "  Number of Nodes: 6\n"
            "  Number of Components: 2\n"
            "  Nodes: ['Node(0)', 'Node(1)', 'Node(2)', 'Node(3)', 'Node(4)']...\n"
            "  Components: ['Component(0)', 'Component(1)']\n"
        )

        assert repr(level) == expected_repr
