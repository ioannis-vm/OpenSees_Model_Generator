"""Unit tests for the Node class."""

#
#   _|_|      _|_|_|  _|      _|    _|_|_|
# _|    _|  _|        _|_|  _|_|  _|
# _|    _|    _|_|    _|  _|  _|  _|  _|_|
# _|    _|        _|  _|      _|  _|    _|
#   _|_|    _|_|_|    _|      _|    _|_|_|
#
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from unittest.mock import MagicMock
from osmg.elements.node import Node
from osmg.graphics.visibility import NodeVisibility


class TestNode:
    """Tests for the Node class."""

    def setup_method(self):
        """Set up a mock UIDGenerator and other common fixtures."""
        self.mock_uid_generator = MagicMock()
        self.mock_uid_generator.new.side_effect = [0, 1, 2]

    def test_node_initialization(self):
        """Test that the Node object initializes correctly."""
        node = Node(coords=[0.0, 0.0, 0.0], uid_generator=self.mock_uid_generator)

        assert node.coords == [0.0, 0.0, 0.0]
        assert node.uid == 0
        assert node.restraint == [False] * 6
        assert isinstance(node.visibility, NodeVisibility)
        self.mock_uid_generator.new.assert_called_once_with(node)

    def test_node_uid_generation(self):
        """Test that the UIDGenerator assigns unique IDs to Nodes."""
        node1 = Node(coords=[0.0, 0.0, 0.0], uid_generator=self.mock_uid_generator)
        node2 = Node(coords=[1.0, 1.0, 1.0], uid_generator=self.mock_uid_generator)
        node3 = Node(coords=[2.0, 2.0, 2.0], uid_generator=self.mock_uid_generator)

        assert node1.uid == 0
        assert node2.uid == 1
        assert node3.uid == 2
        assert self.mock_uid_generator.new.call_count == 3

    def test_node_comparison(self):
        """Test the less-or-equal comparison method for Node objects."""
        node1 = Node(coords=[0.0, 0.0, 0.0], uid_generator=self.mock_uid_generator)
        node2 = Node(coords=[1.0, 1.0, 1.0], uid_generator=self.mock_uid_generator)

        assert node1 <= node2
        assert not (node2 <= node1)
        assert node1 <= node1  # Reflexive property

    def test_node_repr(self):
        """Test the string representation of the Node object."""
        node = Node(coords=[1.0, 1.0, 1.0], uid_generator=self.mock_uid_generator)

        expected_repr = (
            'Node object\n'
            '  uid: 0\n'
            '  coords: [1.0, 1.0, 1.0]\n'
            '  restraint: [False, False, False, False, False, False]\n'
        )
        assert repr(node) == expected_repr

    def test_node_default_visibility(self):
        """Test that the default NodeVisibility object is assigned."""
        node = Node(coords=[0.0, 0.0, 0.0], uid_generator=self.mock_uid_generator)
        assert isinstance(node.visibility, NodeVisibility)

    def test_node_restraint_initialization(self):
        """Test that the restraint attribute is initialized correctly."""
        node = Node(coords=[1.0, 2.0, 3.0], uid_generator=self.mock_uid_generator)
        assert node.restraint == [False] * 6
