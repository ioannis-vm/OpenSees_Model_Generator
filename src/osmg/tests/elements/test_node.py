"""Unit tests for the Node class."""

from unittest.mock import MagicMock

from osmg.model_objects.node import Node


class TestNode:
    """Tests for the Node class."""

    def setup_method(self) -> None:
        """Set up a mock UIDGenerator and other common fixtures."""
        self.mock_uid_generator = MagicMock()
        self.mock_uid_generator.new.side_effect = [0, 1, 2]

    def test_node_initialization(self) -> None:
        """Test that the Node object initializes correctly."""
        node = Node(
            coordinates=(0.0, 0.0, 0.0), uid_generator=self.mock_uid_generator
        )

        assert node.coordinates == (0.0, 0.0, 0.0)
        assert node.uid == 0
        self.mock_uid_generator.new.assert_called_once_with(node)

    def test_node_uid_generation(self) -> None:
        """Test that the UIDGenerator assigns unique IDs to Nodes."""
        node1 = Node(
            coordinates=(0.0, 0.0, 0.0), uid_generator=self.mock_uid_generator
        )
        node2 = Node(
            coordinates=(1.0, 1.0, 1.0), uid_generator=self.mock_uid_generator
        )
        node3 = Node(
            coordinates=(2.0, 2.0, 2.0), uid_generator=self.mock_uid_generator
        )

        assert node1.uid == 0
        assert node2.uid == 1
        assert node3.uid == 2
        assert self.mock_uid_generator.new.call_count == 3

    def test_node_comparison(self) -> None:
        """Test the less-or-equal comparison method for Node objects."""
        node1 = Node(
            coordinates=(0.0, 0.0, 0.0), uid_generator=self.mock_uid_generator
        )
        node2 = Node(
            coordinates=(1.0, 1.0, 1.0), uid_generator=self.mock_uid_generator
        )

        assert node1 <= node2
        assert not (node2 <= node1)
        assert node1 <= node1  # noqa: PLR0124 Reflexive property

    def test_node_repr(self) -> None:
        """Test the string representation of the Node object."""
        node = Node(
            coordinates=(1.0, 1.0, 1.0), uid_generator=self.mock_uid_generator
        )

        expected_repr = 'Node object\n  uid: 0\n  coordinates: (1.0, 1.0, 1.0)\n'
        assert repr(node) == expected_repr
