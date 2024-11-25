"""Unit tests for UIDGenerator."""

import pytest

from osmg.creators.uid import UIDGenerator
from osmg.model_objects.node import Node


class TestUIDGenerator:
    """Tests for the UIDGenerator class."""

    def setup_method(self) -> None:
        """Set up the UIDGenerator instance for testing."""
        self.generator = UIDGenerator()

    def test_generate_unique_ids(self) -> None:
        """Test that unique IDs are generated."""
        node1 = Node(self.generator, (0.0, 0.0))
        node2 = Node(self.generator, (0.0, 0.0))
        node3 = Node(self.generator, (0.0, 0.0))
        assert node1.uid == 0
        assert node2.uid == 1
        assert node3.uid == 2

    def test_invalid_object_type(self) -> None:
        """Test behavior when an invalid object type is passed."""

        class InvalidObject:
            pass

        invalid_object = InvalidObject()
        with pytest.raises(
            ValueError, match=r'Unknown object class: .*InvalidObject.*'
        ):
            self.generator.new(invalid_object)
