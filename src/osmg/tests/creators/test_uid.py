"""Unit tests for UIDGenerator."""

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
from osmg.creators.uid import UIDGenerator
from osmg.elements.node import Node


class TestUIDGenerator:
    """Tests for the UIDGenerator class."""

    def setup_method(self):
        """Set up the UIDGenerator instance for testing."""
        self.generator = UIDGenerator()

    def test_generate_unique_ids(self):
        """Test that unique IDs are generated."""
        node1 = Node([0.0, 0.0], self.generator)
        node2 = Node([0.0, 0.0], self.generator)
        node3 = Node([0.0, 0.0], self.generator)
        assert node1.uid == 0
        assert node2.uid == 1
        assert node3.uid == 2

    def test_invalid_object_type(self):
        """Test behavior when an invalid object type is passed."""

        class InvalidObject:
            pass

        invalid_object = InvalidObject()
        with pytest.raises(
            ValueError, match=r'Unknown object class: .*InvalidObject.*'
        ):
            self.generator.new(invalid_object)
