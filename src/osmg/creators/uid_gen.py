"""Objects that generate unique IDs."""

#
#   _|_|      _|_|_|  _|      _|    _|_|_|
# _|    _|  _|        _|_|  _|_|  _|
# _|    _|    _|_|    _|  _|  _|  _|  _|_|
# _|    _|        _|  _|      _|  _|    _|
#   _|_|    _|_|_|    _|      _|    _|_|_|
#
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from dataclasses import dataclass
from itertools import count
from osmg.elements.node import Node


@dataclass
class UIDGenerator:
    """Generates unique identifiers (uids) for various objects."""

    def new(self, thing: object) -> int:
        """
        Generate a new uid for an object based on its category.

        Arguments:
            thing: The object for which to generate a uid.

        Returns:
            A unique identifier for an object of the given type.
        """
        object_type = thing.__class__
        valid_types = {Node: 'node'}

        if object_type not in valid_types:
            msg = f'Unknown object class: {object_type}'
            raise ValueError(msg)

        type_category = valid_types[object_type]
        if hasattr(self, type_category):
            res = next(getattr(self, type_category))
            assert isinstance(res, int)
        else:
            setattr(self, type_category, count(0))
            res = next(getattr(self, type_category))
            assert isinstance(res, int)
        return res
