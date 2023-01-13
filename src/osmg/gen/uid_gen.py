"""
Objects that generate unique IDs

"""

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


@dataclass
class UIDGenerator:
    """
    Generates unique identifiers (uids) for various objects.

    """

    def new(self, thing: str) -> int:
        """
        Generates a new uid for an object of the given type.

        Arguments:
            object_type: The type of object for which to generate a uid.

        Returns:
            A unique identifier for an object of the given type.

        Example:
            >>> from osmg.gen.uid_gen import UIDGenerator
            >>> generator = UIDGenerator()
            >>> generator.new('node')
            0
            >>> generator.new('node')
            1
            >>> generator.new('element')
            0
            >>> generator.new('element')
            1

        """
        if hasattr(self, thing):
            res = next(getattr(self, thing))
            assert isinstance(res, int)
        else:
            setattr(self, thing, count(0))
            res = next(getattr(self, thing))
            assert isinstance(res, int)
        return res
