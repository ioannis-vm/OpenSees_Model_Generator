"""
Defines PhysicalMaterial objects.
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

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PhysicalMaterial:
    """
    Physical material.
    We use this for self-weight, plotting enhancements etc.

    Example:
        >>> mat1 = PhysicalMaterial(
        ...     1, 'steel', 'structural',
        ...     7850, 210000, 81000, 400)
        >>> print(mat1)
        Physical material object
        Name: steel
        uid: 1
        variety: structural
        density: 7850
        E: 210000
        G: 81000
        <BLANKLINE>

    """

    uid: int
    name: str
    variety: str
    density: float
    e_mod: float
    g_mod: float
    f_y: float

    def __srepr__(self):
        """
        Short version of repr.

        """

        return f"Physical material: {self.name}, {self.uid}"

    def __repr__(self):
        res = ""
        res += "Physical material object\n"
        res += f"Name: {self.name}\n"
        res += f"uid: {self.uid}\n"
        res += f"variety: {self.variety}\n"
        res += f"density: {self.density}\n"
        res += f"E: {self.e_mod}\n"
        res += f"G: {self.g_mod}\n"
        return res
