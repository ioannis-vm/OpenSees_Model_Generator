"""
Model Generator for OpenSees ~ physical material
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PhysicalMaterial:
    """
    Physical material.
    We use this for self-weight, plotting enhancements etc.
    """
    uid: int
    name: str
    variety: str
    density: float
    E: float
    G: float
    Fy: float

    def __srepr__(self):
        return f'Physical material: {self.name}, {self.uid}'

    def __repr__(self):
        res = ''
        res += 'Physical material object\n'
        res += f'Name: {self.name}\n'
        res += f'uid: {self.uid}\n'
        res += f'variety: {self.variety}\n'
        res += f'density: {self.density}\n'
        res += f'E: {self.E}\n'
        res += f'G: {self.G}\n'
        return res
