"""
Model builder for OpenSees ~ Material
"""

#   __                 UC Berkeley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   April 2021
#
# https://github.com/ioannis-vm/OpenSees_Model_Builder

from typing import Optional
from dataclasses import dataclass, field
from functools import total_ordering
from itertools import count
import numpy as np
from grids import GridLine
from node import Node
from utility import common
from utility import transformations
from utility import mesher
from utility import mesher_section_gen

_ids = count(0)


@dataclass
class Material:
    """
    Material object.
    Attributes:
        uniq_id (int): unique identifier
        name (str): Name of the material
        ops_material (str): Name of the material model to use in OpenSees
        density (float): Mass per unit volume of the material
        parameters (dict): Parameters needed to define the material in OpenSees
                           These depend on the meterial model specified.
    """
    name: str
    ops_material: str
    density: float  # mass per unit volume, specified in lb-s**2/in**4
    parameters: dict

    def __post_init__(self):
        self.uniq_id = next(_ids)


@dataclass
class Materials:
    """
    This class is a collector for materials.
    """

    material_list: list[Material] = field(default_factory=list)
    active: Optional[Material] = field(default=None)

    def __post_init__(self):
        """
        Add some default materials used
        to model the connectivity of elements.
        """
        self.material_list.append(Material(
            'fix',
            'Elastic',
            0.00,
            {
                'E': common.STIFF_ROT
            })
        )
        self.material_list.append(Material(
            'release',
            'Elastic',
            0.00,
            {
                'E': common.TINY
            })
        )
        self.material_list.append(Material(
            'steel-UVCuniaxial-fy50',
            'UVCuniaxial',
            490.00/((12.**3)*common.G_CONST),
            {
                'E0': 29000000,
                'Fy': 50000 * 1.1,
                'params': [18000., 10., 18000., 1., 2, 3500., 180., 345., 10.],
                'b': 0.02,
                'G': 11153846.15,
                'b_PZ': 0.02
            })
        )
        self.material_list.append(Material(
            'steel-bilinear-fy50',
            'Steel01',
            490.00/((12.**3)*common.G_CONST),
            {
                'Fy': 50000 * 1.1,
                'E0': 29000000,
                'G': 11153846.15,
                'b': 0.00001,
                'b_PZ': 0.02
            })
        )

        self.material_list.append(Material(
            'steel02-fy50',
            'Steel02',
            490.00/((12.**3)*common.G_CONST),
            {
                'Fy': 50000 * 1.1,
                'E0': 29000000,
                'G':   11153846.15,
                'b': 0.01,
                'params': [19.0, 0.925, 0.15],
                'a1': 0.12,
                'a2': 0.90,
                'a3': 0.18,
                'a4': 0.90,
                'sigInit': 0.00,
                'b_PZ': 0.02
            })
        )

    def add(self, material: Material):
        """
        Add a material in the materials collection,
        if it does not already exist
        """
        if material not in self.material_list:
            self.material_list.append(material)
        else:
            raise ValueError('Material already exists: '
                             + repr(material))

    def set_active(self, name: str):
        """
        Assigns the active material.
        """
        self.active = None
        found = False
        for material in self.material_list:
            if material.name == name:
                self.active = material
                found = True
                break
        if found is False:
            raise ValueError("Material " + name + " does not exist")

    def retrieve(self, name: str):
        """
        Returns the specified material.
        """
        result = None
        for material in self.material_list:
            if material.name == name:
                result = material
                break
        return result

    def __repr__(self):
        out = "Defined materials: " + str(len(self.material_list)) + "\n"
        for material in self.material_list:
            out += repr(material) + "\n"
        return out


