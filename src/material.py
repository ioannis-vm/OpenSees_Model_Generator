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
from itertools import count
from collections import OrderedDict
from utility import common

material_ids = count(0)


@dataclass
class Material:
    """
    Material object.
    Attributes:
        uid (int): unique identifier
        name (str): Name of the material
        ops_material (str): Name of the material model to use in OpenSees
        density (float): Mass per unit volume of the material
        parameters (dict): Parameters needed to define the material in OpenSees
                           These depend on the meterial model specified.
    """
    name: str
    ops_material: str = field(repr=False)
    # mass per unit volume, specified in lb-s**2/in**4
    density: float = field(repr=False)
    parameters: dict = field(repr=False)

    def __post_init__(self):
        self.uid = next(material_ids)


@dataclass
class Materials:
    """
    This class is a collector for materials.
    """

    registry: OrderedDict[str, Material] = field(
        default_factory=OrderedDict, repr=False)
    active: Optional[Material] = field(default=None, repr=False)

    def __post_init__(self):
        """
        Add some default materials used
        to model the connectivity of elements.
        """
        self.registry['fix'] = Material(
            'fix',
            'Elastic',
            0.00,
            {
                'E': common.STIFF_ROT
            })
        self.registry['release'] = Material(
            'release',
            'Elastic',
            0.00,
            {
                'E': common.TINY
            })
        self.registry['steel-UVCuniaxial-fy50'] = Material(
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
        self.registry['steel-bilinear-fy50'] = Material(
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
        self.registry['steel02-fy50'] = Material(
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

    def add(self, material: Material):
        """
        Add a material in the materials collection,
        if it does not already exist
        """
        if material not in self.registry:
            self.registry[material.name] = material
        else:
            raise ValueError(f'Material {material.name} already exists')

    def set_active(self, name: str):
        """
        Assigns the active material.
        """
        material = self.registry.get(name)
        if material is None:
            raise ValueError(f'Undefined material: {name}')
        self.active = material
