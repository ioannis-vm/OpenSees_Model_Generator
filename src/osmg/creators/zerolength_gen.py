"""Objects that generate ZeroLength elements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from osmg.elements.element import ZeroLength

if TYPE_CHECKING:
    from osmg.component_assemblies import ComponentAssembly
    from osmg.creators.material_gen import MaterialGenerator
    from osmg.model import Model
    from osmg.node import Node
    from osmg.elements.uniaxial_material import UniaxialMaterial


nparr = npt.NDArray[np.float64]

#
#   _|_|      _|_|_|  _|      _|    _|_|_|
# _|    _|  _|        _|_|  _|_|  _|
# _|    _|    _|_|    _|  _|  _|  _|  _|_|
# _|    _|        _|  _|      _|  _|    _|
#   _|_|    _|_|_|    _|      _|    _|_|_|
#
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator


@dataclass
class ZeroLengthGenerator:
    """
    Base class for zero-length generators.

    Handles direction-material assignments using MaterialGenerator
    objects.
    """

    model: Model
    material_generators: dict[int, MaterialGenerator]

    def generate(self) -> tuple[list[int], list[UniaxialMaterial]]:
        """
        Generate directions and materials.

        Generate directions and materials using the specified material
        generators.

        Returns:
            dirs (list[int]): List of DOF directions.
            mats (list[UniaxialMaterial]): Corresponding uniaxial materials.
        """
        dirs = list(self.material_generators.keys())
        mats = [
            generator.generate(self.model)
            for generator in self.material_generators.values()
        ]
        return dirs, mats

    def define_element(
        self,
        assembly: ComponentAssembly,
        node_i: Node,
        node_j: Node,
        x_axis: nparr,
        y_axis: nparr,
    ) -> ZeroLength:
        """
        Define a zerolength element.

        Returns:
          The added element.
        """
        dirs, mats = self.generate()
        return ZeroLength(
            assembly,
            self.model.uid_generator.new('element'),
            [node_i, node_j],
            mats,
            dirs,
            x_axis,
            y_axis,
        )
