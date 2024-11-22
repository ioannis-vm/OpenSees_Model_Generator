"""objects that create ZeroLength elements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from osmg.elements.element import ZeroLength

if TYPE_CHECKING:
    from osmg.core.component_assemblies import ComponentAssembly
    from osmg.creators.material import MaterialGenerator
    from osmg.core.model import Model
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
class ZeroLengthCreator:
    """
    Base class for zero-length creators.

    Handles direction-material assignments using MaterialCreator
    objects.
    """

    model: Model
    material_creators: dict[int, MaterialCreator]

    def generate(self) -> tuple[list[int], list[UniaxialMaterial]]:
        """
        Generate directions and materials.

        Generate directions and materials using the specified material
        creators.

        Returns:
            dirs (list[int]): List of DOF directions.
            mats (list[UniaxialMaterial]): Corresponding uniaxial materials.
        """
        dirs = list(self.material_creators.keys())
        mats = [
            creator.generate(self.model)
            for creator in self.material_creators.values()
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
            self.model.uid_creator.new('element'),
            [node_i, node_j],
            mats,
            dirs,
            x_axis,
            y_axis,
        )
