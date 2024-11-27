"""objects that create ZeroLength elements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from osmg.model_objects.element import ZeroLength

if TYPE_CHECKING:
    from osmg.core.common import numpy_array
    from osmg.creators.material import MaterialCreator
    from osmg.creators.uid import UIDGenerator
    from osmg.model_objects.uniaxial_material import UniaxialMaterial
    from osmg.node import Node


@dataclass
class ZeroLengthCreator:
    """
    Base class for zero-length creators.

    Handles direction-material assignments using MaterialCreator
    objects.
    """

    uid_generator: UIDGenerator
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
        mats = [creator.generate() for creator in self.material_creators.values()]
        return dirs, mats

    def define_element(
        self,
        node_i: Node,
        node_j: Node,
        x_axis: numpy_array,
        y_axis: numpy_array,
    ) -> ZeroLength:
        """
        Define a zerolength element.

        Returns:
          The added element.
        """
        dirs, mats = self.generate()
        return ZeroLength(
            uid_generator=self.uid_generator,
            nodes=[node_i, node_j],
            mats=mats,
            dirs=dirs,
            vecx=x_axis,
            vecyp=y_axis,
        )
