"""
Model Generator for OpenSees ~ component assembly
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
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing import Optional

import numpy as np
import numpy.typing as npt

from .collections import NodeCollection
from .collections import elasticBeamColumnCollection
from .collections import dispBeamColumnCollection
from .collections import ComponentCollection

nparr = npt.NDArray[np.float64]


@dataclass
class ComponentAssembly:
    """

    """
    uid: int
    parent_collection: ComponentCollection = field(repr=False)
    external_nodes: NodeCollection = field(
        init=False, repr=False)
    internal_nodes: NodeCollection = field(
        init=False, repr=False)
    element_connectivity: dict[tuple[str, ...], int] = field(
        default_factory=dict, repr=False)
    elastic_beamcolumn_elements: elasticBeamColumnCollection = field(
        init=False, repr=False)
    force_beamcolumn_elements: dispBeamColumnCollection = field(
        init=False, repr=False)

    def __post_init__(self):
        self.external_nodes = NodeCollection(self)
        self.internal_nodes = NodeCollection(self)
        self.elastic_beamcolumn_elements = elasticBeamColumnCollection(self)
        self.force_beamcolumn_elements = dispBeamColumnCollection(self)
