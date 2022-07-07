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
    elastic_beamcolumn_elements: elasticBeamColumnCollection = field(
        init=False, repr=False)
    disp_beamcolumn_elements: dispBeamColumnCollection = field(
        init=False, repr=False)

    def __post_init__(self):
        self.external_nodes = NodeCollection(self)
        self.internal_nodes = NodeCollection(self)
        self.elastic_beamcolumn_elements = elasticBeamColumnCollection(self)
        self.disp_beamcolumn_elements = dispBeamColumnCollection(self)

    def dict_of_elastic_beamcolumn_elements(self):
        res = {}
        for elm in self.elastic_beamcolumn_elements.registry.values():
            res[elm.uid] = elm
        return res

    def list_of_elastic_beamcolumn_elements(self):
        return list(self.dict_of_elastic_beamcolumn_elements())

    def dict_of_disp_beamcolumn_elements(self):
        res = {}
        for elm in self.disp_beamcolumn_elements.registry.values():
            res[elm.uid] = elm
        return res

    def list_of_disp_beamcolumn_elements(self):
        return list(self.dict_of_disp_beamcolumn_elements())

    def dict_of_all_elements(self):
        res = {}
        res.update(self.dict_of_elastic_beamcolumn_elements())
        res.update(self.dict_of_disp_beamcolumn_elements())
        return res

    def list_of_all_elements(self):
        return list(self.dict_of_all_elements().values())
    
    def element_connectivity(self):
        res = {}
        elms = self.list_of_all_elements()
        for elm in elms:
            uids = [nd.uid for nd in elm.eleNodes]
            uids.sort()
            uids_tuple = (*uids,)
            res[uids_tuple] = elm
        return res

