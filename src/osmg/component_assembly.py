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
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from . import collections
from .ops import element


nparr = npt.NDArray[np.float64]


@dataclass
class ComponentAssembly:
    """
    A component assembly represents some part of a structure and holds
    various lower-level elements such as nodes and beamcolumn
    elements.
    Attributes:
      uid (int): Unique identifyer of the component assembly
      parent_collection (ComponentCollection): The collection of
        elements to which the component assembly belongs.
      component_purpose (str): The functionality of the component assembly
      external_nodes (NodeCollection): the external nodes to which the
        component assembly is connected.
        these nodes should exist as part of a level.
      internal_nodes (NodeCollection): internal nodes that are
        required for the connectivity of the elements of the component
        assembly.
        these nodes only exist as part of the component assembly.
      elastic_beamcolumn_elements (CollectionWithConnectivity): ...
      disp_beamcolumn_elements (CollectionWithConnectivity): ...
      zerolength_elements (CollectionWithConnectivity): ...
    """
    uid: int
    parent_collection: collections.Collection[int, ComponentAssembly]
    component_purpose: str
    external_nodes: collections.NodeCollection = field(
        init=False)
    internal_nodes: collections.NodeCollection = field(
        init=False)
    elastic_beamcolumn_elements: (collections.CollectionWithConnectivity[
        int, element.ElasticBeamColumn]) = field(init=False)
    disp_beamcolumn_elements: (collections.CollectionWithConnectivity[
        int, element.DispBeamColumn]) = field(init=False)
    zerolength_elements: (collections.CollectionWithConnectivity[
        int, element.ZeroLength]) = field(init=False)

    def __post_init__(self):
        self.external_nodes = collections.NodeCollection(self)
        self.internal_nodes = collections.NodeCollection(self)
        self.elastic_beamcolumn_elements = \
            collections.CollectionWithConnectivity(self)
        self.disp_beamcolumn_elements = \
            collections.CollectionWithConnectivity(self)
        self.zerolength_elements = collections.CollectionWithConnectivity(self)

    def __srepr__(self):
        """
        Short version of repr
        """
        return f'Component assembly, uid: {self.uid}'

    def __repr__(self):
        res = ''
        res += 'Component assembly object\n'
        res += f'uid: {self.uid}\n'
        res += f'component_purpose: {self.component_purpose}\n'
        res += 'External Nodes\n'
        for node in self.external_nodes.values():
            res += f'  {node.uid}, {node.coords}'
        res += 'Internal Nodes\n'
        for node in self.internal_nodes.values():
            res += f'  {node.uid}, {node.coords}'
        return res

    def dict_of_elastic_beamcolumn_elements(self):
        """
        Returns a dictionary of all ElasticBeamColumn objects in the model.
        The keys are the uids of the objects.
        """
        res = {}
        for elm in self.elastic_beamcolumn_elements.values():
            res[elm.uid] = elm
        return res

    def list_of_elastic_beamcolumn_elements(self):
        """
        Returns a list of all ElasticBeamColumn objects in the model.
        """
        return list(self.dict_of_elastic_beamcolumn_elements().values())

    def dict_of_disp_beamcolumn_elements(self):
        """
        Returns a dictionary of all DispBeamColumn objects in the model.
        The keys are the uids of the objects.
        """
        res = {}
        for elm in self.disp_beamcolumn_elements.values():
            res[elm.uid] = elm
        return res

    def list_of_disp_beamcolumn_elements(self):
        """
        Returns a list of all DispBeamColumn objects in the model.
        """
        return list(self.dict_of_disp_beamcolumn_elements().values())

    def dict_of_beamcolumn_elements(self):
        """
        Returns a dictionary of all beamcolumn elements in the model.
        The keys are the uids of the objects.
        """
        res = {}
        res.update(self.dict_of_elastic_beamcolumn_elements())
        res.update(self.dict_of_disp_beamcolumn_elements())
        return res

    def list_of_all_elements(self):
        """
        Returns a list of all beamcolumn elements in the model.
        """
        return list(self.dict_of_beamcolumn_elements().values())

    def element_connectivity(self):
        """
        Returns the connectivity of all elements. Elements are
        connected to external nodes. Each component assembly can be
        represented by a typle of node uids of its connected nodes in
        ascending order. This method returns a dictionary having these
        tuples as keys, and the associated components as values.
        """
        res = {}
        elms = self.list_of_all_elements()
        for elm in elms:
            uids = [nd.uid for nd in elm.eleNodes]
            uids.sort()
            uids_tuple = (*uids,)
            res[uids_tuple] = elm
        return res
