"""
Model Generator for OpenSees ~ model
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

# pylint: disable=W1512

from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from shapely.geometry import Polygon as shapely_Polygon   # type: ignore
from shapely.geometry import Point   # type: ignore

from .gen.uid_gen import UIDGenerator
from . import collections
from .level import Level

if TYPE_CHECKING:
    from .ops.node import Node
    from .ops.element import ElasticBeamColumn
    from .ops.element import DispBeamColumn
    from .ops.element import ZeroLength
    from .ops.element import TwoNodeLink
    from .component_assembly import ComponentAssembly
    from .ops.section import ElasticSection
    from .ops.section import FiberSection
    from .ops.uniaxial_material import UniaxialMaterial
    from .physical_material import PhysicalMaterial

nparr = npt.NDArray[np.float64]


def transfer_component(
        other: Model,
        component: ComponentAssembly):
    """
    Transfers a single component assembly from one model to
    another, assuming the other model was generated with the
    `initialize_empty_copy` method.
    """
    # note: we don't copy the component assemblies and their contents.
    # we just add the same objects to the other model.
    level = component.parent_collection.parent
    other_level = other.levels.retrieve_by_attr('uid', level.uid)
    for node in component.external_nodes.values():
        if node.uid not in other_level.nodes:
            other_level.nodes.add(node)
    other_level.components.add(component)


@dataclass
class Settings:
    """
    General customization of a model.
        imperial_units (bool):
            True for imperial <3:
                in, lb, lb/(in/s2)
            False for SI:
                m, N, kg
        ndm, ndf: change them to break the code.
    """
    imperial_units: bool = field(default=True)  # false for SI
    ndm: int = field(default=3)  # that's all we support
    ndf: int = field(default=6)  # that's all we support

    def __repr__(self):
        res = ''
        res += '~~~ Model Settings ~~~\n'
        res += f'  Imperial units: {self.imperial_units}\n'
        res += f'  ndm           : {self.ndm}\n'
        res += f'  ndf           : {self.ndf}\n'
        return res


@dataclass(repr=False)
class Model:
    """
    Model object.
    Attributes:
        levels (CollectionActive)
        elastic_sections (Collection)
        fiber_sections (Collection)
        physical_materials (PhysicalMaterialCollection)
        component_connectivity (dict[tuple[int, ...], int])
        uid_generator (UIDGenerator)
        settings
    """
    name: str
    levels: collections.CollectionActive[int, Level] = field(
        init=False)
    elastic_sections: collections.Collection[int, ElasticSection] = field(
        init=False)
    fiber_sections: collections.Collection[int, FiberSection] = field(
        init=False)
    uniaxial_materials: collections.Collection[int, UniaxialMaterial] = field(
        init=False)
    physical_materials: collections.Collection[int, PhysicalMaterial] = field(
        init=False)
    uid_generator: UIDGenerator = field(
        default_factory=UIDGenerator)
    settings: Settings = field(default_factory=Settings)

    def __post_init__(self):
        self.levels = collections.CollectionActive(self)
        self.elastic_sections = collections.Collection(self)
        self.fiber_sections = collections.Collection(self)
        self.uniaxial_materials = collections.Collection(self)
        self.physical_materials = collections.Collection(self)

    def __repr__(self):
        res = ''
        res += '~~~ Model Object ~~~\n'
        res += f'ID: {id(self)}\n'
        res += f'levels: {self.levels.__srepr__()}\n'
        res += f'elastic_sections: {self.elastic_sections.__srepr__()}\n'
        res += f'fiber_sections: {self.fiber_sections.__srepr__()}\n'
        res += f'uniaxial_materials: {self.uniaxial_materials.__srepr__()}\n'
        res += f'physical_materials: {self.physical_materials.__srepr__()}\n'
        return res

    def component_connectivity(self) -> dict[tuple[int, ...],
                                             ComponentAssembly]:
        """
        Returns the connectivity of all component
        assemblies. Component assemblies are collections of
        lower-level components that are connected to primary
        nodes. Each component assembly can be represented by a tuple
        of node uids of its connected nodes in ascending order. This
        method returns a dictionary having these tuples as keys, and
        the associated components as values.
        """
        res = {}
        components = self.list_of_components()
        for component in components:
            uids = [node.uid for node in component.external_nodes.values()]
            uids.sort()
            uids_tuple = (*uids,)
            assert uids_tuple not in res, 'Error! Duplicate component found.'
            res[uids_tuple] = component
        return res

    def add_level(self,
                  uid: int,
                  elevation: float):
        """
        Adds a level to the model.
        """
        lvl = Level(self, uid=uid, elevation=elevation)
        self.levels.add(lvl)

    def dict_of_primary_nodes(self):
        """
        Returns a dictionary of all the primary nodes in the model.
        The keys are the uids of the nodes.
        """
        dict_of_nodes: dict[int, Node] = {}
        for lvl in self.levels.values():
            dict_of_nodes.update(lvl.nodes)
        return dict_of_nodes

    def list_of_primary_nodes(self):
        """
        Returns a list of all the primary nodes in the model.
        """
        list_of_nodes = []
        for lvl in self.levels.values():
            for node in lvl.nodes.values():
                list_of_nodes.append(node)
        return list_of_nodes

    def dict_of_internal_nodes(self):
        """
        Returns a dictionary of all the internal nodes in the model.
        The keys are the uids of the nodes.
        """
        dict_of_nodes: dict[int, Node] = {}
        for lvl in self.levels.values():
            for component in lvl.components.values():
                dict_of_nodes.update(component.internal_nodes)
        return dict_of_nodes

    def list_of_internal_nodes(self):
        """
        Returns a list of all the internal nodes in the model.
        """
        list_of_nodes = []
        for lvl in self.levels.values():
            for component in lvl.components.values():
                for inode in component.internal_nodes.values():
                    list_of_nodes.append(inode)
        return list_of_nodes

    def dict_of_all_nodes(self):
        """
        Returns a dictionary of all the nodes in the model.
        The keys are the uids of the nodes.
        """
        dict_of_nodes: dict[int, Node] = {}
        dict_of_nodes.update(self.dict_of_primary_nodes())
        dict_of_nodes.update(self.dict_of_internal_nodes())
        return dict_of_nodes

    def list_of_all_nodes(self):
        """
        Returns a list of all the nodes in the model.
        """
        list_of_nodes = []
        list_of_nodes.extend(self.list_of_primary_nodes())
        list_of_nodes.extend(self.list_of_internal_nodes())
        return list_of_nodes

    def dict_of_components(self):
        """
        Returns a dictionary of all the component assemblies in the
        model.
        The keys are the uids of the component assemblies.
        """
        comps: dict[int, ComponentAssembly] = {}
        for lvl in self.levels.values():
            for component in lvl.components.values():
                comps[component.uid] = component
        return comps

    def list_of_components(self):
        """
        Returns a list of all the component assembiles in the
        model.
        """
        return list(self.dict_of_components().values())

    def dict_of_elastic_beamcolumn_elements(self):
        """
        Returns a dictionary of all ElasticBeamColumn objects in the model.
        The keys are the uids of the objects.
        """
        elems: dict[int, ElasticBeamColumn] = {}
        for lvl in self.levels.values():
            for component in lvl.components.values():
                elems.update(component.elastic_beamcolumn_elements)
        return elems

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
        elems: dict[int, DispBeamColumn] = {}
        for lvl in self.levels.values():
            for component in lvl.components.values():
                elems.update(component.disp_beamcolumn_elements)
        return elems

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
        elems = {}
        elems.update(self.dict_of_elastic_beamcolumn_elements())
        elems.update(self.dict_of_disp_beamcolumn_elements())
        return elems

    def list_of_beamcolumn_elements(self):
        """
        Returns a list of all beamcolumn elements in the model.
        """
        elems = []
        elems.extend(self.list_of_elastic_beamcolumn_elements())
        elems.extend(self.list_of_disp_beamcolumn_elements())
        return elems

    def dict_of_zerolength_elements(self):
        """
        Returns a dictionary of all zerolength elements in the model.
        The keys are the uids of the objects.
        """
        elems: dict[int, ZeroLength] = {}
        for lvl in self.levels.values():
            for component in lvl.components.values():
                elems.update(component.zerolength_elements)
        return elems

    def list_of_zerolength_elements(self):
        """
        Returns a list of all zerolength elements in the model.
        """
        return list(self.dict_of_zerolength_elements().values())

    def dict_of_twonodelink_elements(self):
        """
        Returns a dictionary of all twonodelink elements in the model.
        The keys are the uids of the objects.
        """
        elems: dict[int, TwoNodeLink] = {}
        for lvl in self.levels.values():
            for component in lvl.components.values():
                elems.update(component.twonodelink_elements)
        return elems

    def list_of_twonodelink_elements(self):
        """
        Returns a list of all twonodelink elements in the model.
        """
        return list(self.dict_of_twonodelink_elements().values())

    def bounding_box(self, padding: float) -> tuple[nparr, nparr]:
        """
        Returns the axis-aligned bouding box of the building
        """
        p_min = np.full(3, np.inf)
        p_max = np.full(3, -np.inf)
        for node in self.list_of_primary_nodes():
            point: nparr = np.array(node.coords)
            p_min = np.minimum(p_min, point)
            p_max = np.maximum(p_max, point)
        p_min -= np.full(3, padding)
        p_max += np.full(3, padding)
        return p_min, p_max

    def reference_length(self):
        """
        Returns the largest dimension of the
        bounding box of the building
        (used in graphics)
        """
        p_min, p_max = self.bounding_box(padding=0.00)
        ref_len = np.max(p_max - p_min)
        return ref_len

    def initialize_empty_copy(self, name):
        """
        Initializes a shallow empty copy of the model.
        Used to create subset models.
        """
        res = Model(name)
        # copy the settings attributes
        res.settings.imperial_units = self.settings.imperial_units
        res.settings.ndf = self.settings.ndf
        res.settings.ndm = self.settings.ndm
        # make a copy of the levels
        for lvlkey, lvl in self.levels.items():
            res.add_level(lvlkey, lvl.elevation)
        # giv access to the materials and sections
        res.elastic_sections = self.elastic_sections
        res.fiber_sections = self.fiber_sections
        res.physical_materials = self.physical_materials
        res.uniaxial_materials = self.uniaxial_materials
        return res

    def transfer_by_polygon_selection(
            self,
            other: Model,
            coords: nparr):
        """
        Uses `transfer_component` to transfer all components of which
        the projection to the XY plane falls inside the specified
        polygon.
        """
        all_components = self.list_of_components()
        selected_components = []
        shape = shapely_Polygon(coords)
        for component in all_components:
            accept = True
            nodes = component.external_nodes.values()
            for node in nodes:
                if not shape.contains(Point(node.coords[0:2])):
                    accept = False
                    break
            if accept:
                selected_components.append(component)
        for component in selected_components:
            transfer_component(other, component)
