"""
Model Builder for OpenSeesPy ~ Level module
"""

#   __                 UC Berkeley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   April 2021
#
# https://github.com/ioannis-vm/OpenSees_Model_Builder

from __future__ import annotations
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Optional
from collections import OrderedDict
import numpy as np
from node import Node, Nodes
from components import LineElement
from components import LineElementSequence_Steel_W_PanelZone
from components import LineElementSequences
from utility import common


# pylint: disable=unsubscriptable-object
# pylint: disable=invalid-name


def previous_element(dct: OrderedDict, key):
    """
    Returns the previous object in an OrderedDict
    given a target key, assuming it is in the OrderedDict.
    If it is not, it returns None.
    If the target key is the first object, it returns None.
    """
    if key in dct:
        key_list = list(dct.keys())
        idx = key_list.index(key)
        if idx == 0:
            ans = None
        else:
            ans = dct[key_list[idx-1]]
    else:
        ans = None
    return ans


@dataclass
@total_ordering
class Level:
    """
    Individual building floor level.
    A level contains building components such as nodes, beams and columns.
    Attributes:
        name (str): Unique name of the level
        elevation (float): Elevation of the level
        restraint (str): Can be any of "free", "pinned" or "fixed".
                         All nodes defined in that level will have that
                         restraint.
        previous_lvl (Level): Points to the level below that level, if
                              the considered level is not the base..
        surface_DL (float): Uniformly distributed dead load of the level.
                            This load can be distributed to the
                            structural members of the level automatically.
                            It is also converted and applied as mass.
        diaphragm (bool): True for a rigid diaphragm, False otherwise.
        nodes_primary (Nodes): Primary nodes of the level. Primary means
                               that these nodes are used to connect
                               components of different elements
                               to them, contrary to being internal
                               nodes of a particular
                               element. A rigid diaphragm constraint can be
                               optionally assigned to these nodes.
        columns (LineElementSequences): Columns of the level.
        beams (LineElementSequences): Beams of the level.
        braces (LineElementSequences): Braces of the level.
        parent_node (Node): If tributary area analysis is done and floors
                            are assumed, a node is created at the
                            center of mass of the level, and acts as the
                            parent node of the rigid diaphragm constraint.
                            The mass in the X-Y direction of all the nodes
                            of that level is then accumulated to that
                            node, together with their contribution in the
                            rotational inertia of the level.
        floor_coordinates (np.ndarray): An array of a sequence of
                          points that define the floor area that is
                          inferred from the beams if tributary area
                          analysis is done.
        floor_bisector_lines (np.ndarray): The lines used to separate
                             the tributary areas, used for plotting.
    """
    name: str
    elevation: float
    restraint: str = field(default="free")
    previous_lvl: Optional[Level] = field(default=None, repr=False)
    surface_DL: float = field(default=0.00, repr=False)
    diaphragm: bool = field(default=False)
    nodes_primary: Nodes = field(default_factory=Nodes, repr=False)
    columns: LineElementSequences = field(
        default_factory=LineElementSequences, repr=False)
    beams: LineElementSequences = field(
        default_factory=LineElementSequences, repr=False)
    braces: LineElementSequences = field(
        default_factory=LineElementSequences, repr=False)
    parent_node: Optional[Node] = field(default=None, repr=False)
    floor_coordinates: Optional[np.ndarray] = field(default=None, repr=False)
    floor_bisector_lines: Optional[list[np.ndarray]] = field(
        default=None, repr=False)

    def __post_init__(self):
        if self.restraint not in ["free", "fixed", "pinned"]:
            raise ValueError('Invalid restraint type: ' + self.restraint)

    def __eq__(self, other):
        return self.name == other.name

    def __le__(self, other):
        return self.elevation <= other.elevation

    def look_for_node(self, x_coord: float, y_coord: float):
        """
        Returns the node that occupies a given point
        at the current level, if it exists
        """
        candidate_pt = np.array([x_coord, y_coord,
                                 self.elevation])
        for other_node in self.nodes_primary.registry.values():
            other_pt = other_node.coords
            if np.linalg.norm(candidate_pt - other_pt) < common.EPSILON:
                return other_node
        return None

    def look_for_beam(self, x_coord: float, y_coord: float):
        """
        Returns a beam if the path of its middle_segment
        crosses the given point.
        """
        candidate_pt = np.array([x_coord, y_coord])
        for beam in self.beams.element_list:
            if beam.middle_segment.crosses_point(candidate_pt):
                return beam
        return None

    def assign_surface_DL(self,
                          load_per_area: float):
        self.surface_DL = load_per_area

    def list_of_primary_nodes(self):
        return self.nodes_primary.registry.values()

    def list_of_all_nodes(self):
        """
        Returns a list containing all the nodes
        of that level *except* the parent node.
        """
        primary = list(self.nodes_primary.registry.values())
        internal = []
        for col in self.columns.element_list:
            internal.extend(col.internal_nodes())
        for bm in self.beams.element_list:
            internal.extend(bm.internal_nodes())
        result = [i for i in primary + internal if i]
        # (to remove Nones if they exist)
        return result

    def list_of_line_elems(self):
        result = []
        for elm in self.beams.element_list + \
                self.columns.element_list + \
                self.braces.element_list:
            if isinstance(elm, LineElement):
                result.append(elm)
        return result

    def list_of_steel_W_panel_zones(self):
        cols = self.columns.element_list
        pzs = []
        for col in cols:
            if isinstance(col, LineElementSequence_Steel_W_PanelZone):
                pzs.append(col.end_segment_i)
        return pzs


@dataclass
class Levels:
    """
    Stores the floor levels of a building.
    No two floor levels can have the same height(no multi-tower support).
    Levels must be defined in order, from the lower elevation
    to the highest.
    Attributes:
        registry (OrderedDict[str, Level]): OrderedDict containing
        unique levels
        active (list[str]): list of active level keys
    """

    registry: OrderedDict[str, Level] = field(
        default_factory=OrderedDict, repr=False)
    active: list[str] = field(default_factory=list)

    def add(self, lvl: Level):
        """
        Adds a new level. The levels must be added in ascending
        elevations.

        Parameters:
            lvl(Level): the level to add
        """
        # Verify level name is unique
        if lvl.name in self.registry:
            raise ValueError('Level name already exists: ' + repr(lvl))
        # Verify level elevation is unique
        if lvl.elevation in [lev.elevation
                             for lev in self.registry.values()]:
            raise ValueError('Level elevation already exists: ' + repr(lvl))

        self.registry[lvl.name] = lvl
        previous_lvl = previous_element(self.registry, lvl.name)
        if previous_lvl:
            lvl.previous_lvl = previous_lvl

    def set_active(self, names: list[str]):
        """
        Sets the active levels (one or more).
        At least one level must be active when defining elements.
        Any element addition or modification call will
        only affect the active levels.
        Args:
            names (list[str]): Names of the levels to set as active
        """
        self.active = []
        if names == "all":
            self.active = list(self.registry.keys())
        elif names == "all_above_base":
            self.active = list(self.registry.keys())[1::]
        else:
            self.active = names
