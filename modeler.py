"""
Building Modeler for OpenSeesPy ~ Modeler module
"""

#   __                 UC Berkeley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   April 2021
#
# https://github.com/ioannis-vm/OpenSeesPy_Building_Modeler

from __future__ import annotations
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Optional
import json
import numpy as np
from grids import GridLine
from components import Node, Nodes
from components import LineElement
from components import EndRelease
from components import LineElementSequence
from components import LineElementSequence_Fixed
from components import LineElementSequence_Pinned
from components import LineElementSequence_FixedPinned
from components import LineElementSequence_RBS
from components import LineElementSequence_RBS_j
from components import LineElementSequence_IMK
from components import LineElementSequence_Steel_W_PanelZone
from components import LineElementSequence_Steel_W_PanelZone_IMK
from components import LineElementSequence_W_grav_sear_tab
from components import LineElementSequences
from components import Sections, Materials
from utility import common
from utility import trib_area_analysis
from utility import mesher
from utility.graphics import preprocessing_3D
from utility.graphics import preprocessing_2D


# pylint: disable=unsubscriptable-object
# pylint: disable=invalid-name


def previous_element(lst: list, obj):
    """
    Returns the previous object in a list
    given a target object, assuming it is in the list.
    If it is not, it returns None.
    If the target is the first object, it returns None.
    """
    try:
        idx = lst.index(obj)
    except ValueError:
        return None
    if idx == 0:
        return None
    return lst[idx - 1]


def point_exists_in_list(pt: np.ndarray,
                         pts: list[np.ndarray]) -> bool:
    """
    Determines whether a given list containing points
    (represented with numpy arrays) contains a point
    that is equal (with a fudge factor) to a given point.
    Args:
        pt (np.ndarray): A numpy array to look for
        pts (list[np.ndarray]): A list to search for pt
    """
    for other in pts:
        dist = np.linalg.norm(pt - other)
        if dist < common.EPSILON:
            return True
    return False


@dataclass
class GridSystem:
    """
    This class is a collector for the gridlines, and provides
    methods that perform operations using gridlines.
    """

    grids: list[GridLine] = field(default_factory=list)

    def get(self, gridline_tag: str):
        """
        Returns the gridline with the given tag,
        or None if there is no gridline with the
        specified tag.
        """
        result = None
        for gridline in self.grids:
            if gridline.tag == gridline_tag:
                result = gridline
        return result

    def add(self, grdl: "GridLine"):
        """
        Add a gridline in the grid system,
        if it is not already in
        """
        if grdl not in self.grids:
            self.grids.append(grdl)
        else:
            raise ValueError('Gridline already exists: '
                             + repr(grdl))
        self.grids.sort()

    def remove(self, grdl: "GridLine"):
        """
        Remove a gridline from the grid system
        """
        self.grids.remove(grdl)

    def clear(self, tags: list[str]):
        """
        Removes the gridlines in the given list,
        specified by their tag.
        """
        for tag in tags:
            grdl = self.get(tag)
            self.grids.remove(grdl)

    def clear_all(self):
        """
        Removes all gridlines.
        """
        self.grids = []

    def intersection_points(self):
        """
        Returns a list of all the points
        defined by gridline intersections
        """
        pts = []  # intersection points
        for i, grd1 in enumerate(self.grids):
            for j in range(i+1, len(self.grids)):
                grd2 = self.grids[j]
                pt = grd1.intersect(grd2)
                if pt is not None:  # if an intersection point exists
                    # and is not already in the list
                    if not point_exists_in_list(pt, pts):
                        pts.append(pt)
        return pts

    def intersect(self, grd: GridLine):
        """
        Returns a list of all the points
        defined by the intersection of a given
        gridline with all the other gridlines
        in the gridsystem
        """
        pts = []  # intersection points
        for other_grd in self.grids:
            # ignore current grid
            if other_grd == grd:
                continue
            # get the intersection point, if any
            pt = grd.intersect(other_grd)
            if pt is not None:  # if there is an intersection
                # and is not already in the list
                if not point_exists_in_list(pt, pts):
                    pts.append(pt)
            # We also need to sort the list.
            # We do this by sorting the instersection points
            # by their distance from the current gridline's
            # starting point.
            distances = [np.linalg.norm(pt-grd.start_np)
                         for pt in pts]
            pts = [x for _, x in sorted(zip(distances, pts))]
        return pts

    def __repr__(self):
        out = "The building has " + \
            str(len(self.grids)) + " gridlines\n"
        for grd in self.grids:
            out += repr(grd) + "\n"
        return out


@dataclass
@total_ordering
class Group:
    """
    This class is be used to group together
    elements of any kind.
    """

    name: str
    elements: list = field(init=False, repr=False)

    def __post_init__(self):
        self.elements = []

    def __eq__(self, other):
        return self.name == other.name

    def __le__(self, other):
        return self.name <= other.name

    def add(self, element):
        """
        Add an element in the group,
        if it is not already in
        """
        if element not in self.elements:
            self.elements.append(element)

    def remove(self, element):
        """
        Remove something from the group
        """
        self.elements.remove(element)

    def __repr__(self):
        return(
            "Group(name=" + self.name + "): "
            + str(len(self.elements)) + " elements.")


@dataclass
class Groups:
    """
    Stores the  groups of a building.
    No two groups can have the same name.
    Elements can belong in multiple groups.
    """

    group_list: list[Group] = field(default_factory=list)
    active:     list[Group] = field(default_factory=list)

    def add(self, grp: Group):
        """
        Adds a new element group

        Parameters:
            grp(Group): the element group to add
        """
        # Verify element group name is unique
        if grp in self.group_list:
            raise ValueError('Group name already exists: ' + repr(grp))
        # Append the new element group in the list
        self.group_list.append(grp)
        # Sort the element groups in ascending order (name-wise)
        self.group_list.sort()

    def retrieve_by_name(self, name: str) -> Group:
        """
        Returns a variable pointing to the group that has the
        given name.
        Args:
            name (str): Name of the group to retrieve
        Returns:
            group (Group)
        """
        for grp in self.group_list:
            if grp.name == name:
                return grp
        raise ValueError("Group " + name + " does not exist")

    def set_active(self, names: list[str]):
        """
        Specifies the active groups(one or more).
        Adding any element to the building will also
        add that element to the active groups.
        The active groups can also be set to an empty list.
        In that case, new elements will not be added toa any groups.
        Args:
            names (list[str]): Names of groups to set as active
        """
        self.active = []
        for name in names:
            grp = self.retrieve_by_name(name)
            self.active.append(grp)

    def add_element(self, element):
        """
        Adds an element to all active groups.
        """
        for grp in self.active:
            grp.add(element)

    def __repr__(self):
        out = "The building has " + \
            str(len(self.group_list)) + " groups\n"
        for grp in self.group_list:
            out += repr(grp) + "\n"
        return out


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
        for other_node in self.nodes_primary.node_list:
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
        return self.nodes_primary.node_list

    def list_of_all_nodes(self):
        """
        Returns a list containing all the nodes
        of that level *except* the parent node.
        """
        primary = self.nodes_primary.node_list
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
        level_list (list[Level]): list containing unique levels
        active (list[Level]): list of active levels
    """

    level_list: list[Level] = field(default_factory=list)
    active: list[Level] = field(default_factory=list)

    def add(self, lvl: Level):
        """
        Adds a new level. The levels must be added in ascending
        elevations.

        Parameters:
            lvl(Level): the level to add
        """
        # Verify level name is unique
        if lvl in self.level_list:
            raise ValueError('Level name already exists: ' + repr(lvl))
        # Verify level elevation is unique
        if lvl.elevation in [lev.elevation
                             for lev in self.level_list]:
            raise ValueError('Level elevation already exists: ' + repr(lvl))
        # Don't accept levels out of order
        if self.level_list:
            if lvl.elevation < self.level_list[-1].elevation:
                raise ValueError(
                    'Levels should be defined from the bottom up for now..')
        # Append the new level in the level list
        self.level_list.append(lvl)
        previous_lvl = previous_element(self.level_list, lvl)
        if previous_lvl:
            lvl.previous_lvl = previous_lvl

        # If there's no active level, make
        # the newly added level active
        if not self.active:
            self.active.append(lvl)
            self.active.sort()

    def retrieve_by_name(self, name: str):
        """"
        Returns a variable pointing to the level that has the
        given name.
        Args:
            name (str): Name of the level to retrieve
        Returns:
            level (Level)
        """
        for lvl in self.level_list:
            if lvl.name == name:
                return lvl
        raise ValueError("Level " + name + " does not exist")

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
            self.active = self.level_list
        elif names == "all_above_base":
            self.active = self.level_list[1::]
        else:
            for name in names:
                retrieved_level = self.retrieve_by_name(name)
                if retrieved_level not in self.active:
                    self.active.append(retrieved_level)

    def __repr__(self):
        out = "The building has " + \
            str(len(self.level_list)) + " levels\n"
        for lvl in self.level_list:
            out += repr(lvl) + "\n"
        return out


@dataclass
class Selection:
    """
    This class enables the ability to select elements
    to modify them.

    """
    nodes: Nodes = field(default_factory=Nodes, repr=False)
    beams: LineElementSequences = field(
        default_factory=LineElementSequences, repr=False)
    columns: LineElementSequences = field(
        default_factory=LineElementSequences, repr=False)
    braces: LineElementSequences = field(
        default_factory=LineElementSequences, repr=False)
    line_elements: list[LineElement] = field(
        default_factory=list, repr=False)

    def clear(self):
        """
        Clears all selected elements.
        """
        self.nodes = Nodes()
        self.beams = LineElementSequences()
        self.columns = LineElementSequences()
        self.braces = LineElementSequences()
        self.line_elements = []

    #############################################
    # Methods that modify selected elements     #
    #############################################
    def add_UDL(self, udl: np.ndarray):
        """
        Adds the specified UDL to the selected
        line elements.
        """
        for line_element in self.line_elements:
            line_element.add_udl_glob(udl, ltype='other')

    #############################################
    # Methods that return objects               #
    #############################################

    def list_of_line_element_sequences(self):
        """
        Returns all selected LineElementSequences.
        """
        return self.beams.element_list + \
            self.columns.element_list + self.braces.element_list

    def list_of_line_elements(self):
        sequences = self.list_of_line_element_sequences()
        result = []
        for sequence in sequences:
            for elm in sequence:
                if isinstance(elm, LineElement):
                    result.append(elm)
        result.extend(self.line_elements)
        return result

    def list_of_primary_nodes(self):
        """
        Returns a list of unique primary nodes on which all the
        selected elements are connected to.
        """
        gather = []
        for elem in self.list_of_line_element_sequences():
            gather.extend(elem.primary_nodes())
        # remove duplicates
        result = []
        return [result.append(x) for x in gather if x not in gather]

    def list_of_internal_nodes(self):
        """
        Returns a list of all secondary nodes that exist
        in the selected elements.
        """
        result = []
        for elem in self.list_of_line_element_sequences():
            result.extend(elem.internal_nodes())
        return result


@dataclass
class Building:
    """
    This class manages building objects.
    Attributes:
        gridsystem (GridSystem): Gridsystem used to
                   define or modify elements.
        levels (Levels): Levels of the building
        groups (Groups): Groups of the building
        sections (Sections): Sections used
        materials (Materials): Materials used
        active_placement (str): Placement parameter to use
                          for newly defined elements
                          where applicable (see Section).
        active_angle (float): Angle parameter to use for
                          newly defined elements.

    """
    gridsystem: GridSystem = field(default_factory=GridSystem)
    levels: Levels = field(default_factory=Levels)
    groups: Groups = field(default_factory=Groups)
    sections: Sections = field(default_factory=Sections)
    materials: Materials = field(default_factory=Materials)
    selection: Selection = field(default_factory=Selection)
    active_placement: str = field(default='centroid')
    active_angle: float = field(default=0.00)
    global_restraints: list = field(default_factory=list)

    ###############################################
    # 'Add' methods - add objects to the building #
    ###############################################

    def add_node(self,
                 x: float,
                 y: float) -> list[Node]:
        """
        Adds a node at a particular point in all active levels.
        Returns all added nodes.
        """
        added_nodes = []
        for level in self.levels.active:
            node = Node([x, y,
                         level.elevation], level.restraint)
            level.nodes_primary.add(node)
            added_nodes.append(node)
        return added_nodes

    def add_level(self,
                  name: str,
                  elevation: float,
                  restraint: str = "free"
                  ) -> Level:
        """
        Adds a level to the building.
        Levels must be defined in increasing elevations.
        Args:
            name (str): Unique name of the level
            elevation (float): Elevation of the level
            restraint (str): Can be any of "free", "pinned" or "fixed".
                             All nodes defined in that level will have that
                             restraint.
        """
        level = Level(name, elevation, restraint)
        self.levels.add(level)
        return level

    def add_gridline(self,
                     tag: str,
                     start: list[float],
                     end: list[float]
                     ) -> GridLine:
        """
        Adds a new gridline to the building.
        Args:
           tag (str): Name of the gridline
           start (list(float]): X,Y coordinates of starting point
           end ~ similar to start
        Regurns:
            gridline object
        """
        gridline = GridLine(tag, start, end)
        self.gridsystem.add(gridline)
        return gridline

    def add_sections_from_json(self,
                               filename: str,
                               sec_type: str,
                               labels: list[str]):
        """
        Add sections from a section database json file.
        Only the specified sections(given the labels) are added,
        even if more are present in the file.
        Args:
            filename (str): Path of the file
            sec_type (str): Section type to be assigned
                            to all the defined sections
                            (see sections).
                            I.e. don't import W and HSS
                            sections at once!
            labels (list[str]): Names of the sections to add.
        """
        if not self.materials.active:
            raise ValueError("No active material specified")
        if sec_type == 'W':
            with open(filename, "r") as json_file:
                section_dictionary = json.load(json_file)
            for label in labels:
                try:
                    sec_data = section_dictionary[label]
                except KeyError:
                    raise KeyError("Section " + label + " not found in file.")
                self.sections.generate_W(label,
                                         self.materials.active,
                                         sec_data)
        if sec_type == "HSS":
            with open(filename, "r") as json_file:
                section_dictionary = json.load(json_file)
            for label in labels:
                try:
                    sec_data = section_dictionary[label]
                except KeyError:
                    raise KeyError("Section " + label + " not found in file.")
                self.sections.generate_HSS(label,
                                           self.materials.active,
                                           sec_data)

    def add_gridlines_from_dxf(self,
                               dxf_file: str) -> list[GridLine]:
        """
        Parses a given DXF file and adds gridlines from
        all the lines defined in that file.
        Args:
            dxf_file (str): Path of the DXF file.
        Returns:
            grds (list[GridLine]): Added gridlines
        """
        i = 100000  # anything > 8 works
        j = 0
        xi = 0.00
        xj = 0.00
        yi = 0.00
        yj = 0.00
        grds = []
        with open(dxf_file, 'r') as f:
            while True:
                ln = f.readline()
                if ln == "":
                    break
                ln = ln.strip()
                if ln == "AcDbLine":
                    i = 0
                if i == 2:
                    xi = float(ln)
                if i == 4:
                    yi = float(ln)
                if i == 6:
                    xj = float(ln)
                if i == 8:
                    yj = float(ln)
                    grd = self.add_gridline(str(j), [xi, yi], [xj, yj])
                    grds.append(grd)
                    j += 1
                i += 1
        return grds

    def add_group(self, name: str) -> Group:
        """
        Adds a new group to the building.
        Args:
            name: Name of the group to be added.
        Returns:
            group (Group): Added group.
        """
        group = Group(name)
        self.groups.add(group)
        return group

    def add_column_at_point(self,
                            x: float,
                            y: float,
                            n_sub=1,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear', ends={'type': 'fixed'},
                            metadata=None) \
            -> list[LineElementSequence]:
        """
        Adds a vertical column at the given X, Y
        location at all the active levels.
        Existing nodes are used, otherwise they are created.
        Args:
            x (float): X coordinate in the global system
            y (float): Y coordinate in the global system
            n_sub (int): Number of internal elements to add
            model_as (dict): Either
                           {'type': 'elastic'}
                           or
                           {'type': 'fiber', 'n_x': n_x, 'n_y': n_y}
            geomTransf: {Linear, PDelta}
            ends (dict): {'type': 'fixed, 'dist': float}', or
                         {'type': 'pinned', 'dist': float} or
                         {'type': 'fixed-pinned', 'dist': float} or
                         {'type': 'RBS', 'dist': float,
                          'length': float, 'factor': float, 'n_sub': int}

        Returns:
            columns (list[LineElementSequence]): Added columns.
        """
        if not self.sections.active:
            raise ValueError("No active section")
        columns = []
        for level in self.levels.active:
            if level.previous_lvl:  # if previous level exists
                # check to see if top node exists
                top_node = level.look_for_node(x, y)
                # create it if it does not exist
                if not top_node:
                    top_node = Node(
                        np.array([x, y, level.elevation]), level.restraint)
                    level.nodes_primary.add(top_node)
                # check to see if bottom node exists
                bot_node = level.previous_lvl.look_for_node(
                    x, y)
                # create it if it does not exist
                if not bot_node:
                    bot_node = Node(
                        np.array([x, y, level.previous_lvl.elevation]),
                        level.previous_lvl.restraint)
                    level.previous_lvl.nodes_primary.add(bot_node)
                # add the column connecting the two nodes
                if ends['type'] in ['fixed', 'steel_W_PZ', 'steel_W_PZ_IMK']:
                    # if ends['type'] is 'steel_W_PZ' or
                    # 'steel_W_IMK_PZ' then
                    # LineElementSequence_Fixed will be replaced with
                    # a LineElementSequence_Steel_W_PanelZone or
                    # a LineElementSequence_Steel_W_PanelZone_IMK during
                    # preprocessing. See the preprocess() method.
                    metadata = {'ends': ends}
                    column = LineElementSequence_Fixed(
                        node_i=top_node,
                        node_j=bot_node,
                        ang=self.active_angle,
                        offset_i=np.zeros(3),
                        offset_j=np.zeros(3),
                        section=self.sections.active,
                        n_sub=n_sub,
                        model_as=model_as,
                        geomTransf=geomTransf,
                        placement=self.active_placement,
                        end_dist=0.05, metadata=metadata)
                elif ends['type'] == 'pinned':
                    column = LineElementSequence_Pinned(
                        node_i=top_node,
                        node_j=bot_node,
                        ang=self.active_angle,
                        offset_i=np.zeros(3),
                        offset_j=np.zeros(3),
                        section=self.sections.active,
                        n_sub=n_sub,
                        model_as=model_as,
                        geomTransf=geomTransf,
                        placement=self.active_placement,
                        end_dist=ends['dist'],
                        metadata=metadata,
                        mat_fix=self.materials.retrieve('fix'),
                        camber=0.00)
                elif ends['type'] == 'fixed-pinned':
                    column = LineElementSequence_FixedPinned(
                        node_i=top_node,
                        node_j=bot_node,
                        ang=self.active_angle,
                        offset_i=np.zeros(3),
                        offset_j=np.zeros(3),
                        section=self.sections.active,
                        n_sub=n_sub,
                        model_as=model_as,
                        geomTransf=geomTransf,
                        placement=self.active_placement,
                        end_dist=ends['dist'],
                        metadata=metadata,
                        mat_fix=self.materials.retrieve('fix'),
                        mat_release=self.materials.retrieve('release'),
                        camber=0.00)
                else:
                    raise ValueError('Invalid end-type')
                ok = level.columns.add(column)
                if ok:
                    columns.append(column)
                    self.groups.add_element(column)
                    top_node.column_below = column
                    bot_node.column_above = column
        return columns

    def add_beam_at_points(self,
                           start: np.ndarray,
                           end: np.ndarray,
                           n_sub=1,
                           offset_i=np.zeros(shape=3).copy(),
                           offset_j=np.zeros(shape=3).copy(),
                           snap_i="centroid",
                           snap_j="centroid",
                           ends={'type': 'fixed'},
                           model_as={'type': 'elastic'},
                           geomTransf='Linear'):
        """
        Adds a beam connecting the given points
        at all the active levels.
        Existing nodes are used, otherwise they are created.
        Args:
            start (np.ndarray): X,Y coordinates of point i
            end (np.ndarray): X,Y coordinates of point j
            n_sub (int): Number of internal elements to add
            offset_i (np.ndarray): X,Z,Y components of a
                      vector that starts at node i and goes
                      to the internal end of the rigid offset
                      of the i-side of the beam.
            offset_j ~ similar to offset i, for the j side.
            snap_i (str): Tag used to infer an offset based
                          on the section of an existing column
                          at node i
            snap_j ~ similar to snap_i, for the j side.
            ends (dict): {'type': 'fixed, 'dist': float}', or
                         {'type': 'pinned', 'dist': float} or
                         {'type': 'fixed-pinned', 'dist': float} or
                         {'type': 'RBS', 'dist': float,
                          'length': float, 'factor': float, 'n_sub': int}
                        For the pinned or fixed-pinned case,
                          `dist` represents the
                            proportion of the distance between the element's
                            ends to the release, relative to the element's
                            length, both considered without the offsets.
                        For the RBS case,
                          `dist` is similar to the pinned case,
                            but it corresponds to the start of the RBS portion.
                          `length` represents the length of the RBS portion.
                          `factor` represents the proportion of the RBS
                            section's width, relative to the original section.
                          `n_sub` represents how many LineElements should be
                            used for the RBS portion
            model_as (dict): Either
                           {'type': 'elastic'}
                           or
                           {'type': 'fiber', 'n_x': n_x, 'n_y': n_y}
            geomTransf: {Linear, PDelta}
        Returns:
            beams (list[LineElementSequence]): added beams.
        """

        if not self.sections.active:
            raise ValueError("No active section specified")
        beams = []
        for level in self.levels.active:

            # - #
            # i #
            # - #
            # check to see if start node exists
            start_node = level.look_for_node(*start)
            if start_node:
                # check if there is a column at node i
                col = start_node.column_below
                if col:
                    o_s_i = col.snap_offset(snap_i)
                    connection_offset_i = col.offset_i - o_s_i
                else:
                    connection_offset_i = np.zeros(3)
            else:
                # check to see if a beam crosses that point
                start_beam = level.look_for_beam(*start)
                if start_beam:
                    start_node, connection_offset_i = \
                        start_beam.middle_segment.connect(
                            start, level.elevation)
                else:
                    # no start node or crossing beam found
                    # create a start node
                    start_node = Node(
                        np.array([*start, level.elevation]), level.restraint)
                    level.nodes_primary.add(start_node)
                    connection_offset_i = np.zeros(3)
            # - #
            # j #
            # - #
            # check to see if end node exists
            end_node = level.look_for_node(*end)
            if end_node:
                # check if there is a column at node j
                col = end_node.column_below
                if col:
                    o_s_j = col.snap_offset(snap_j)
                    connection_offset_j = col.offset_j - o_s_j
                else:
                    connection_offset_j = np.zeros(3)
            else:
                # check to see if a beam crosses that point
                end_beam = level.look_for_beam(*end)
                if end_beam:
                    end_node, connection_offset_j = \
                        end_beam.middle_segment.connect(
                            end, level.elevation)
                else:
                    # no end node or crossing beam found
                    # create an end node
                    end_node = Node(
                        np.array([*end, level.elevation]), level.restraint)
                    level.nodes_primary.add(end_node)
                    connection_offset_j = np.zeros(3)

            # ---------------- #
            # element creation #
            # ---------------- #
            # add the beam connecting the two nodes
            if ends['type'] == 'fixed':
                beam = LineElementSequence_Fixed(
                    node_i=start_node,
                    node_j=end_node,
                    ang=self.active_angle,
                    offset_i=offset_i+connection_offset_i,
                    offset_j=offset_j+connection_offset_j,
                    section=self.sections.active,
                    n_sub=n_sub,
                    model_as=model_as,
                    geomTransf=geomTransf,
                    placement=self.active_placement,
                    end_dist=0.05,
                    metadata=None)
            elif ends['type'] == 'pinned':
                beam = LineElementSequence_Pinned(
                    node_i=start_node,
                    node_j=end_node,
                    ang=self.active_angle,
                    offset_i=offset_i+connection_offset_i,
                    offset_j=offset_j+connection_offset_j,
                    section=self.sections.active,
                    n_sub=n_sub,
                    model_as=model_as,
                    geomTransf=geomTransf,
                    placement=self.active_placement,
                    end_dist=ends['dist'],
                    metadata=None,
                    mat_fix=self.materials.retrieve('fix'),
                    camber=0.00)
            elif ends['type'] == 'fixed-pinned':
                beam = LineElementSequence_FixedPinned(
                    node_i=start_node,
                    node_j=end_node,
                    ang=self.active_angle,
                    offset_i=offset_i+connection_offset_i,
                    offset_j=offset_j+connection_offset_j,
                    section=self.sections.active,
                    n_sub=n_sub,
                    model_as=model_as,
                    geomTransf=geomTransf,
                    placement=self.active_placement,
                    end_dist=ends['dist'],
                    metadata=None,
                    mat_fix=self.materials.retrieve('fix'),
                    mat_release=self.materials.retrieve('release'),
                    camber=0.00)
            elif ends['type'] == 'RBS':
                beam = LineElementSequence_RBS(
                    node_i=start_node,
                    node_j=end_node,
                    ang=self.active_angle,
                    offset_i=offset_i+connection_offset_i,
                    offset_j=offset_j+connection_offset_j,
                    section=self.sections.active,
                    n_sub=n_sub,
                    model_as=model_as,
                    geomTransf=geomTransf,
                    placement=self.active_placement,
                    end_dist=ends['dist'],
                    metadata=None,
                    rbs_length=ends['length'],
                    rbs_reduction=ends['factor'],
                    rbs_n_sub=ends['n_sub'])
            elif ends['type'] == 'RBS_j':
                beam = LineElementSequence_RBS_j(
                    node_i=start_node,
                    node_j=end_node,
                    ang=self.active_angle,
                    offset_i=offset_i+connection_offset_i,
                    offset_j=offset_j+connection_offset_j,
                    section=self.sections.active,
                    n_sub=n_sub,
                    model_as=model_as,
                    geomTransf=geomTransf,
                    placement=self.active_placement,
                    end_dist=ends['dist'],
                    metadata=None,
                    rbs_length=ends['length'],
                    rbs_reduction=ends['factor'],
                    rbs_n_sub=ends['n_sub'])
            elif ends['type'] == 'steel_W_IMK':
                beam = LineElementSequence_IMK(
                    node_i=start_node,
                    node_j=end_node,
                    ang=self.active_angle,
                    offset_i=offset_i+connection_offset_i,
                    offset_j=offset_j+connection_offset_j,
                    section=self.sections.active,
                    n_sub=n_sub,
                    model_as=model_as,
                    geomTransf=geomTransf,
                    placement=self.active_placement,
                    end_dist=ends['dist'],
                    metadata=ends,
                    mat_fix=self.materials.retrieve('fix'))
            elif ends['type'] == 'steel W shear tab':
                beam = LineElementSequence_W_grav_sear_tab(
                    node_i=start_node,
                    node_j=end_node,
                    ang=self.active_angle,
                    offset_i=offset_i+connection_offset_i,
                    offset_j=offset_j+connection_offset_j,
                    section=self.sections.active,
                    n_sub=n_sub,
                    model_as=model_as,
                    geomTransf=geomTransf,
                    placement=self.active_placement,
                    end_dist=ends['dist'],
                    metadata=ends,
                    mat_fix=self.materials.retrieve('fix'))
            else:
                raise ValueError('Invalid end-type')
            ok = level.beams.add(beam)
            if ok:
                beams.append(beam)
                self.groups.add_element(beam)
                start_node.beams.append(beam)
                end_node.beams.append(beam)
        return beams

    def add_brace_at_points(self,
                            start: np.ndarray,
                            end: np.ndarray,
                            btype: str = 'single',
                            model_as: dict = {'type': 'elastic'},
                            geomTransf: str = 'Linear',
                            n_sub: int = 5,
                            release_distance: float = 0.005,
                            camber: float = 0.05):
        """
        Adds a brace connecting the given points
        at all the active levels.
        For single braces, the start node corresponds to the
        level above, and the end node to the level below.
        Existing nodes are used, otherwise they are created.
        Args:
            start (np.ndarray): X,Y coordinates of point i
            end (np.ndarray): X,Y coordinates of point j
            btype (str): Flag for the type of bracing to model
            n_sub (int): Number of internal elements to add
            camber (float): Initial imperfection modeled as
                            parabolic camber, expressed
                            as a proportion of the element's
                            length.
        Returns:
            braces (list[LineElementSequence]): added braces.
        """
        if not self.sections.active:
            raise ValueError("No active section specified")
        if self.active_angle != 0.00:
            raise ValueError("Only ang=0.00 is currently supported")
        braces = []
        if btype != 'single':
            raise ValueError("Only `single` brace type supported")
        for level in self.levels.active:
            # check to see if start node exists
            start_node = level.look_for_node(*start)
            if not start_node:
                # create it if it does not exist
                start_node = Node(
                    np.array([*start, level.elevation]), level.restraint)
                level.nodes_primary.add(start_node)
            # check to see if end node exists
            end_node = level.previous_lvl.look_for_node(*end)
            if not end_node:
                # create it if it does not exist
                end_node = Node(
                    np.array([*end, level.elevation]), level.restraint)
                level.nodes_primary.add(end_node)
            brace = LineElementSequence_Pinned(
                node_i=start_node,
                node_j=end_node,
                ang=self.active_angle,
                offset_i=np.zeros(3),
                offset_j=np.zeros(3),
                section=self.sections.active,
                n_sub=n_sub,
                model_as=model_as,
                geomTransf=geomTransf,
                placement='centroid',
                end_dist=release_distance,
                mat_fix=self.materials.retrieve('fix'),
                mat_release=self.materials.retrieve('release'),
                camber=camber)
            braces.append(brace)
            self.groups.add_element(brace)
            level.braces.add(brace)
        return braces

    def add_columns_from_grids(self, n_sub=1,
                               model_as={'type': 'elastic'},
                               geomTransf='Linear',
                               ends={'type': 'fixed'}):
        """
        Uses the currently defined gridsystem to obtain all locations
        where gridlines intersect, and places a column on
        all such locations.
        Args:
            n_sub (int): Number of internal elements to add.
        Returns:
            columns (list[LineElementSequence]): added columns
        """
        isect_pts = self.gridsystem.intersection_points()
        columns = []
        for pt in isect_pts:
            cols = self.add_column_at_point(
                *pt,
                n_sub=n_sub,
                model_as=model_as,
                geomTransf=geomTransf,
                ends=ends)
            columns.extend(cols)
        return columns

    def add_beams_from_gridlines(self, n_sub=1, ends={'type': 'fixed'},
                                 model_as={'type': 'elastic'}):
        """
        Uses the currently defined gridsystem to obtain all locations
        where gridlines intersect. For each gridline, beams are placed
        connecting all the intersection locations of that
        gridline with all other gridlines.
        Args:
            n_sub (int): Number of internal elements to add
            ends (dict): {'type': 'fixed, 'dist': float}', or
                         {'type': 'pinned', 'dist': float} or
                         {'type': 'RBS', 'dist': float,
                          'length': float, 'factor': float, 'n_sub': int}
                        For the pinned or fixed-pinned case,
                          `dist` represents the
                            proportion of the distance between the element's
                            ends to the release, relative to the element's
                            length, both considered without the offsets.
                        For the RBS case,
                          `dist` is similar to the pinned case,
                            but it corresponds to the start of the RBS portion.
                          `length` represents the length of the RBS portion.
                          `factor` represents the proportion of the RBS
                            section's width, relative to the original section.
                          `n_sub` represents how many LineElements should be
                            used for the RBS portion
        """
        beams = []
        for grid in self.gridsystem.grids:
            bms = self.add_beam_at_points(
                grid.start_np,
                grid.end_np,
                n_sub=n_sub,
                ends=ends,
                model_as=model_as)
            beams.extend(bms)
        return beams

    def add_beams_from_grid_intersections(
        self, n_sub=1, ends={'type': 'fixed'},
            model_as={'type': 'elastic'}):
        """
        Uses the currently defined gridsystem to obtain all locations
        where gridlines intersect. For each gridline, beams are placed
        connecting all the intersection locations of that
        gridline with all other gridlines.
        Args:
            n_sub (int): Number of internal elements to add
            ends (dict): {'type': 'fixed, 'dist': float}', or
                         {'type': 'pinned', 'dist': float} or
                         {'type': 'RBS', 'dist': float,
                          'length': float, 'factor': float}
                        For the pinned case,
                          `dist` represents the
                            proportion of the distance between the element's
                            ends to the release, relative to the element's
                            length, both considered without the offsets.
                        For the RBS case,
                          `dist` is similar to the pinned case,
                            but it corresponds to the start of the RBS portion.
                          `length` represents the length of the RBS portion.
                          `factor` represents the proportion of the RBS
                            section's width, relative to the original section.
                          `n_sub` represents how many LineElements should be
                            used for the RBS portion
        """
        beams = []
        for grid in self.gridsystem.grids:
            isect_pts = self.gridsystem.intersect(grid)
            for i in range(len(isect_pts)-1):
                bms = self.add_beam_at_points(
                    isect_pts[i],
                    isect_pts[i+1],
                    n_sub=n_sub,
                    ends=ends,
                    model_as=model_as)
                beams.extend(bms)
        return beams

    def add_braces_from_grids(self, btype="single", n_sub=5, camber=0.05,
                              model_as={'type': 'elastic'},
                              release_distance=0.005):
        """
        Uses the currently defined gridsystem.
        For each gridline, braces are placed
        connecting the starting point at the top level
        and the end point at the level below.
        Args:
            n_sub (int): Number of internal elements to add
        """
        braces = []
        for grid in self.gridsystem.grids:
            start_pt = grid.start_np
            end_pt = grid.end_np
            brcs = self.add_brace_at_points(
                start_pt,
                end_pt,
                btype=btype,
                model_as=model_as,
                n_sub=n_sub,
                release_distance=release_distance,
                camber=camber)
            braces.extend(brcs)
        return braces

    #############################################
    # Select methods - select objects           #
    #############################################

    def select_all_at_level(self, lvl_name: str):
        """
        Selects all selectable objects at a given level,
        specified by the level's name.
        """
        lvl = self.levels.retrieve_by_name(lvl_name)
        for beam in lvl.beams.element_list:
            self.selection.beams.add(beam)
        for column in lvl.columns.element_list:
            self.selection.columns.add(column)
        for brace in lvl.braces.element_list:
            self.selection.braces.add(brace)

    def select_all(self):
        """
        Selects all selectable objects.
        """
        for lvl in self.levels.level_list:
            self.select_all_at_level(lvl)

    def select_group(self, group_name: str):
        """
        Selects all selectable objects contained
        in a given group, specified by the
        group's name.
        """
        grp = self.groups.retrieve_by_name(group_name)
        for elm in grp.elements:
            if isinstance(elm, LineElementSequence):
                if elm.function == "beam":
                    self.selection.beams.add(elm)
                elif elm.function == "column":
                    self.selection.columns.add(elm)
                elif elm.function == "brace":
                    self.selection.braces.add(elm)

    def select_perimeter_beams_story(self, lvl_name: str):
        lvl = self.levels.retrieve_by_name(lvl_name)
        beams = lvl.beams.element_list
        if not beams:
            return
        line_elements = []
        for beam in beams:
            line_elements.extend(beam.internal_line_elems())
        edges, edge_to_elem_map = \
            trib_area_analysis.list_of_beams_to_mesh_edges_external(
                line_elements)
        halfedges = mesher.define_halfedges(edges)
        halfedge_to_elem_map = {}
        for h in halfedges:
            halfedge_to_elem_map[h.uniq_id] = edge_to_elem_map[h.edge.uniq_id]
        loops = mesher.obtain_closed_loops(halfedges)
        external, _, trivial = mesher.orient_loops(loops)
        # Sanity checks.
        mesher.sanity_checks(external, trivial)
        loop = external[0]
        for h in loop:
            line_element = halfedge_to_elem_map[h.uniq_id]
            self.selection.line_elements.append(line_element)

    def select_perimeter_beams_all(self):
        for lvl in self.levels.level_list:
            self.select_perimeter_beams_story(lvl.name)

    #############################################
    # Remove methods - remove objects           #
    #############################################

    def clear_gridlines_all(self):
        self.gridsystem.clear_all()

    def clear_gridlines(self, tags: list[str]):
        self.gridsystem.clear(tags)

    def delete_selected(self):
        """
        Deletes the selected objects.
        """
        for lvl in self.levels.level_list:
            for node in self.selection.nodes.node_list:
                lvl.nodes_primary.remove(node)
            for beam in self.selection.beams.element_list:
                lvl.beams.remove(beam)
            for column in self.selection.columns.element_list:
                lvl.columns.remove(column)
            for brace in self.selection.braces.element_list:
                lvl.braces.remove(brace)

    #############################################
    # Set active methods - alter active objects #
    #############################################

    def set_active_levels(self, names: list[str]):
        """
        Sets the active levels of the building.
        An empty `names` list is interpreted as
        activating all levels.
        """
        self.levels.set_active(names)

    def set_active_groups(self, names: list[str]):
        """
        Sets the active groups of the building.
        """
        self.groups.set_active(names)

    def set_active_material(self, name: str):
        """
        Sets the active material.
        """
        self.materials.set_active(name)

    def set_active_section(self, name: str):
        """
        Sets the active section.
        """
        self.sections.set_active(name)

    def set_active_placement(self, placement: str):
        """
        Sets the active placement
        """
        self.active_placement = placement

    def set_active_angle(self, ang: float):
        """
        Sets the active angle
        """
        self.active_angle = ang

    def set_global_restraints(self, restraints):
        """
        Sets global restraints
        Args:
            restraints (list[int]): OpenSees-like
                restraint vector (ones and zeros)
        """
        self.global_restraints = restraints

    ############################
    # Methods for adding loads #
    ############################

    def assign_surface_DL(self,
                          load_per_area: float):
        """
        Assigns surface loads on the active levels
        """
        for level in self.levels.active:
            level.assign_surface_DL(load_per_area)

    #########################
    # Preprocessing methods #
    #########################

    def list_of_beams(self):
        list_of_beams = []
        for lvl in self.levels.level_list:
            list_of_beams.extend(lvl.beams.element_list)
        return list_of_beams

    def list_of_columns(self):
        list_of_columns = []
        for lvl in self.levels.level_list:
            list_of_columns.extend(lvl.columns.element_list)
        return list_of_columns

    def list_of_braces(self):
        list_of_braces = []
        for lvl in self.levels.level_list:
            list_of_braces.extend(lvl.braces.element_list)
        return list_of_braces

    def list_of_line_element_sequences(self):
        result = []
        result.extend(self.list_of_beams())
        result.extend(self.list_of_columns())
        result.extend(self.list_of_braces())
        return result

    def list_of_line_elements(self):

        sequences = self.list_of_line_element_sequences()
        result = []
        for sequence in sequences:
            for elm in sequence.internal_elems():
                if isinstance(elm, LineElement):
                    result.append(elm)
        return result

    def list_of_endreleases(self):

        sequences = self.list_of_line_element_sequences()
        result = []
        for sequence in sequences:
            for elm in sequence.internal_elems():
                if isinstance(elm, EndRelease):
                    result.append(elm)
        return result

    def list_of_primary_nodes(self):
        list_of_nodes = []
        for lvl in self.levels.level_list:
            for node in lvl.nodes_primary.node_list:
                list_of_nodes.append(node)
        return list_of_nodes

    def list_of_parent_nodes(self):
        list_of_parent_nodes = []
        for lvl in self.levels.level_list:
            if lvl.parent_node:
                list_of_parent_nodes.append(lvl.parent_node)
        return list_of_parent_nodes

    def list_of_internal_nodes(self):
        list_of_internal_nodes = []
        sequences = self.list_of_line_element_sequences()
        for sequence in sequences:
            list_of_internal_nodes.extend(sequence.internal_nodes())
        return list_of_internal_nodes

    def list_of_all_nodes(self):
        return self.list_of_primary_nodes() + \
            self.list_of_internal_nodes() + \
            self.list_of_parent_nodes()

    def list_of_steel_W_panel_zones(self):
        cols = self.list_of_columns()
        pzs = []
        for col in cols:
            if isinstance(col, LineElementSequence_Steel_W_PanelZone):
                pzs.append(col.end_segment_i)
            if isinstance(col, LineElementSequence_Steel_W_PanelZone_IMK):
                pzs.append(col.end_segment_i)
        return pzs

    def retrieve_beam(self, uniq_id: int) -> LineElementSequence:
        beams = self.list_of_beams()
        result = None
        for beam in beams:
            if beam.uniq_id == uniq_id:
                result = beam
                break
        return result

    def retrieve_column(self, uniq_id: int) -> LineElementSequence:
        columns = self.list_of_columns()
        result = None
        for col in columns:
            if col.uniq_id == uniq_id:
                result = col
                break
        return result

    def reference_length(self):
        """
        Returns the largest dimension of the
        bounding box of the building
        (used in graphics)
        """
        p_min = np.full(3, np.inf)
        p_max = np.full(3, -np.inf)
        for node in self.list_of_primary_nodes():
            p = np.array(node.coords)
            p_min = np.minimum(p_min, p)
            p_max = np.maximum(p_max, p)
        ref_len = np.max(p_max - p_min)
        return ref_len

    def preprocess(self, assume_floor_slabs=True, self_weight=True,
                   steel_panel_zones=False,
                   elevate_column_splices=0.00):
        """
        Preprocess the building. No further editing beyond this point.
        This method initiates automated calculations to
        get things ready for running an analysis.
        """
        def apply_floor_load(lvl):
            """
            Given a building level, distribute
            the surface load of the level on the beams
            of that level.
            """
            if lvl.floor_coordinates is not None:
                for beam in lvl.beams.element_list:
                    for line_elm in beam.internal_line_elems():
                        udlZ_val = - line_elm.tributary_area * \
                            lvl.surface_DL / line_elm.length_clear()
                        line_elm.add_udl_glob(
                            np.array([0.00, 0.00, udlZ_val]),
                            ltype='floor')
                for node in lvl.list_of_all_nodes():
                    pZ_val = - node.tributary_area * \
                        lvl.surface_DL
                    node.load_fl += np.array((0.00, 0.00, -pZ_val,
                                              0.00, 0.00, 0.00))
        # ~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # steel connection panel zones #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        if steel_panel_zones:
            self._preprocess_steel_panel_zones()

        # ~~~~~~~~~~~~~~~~~~~~~~~ #
        # column splice elevating #
        # ~~~~~~~~~~~~~~~~~~~~~~~ #
        
        if elevate_column_splices != 0.00:
            self._elevate_steel_column_splices(elevate_column_splices)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # tributary areas, weight and mass #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        for lvl in self.levels.level_list:
            if lvl.parent_node:
                # remove parent nodes
                lvl.parent_node = None
                # floor-associated level parameters
                lvl.floor_bisector_lines = None
                lvl.floor_coordinates = None
                # zero-out floor load/mass contribution
                for node in lvl.list_of_primary_nodes():
                    node.load_fl = np.zeros(6)
                for elm in lvl.list_of_internal_elems():
                    for ielm in elm.internal_elems:
                        ielm.udl_fl = np.zeros(3)

        for lvl in self.levels.level_list:
            if assume_floor_slabs:
                beams = []
                for seq in lvl.beams.element_list:
                    beams.extend(seq.internal_line_elems())
                if beams:
                    coords, bisectors = \
                        trib_area_analysis.calculate_tributary_areas(
                            beams)
                    lvl.diaphragm = True
                    lvl.floor_coordinates = coords
                    lvl.floor_bisector_lines = bisectors
                    # distribute floor loads on beams and nodes
                    apply_floor_load(lvl)

        # frame element self-weight
        if self_weight:
            for elm in self.list_of_line_element_sequences():
                elm.apply_self_weight_and_mass(1.00)
        if assume_floor_slabs:
            for lvl in self.levels.level_list:
                # accumulate all the mass at the parent nodes
                if lvl.diaphragm:
                    properties = mesher.geometric_properties(
                        lvl.floor_coordinates)
                    floor_mass = -lvl.surface_DL * \
                        properties['area'] / common.G_CONST
                    assert(floor_mass >= 0.00),\
                        "Error: floor area properties\n" + \
                        "Overall floor area should be negative" + \
                        " (by convention)."
                    floor_centroid = properties['centroid']
                    floor_mass_inertia = properties['inertia']['ir_mass']\
                        * floor_mass
                    lvl.parent_node = Node(
                        np.array([floor_centroid[0], floor_centroid[1],
                                  lvl.elevation]), "parent")
                    lvl.parent_node.mass = np.array([floor_mass,
                                                     floor_mass,
                                                     0.,
                                                     0., 0.,
                                                     floor_mass_inertia])

    def _preprocess_steel_panel_zones(self):
        """
        TODO docstring
        """
        for lvl in self.levels.level_list:
            for i_col, col in enumerate(lvl.columns.element_list):
                node = col.node_i
                # get a list of all the connected beams
                beams = node.beams
                # determine the ones that are properly connected
                # for panel zone modeling
                panel_beam_front = None
                panel_beam_back = None
                panel_beam_front_side = None
                panel_beam_back_side = None
                for bm in beams:
                    # beam must be Fixed or RBS
                    if not isinstance(bm, (LineElementSequence_Fixed,
                                           LineElementSequence_RBS,
                                           LineElementSequence_RBS_j,
                                           LineElementSequence_IMK)):
                        continue
                    # it must have a top_center placement
                    if bm.placement != 'top_center':
                        continue
                    # the frame must lie on a plane
                    if not np.abs(np.dot(bm.x_axis, col.y_axis)) >= \
                       1.00 - common.EPSILON:
                        continue
                    # the column must be vertical
                    if not col.x_axis[2] == -1:
                        continue
                    # if all conditions are met
                    # a panel zone will be modeled based on
                    # the properties of that beam
                    # determine if it's front or back relative to
                    # the y axis of the column

                    if bm.node_i == node:
                        bm_other_node = bm.node_j
                        this_side = 'i'
                    else:
                        bm_other_node = bm.node_i
                        this_side = 'j'
                    bm_vec = (bm_other_node.coords - node.coords)
                    if np.dot(bm_vec, col.y_axis) > 0.00:
                        panel_beam_front = bm
                        panel_beam_front_side = this_side
                    else:
                        panel_beam_back = bm
                        panel_beam_back_side = this_side
                # check that the beams are connected at the face of the column
                if panel_beam_front:
                    if panel_beam_front_side == 'i':
                        assert np.abs(np.linalg.norm(
                            2. * panel_beam_back.offset_i[0:2]) -
                            col.section.properties['d']) < common.EPSILON, \
                            'Incorrect connectivity'
                    else:
                        assert np.abs(np.linalg.norm(
                            2. * panel_beam_front.offset_j[0:2]) -
                            col.section.properties['d']) < common.EPSILON, \
                            'Incorrect connectivity'
                if panel_beam_back:
                    if panel_beam_back_side == 'i':
                        assert np.abs(np.linalg.norm(
                            2. * panel_beam_back.offset_i[0:2]) -
                            col.section.properties['d']) < common.EPSILON, \
                            'Incorrect connectivity'
                    else:
                        assert np.abs(np.linalg.norm(
                            2. * panel_beam_back.offset_j[0:2]) -
                            col.section.properties['d']) < common.EPSILON, \
                            'Incorrect connectivity'

                if panel_beam_front:
                    beam_depth = panel_beam_front.section.properties['d']
                elif panel_beam_back:
                    beam_depth = panel_beam_back.section.properties['d']
                else:
                    continue

                # Or maybe it's time to do some more programming?
                if panel_beam_front and panel_beam_back:
                    assert panel_beam_front.section.properties['d'] == \
                        panel_beam_back.section.properties['d'], \
                        "Incompatible beam depths. Should be equal."

                # replace column with a LineElementSequence_Steel_W_PanelZone
                # ... define the new element
                # note: unfortunately here we can't just make changes
                # to the column and have those changes be applied
                # wherever this column is referenced, because we
                # need to replace it with an object that has a different
                # type. We therefore need to manually assign every variable
                # that was pointing to it to the new object.
                # (I wish python had pointers, or alternatively taht I knew
                #  more python. I'm a structural engineer.)
                if col.metadata['ends']['type'] == 'fixed':
                    col = LineElementSequence_Steel_W_PanelZone(
                        col.node_i, col.node_j, col.ang,
                        col.offset_i, col.offset_j, col.section,
                        col.n_sub, col.model_as, col.geomTransf,
                        col.placement, col.end_dist,
                        col.metadata,
                        self.sections.retrieve('rigid'),
                        self.materials.retrieve('fix'),
                        beam_depth, col.section.material.parameters['b_PZ'])
                elif col.metadata['ends']['type'] == 'steel_W_PZ':
                    col = LineElementSequence_Steel_W_PanelZone(
                        col.node_i, col.node_j, col.ang,
                        col.offset_i, col.offset_j, col.section,
                        col.n_sub, col.model_as, col.geomTransf,
                        col.placement, col.end_dist,
                        col.metadata,
                        self.sections.retrieve('rigid'),
                        self.materials.retrieve('fix'),
                        beam_depth, col.section.material.parameters['b_PZ'])
                elif col.metadata['ends']['type'] == 'steel_W_PZ_IMK':
                    col = LineElementSequence_Steel_W_PanelZone_IMK(
                        col.node_i, col.node_j, col.ang,
                        col.offset_i, col.offset_j, col.section,
                        col.n_sub, col.model_as, col.geomTransf,
                        col.placement, col.end_dist,
                        col.metadata,
                        self.sections.retrieve('rigid'),
                        self.materials.retrieve('fix'),
                        beam_depth, col.section.material.parameters['b_PZ'])
                else:
                    raise ValueError('Invalid end type for W column')
                # ... replace it in the node's `column_below` attribute
                node.column_below = col
                # ... replace it in the underlying node's `column_above`
                nj = col.node_j
                nj.column_above = col
                # ... replace it in the leven's column container object
                lvl.columns.element_list[i_col] = col

                # modify beam connectivity
                panel_zone_segment = col.end_segment_i
                if panel_beam_front:
                    if panel_beam_front_side == 'i':
                        sgm = panel_beam_front.end_segment_i
                        sgm.offset = np.zeros(3).copy()
                        sgm.internal_elems[0].node_i = \
                            panel_zone_segment.n_front
                        sgm.internal_elems[0].offset_i = \
                            np.zeros(3).copy()
                        sgm.n_external = \
                            panel_zone_segment.n_front
                        panel_beam_front.node_i = panel_zone_segment.n_front
                        panel_beam_front.offset_i = np.zeros(3).copy()
                    elif panel_beam_front_side == 'j':
                        sgm = panel_beam_front.end_segment_j
                        sgm.offset = np.zeros(3).copy()
                        sgm.internal_elems[-1].node_j = \
                            panel_zone_segment.n_front
                        sgm.internal_elems[-1].offset_j = \
                            np.zeros(3).copy()
                        sgm.n_external = \
                            panel_zone_segment.n_front
                        panel_beam_front.node_j = panel_zone_segment.n_front
                        panel_beam_front.offset_j = np.zeros(3).copy()
                    else:
                        raise ValueError('This should never happen!')

                if panel_beam_back:
                    if panel_beam_back_side == 'i':
                        sgm = panel_beam_back.end_segment_i
                        sgm.offset = np.zeros(3).copy()
                        sgm.internal_elems[0].node_i = \
                            panel_zone_segment.n_back
                        sgm.internal_elems[0].offset_i = \
                            np.zeros(3).copy()
                        sgm.n_external = \
                            panel_zone_segment.n_back
                        panel_beam_back.node_i = panel_zone_segment.n_back
                        panel_beam_back.offset_i = np.zeros(3).copy()
                    elif panel_beam_back_side == 'j':
                        sgm = panel_beam_back.end_segment_j
                        sgm.offset = np.zeros(3).copy()
                        sgm.internal_elems[-1].node_j = \
                            panel_zone_segment.n_back
                        sgm.internal_elems[-1].offset_j = \
                            np.zeros(3).copy()
                        sgm.n_external = \
                            panel_zone_segment.n_back
                        panel_beam_back.node_j = panel_zone_segment.n_back
                        panel_beam_back.offset_j = np.zeros(3).copy()
                    else:
                        raise ValueError('This should never happen!')

    def _elevate_steel_column_splices(self, relative_len):
        """
        TODO - add docstring
        """
        for col in self.list_of_columns():
            # check to see if there is a column at the level below
            n_j = col.node_j
            if n_j.column_below:
                sec = n_j.column_below.section
                if sec is not col.section:
                    col.end_segment_j.section = sec
                    for elm in col.end_segment_j.internal_elems:
                        elm.section = sec
                    z_test = col.node_j.coords[2] + \
                        col.length_clear * relative_len
                    for elm in col.middle_segment.internal_elems:
                        if elm.node_i.coords[2] < z_test:
                            elm.section = sec
        
    def level_masses(self):
        lvls = self.levels.level_list
        n_lvls = len(lvls)
        level_masses = np.full(n_lvls, 0.00)
        for i, lvl in enumerate(lvls):
            total_mass = 0.00
            for node in lvl.list_of_all_nodes():
                if node.restraint_type == "free":
                    total_mass += node.mass[0]
            if lvl.parent_node:
                total_mass += lvl.parent_node.mass[0]
            level_masses[i] = total_mass
        return level_masses

    ###############################
    # Preprocessing Visualization #
    ###############################

    def plot_building_geometry(self,
                               extrude_frames=False,
                               offsets=True,
                               gridlines=True,
                               global_axes=True,
                               diaphragm_lines=True,
                               tributary_areas=True,
                               just_selection=False,
                               parent_nodes=True,
                               frame_axes=True):
        preprocessing_3D.plot_building_geometry(
            self,
            extrude_frames=extrude_frames,
            offsets=offsets,
            gridlines=gridlines,
            global_axes=global_axes,
            diaphragm_lines=diaphragm_lines,
            tributary_areas=tributary_areas,
            just_selection=just_selection,
            parent_nodes=parent_nodes,
            frame_axes=frame_axes)

    def plot_2D_level_geometry(self,
                               lvlname: str,
                               extrude_frames=False):
        preprocessing_2D.plot_2D_level_geometry(
            self,
            lvlname,
            extrude_frames=extrude_frames)
