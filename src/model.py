"""
Model Builder for OpenSeesPy ~ Model module
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
import json
import numpy as np
from grids import GridLine, GridSystem
from node import Node
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
from material import Materials
from section import Sections
from level import Level, Levels
from group import Group, Groups
from selection import Selection
from utility import common
from utility import trib_area_analysis
from utility import mesher
from utility.graphics import preprocessing_3D
from utility.graphics import preprocessing_2D


# pylint: disable=unsubscriptable-object
# pylint: disable=invalid-name


@dataclass
class Model:
    """
    This class manages building objects.
    Attributes:
        gridsystem (GridSystem): Gridsystem used to
                   define or modify elements.
        levels (Levels): Levels of the building
        groups (Groups): Groups of the building
        sections (Sections): Sections used
        materials (Materials): Materials used
        line_connectivity (dict[tuple, int]): How primary nodes
            are connected with LineElementSequences. Used to prevent
            defining many elements that connect the same two nodes.
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
    line_connectivity: dict[tuple, int] = field(default_factory=dict)
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
        for levelname in self.levels.active:
            level = self.levels.registry[levelname]
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
            column (LineElementSequence): Added column.
        """
        if not self.sections.active:
            raise ValueError("No active section")
        for levelname in self.levels.active:
            level = self.levels.registry[levelname]
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
                # check if column connecting the two nodes already exists
                node_ids = [top_node.uniq_id, bot_node.uniq_id]
                node_ids.sort()
                node_ids = tuple(node_ids)
                if node_ids in self.line_connectivity:
                    raise ValueError('Element already exists')
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
                        mat_fix=self.materials.registry['fix'],
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
                        mat_fix=self.materials.registry['fix'],
                        mat_release=self.materials.registry['release'],
                        camber=0.00)
                else:
                    raise ValueError('Invalid end-type')
                level.columns.add(column)
                self.line_connectivity[node_ids] = column.uniq_id
                top_node.column_below = column
                bot_node.column_above = column
        return column

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
        for levelname in self.levels.active:
            level = self.levels.registry[levelname]

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
            # check for an existing LineElementSequence connecting
            # the start and end nodes.
            node_ids = [start_node.uniq_id, end_node.uniq_id]
            node_ids.sort()
            node_ids = tuple(node_ids)
            if node_ids in self.line_connectivity:
                raise ValueError('Element already exists')

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
                    mat_fix=self.materials.registry['fix'],
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
                    mat_fix=self.materials.registry['fix'],
                    mat_release=self.materials.registry['release'],
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
                    mat_fix=self.materials.registry['fix'])
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
                    mat_fix=self.materials.registry['fix'])
            else:
                raise ValueError('Invalid end-type')
            level.beams.add(beam)
            beams.append(beam)
            self.line_connectivity[node_ids] = beam.uniq_id
            start_node.beams.append(beam)
            end_node.beams.append(beam)
        return beams

    #############################################
    # Select methods - select objects           #
    #############################################

    def select_all_at_level(self, lvl_name: str):
        """
        Selects all selectable objects at a given level,
        specified by the level's name.
        """
        lvl = self.levels.registry[lvl_name]
        for beam in lvl.beams.registry.values():
            self.selection.beams.add(beam)
        for column in lvl.columns.registry.values():
            self.selection.columns.add(column)
        for brace in lvl.braces.registry.values():
            self.selection.braces.add(brace)

    def select_all(self):
        """
        Selects all selectable objects.
        """
        for lvl in self.levels.registry.values():
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
        lvl = self.levels.registry[lvl_name]
        beams = lvl.beams.registry.values()
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
        for lvl in self.levels.registry.values():
            self.select_perimeter_beams_story(lvl.name)

    #############################################
    # Remove methods - remove objects           #
    #############################################

    def clear_gridlines_all(self):
        self.gridsystem.clear_all()

    def clear_gridlines(self, tags: list[str]):
        self.gridsystem.clear(tags)

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
        for levelname in self.levels.active:
            level = self.levels.registry[levelname]
            level.assign_surface_DL(load_per_area)

    #########################
    # Preprocessing methods #
    #########################

    def list_of_beams(self):
        list_of_beams = []
        for lvl in self.levels.registry.values():
            list_of_beams.extend(lvl.beams.registry.values())
        return list_of_beams

    def list_of_columns(self):
        list_of_columns = []
        for lvl in self.levels.registry.values():
            list_of_columns.extend(lvl.columns.registry.values())
        return list_of_columns

    def list_of_braces(self):
        list_of_braces = []
        for lvl in self.levels.registry.values():
            list_of_braces.extend(lvl.braces.registry.values())
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
        for lvl in self.levels.registry.values():
            for node in lvl.nodes_primary.registry.values():
                list_of_nodes.append(node)
        return list_of_nodes

    def list_of_parent_nodes(self):
        list_of_parent_nodes = []
        for lvl in self.levels.registry.values():
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
                for beam in lvl.beams.registry.values():
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

        for lvl in self.levels.registry.values():
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

        for lvl in self.levels.registry.values():
            if assume_floor_slabs:
                beams = []
                for seq in lvl.beams.registry.values():
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
            for lvl in self.levels.registry.values():
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
        for lvl in self.levels.registry.values():
            columns = list(lvl.columns.registry.values())
            for col in columns:
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
                previous_id = col.uniq_id
                if col.metadata['ends']['type'] == 'fixed':
                    col = LineElementSequence_Steel_W_PanelZone(
                        col.node_i, col.node_j, col.ang,
                        col.offset_i, col.offset_j, col.section,
                        col.n_sub, col.model_as, col.geomTransf,
                        col.placement, col.end_dist,
                        col.metadata,
                        self.sections.registry['rigid'],
                        self.materials.registry['fix'],
                        beam_depth, col.section.material.parameters['b_PZ'])
                elif col.metadata['ends']['type'] == 'steel_W_PZ':
                    col = LineElementSequence_Steel_W_PanelZone(
                        col.node_i, col.node_j, col.ang,
                        col.offset_i, col.offset_j, col.section,
                        col.n_sub, col.model_as, col.geomTransf,
                        col.placement, col.end_dist,
                        col.metadata,
                        self.sections.registry['rigid'],
                        self.materials.registry['fix'],
                        beam_depth, col.section.material.parameters['b_PZ'])
                elif col.metadata['ends']['type'] == 'steel_W_PZ_IMK':
                    col = LineElementSequence_Steel_W_PanelZone_IMK(
                        col.node_i, col.node_j, col.ang,
                        col.offset_i, col.offset_j, col.section,
                        col.n_sub, col.model_as, col.geomTransf,
                        col.placement, col.end_dist,
                        col.metadata,
                        self.sections.registry['rigid'],
                        self.materials.registry['fix'],
                        beam_depth, col.section.material.parameters['b_PZ'])
                else:
                    raise ValueError('Invalid end type for W column')
                # ... replace it in the node's `column_below` attribute
                node.column_below = col
                # ... replace it in the underlying node's `column_above`
                nj = col.node_j
                nj.column_above = col
                # ... replace it in the level's column container object
                lvl.columns.remove(previous_id)
                lvl.columns.add(col)

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
        lvls = self.levels.registry.values()
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
