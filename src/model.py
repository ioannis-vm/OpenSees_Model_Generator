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

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from typing import OrderedDict
import json
import numpy as np
import node
import components
from components import MiddleSegment
from components import EndSegment_Fixed
from components import EndSegment_Steel_W_PanelZone
from components import EndSegment_Steel_W_PanelZone_IMK
from components import ComponentAssembly
from components import Materials
import section
from section import Sections
from level import Level, Levels
from selection import Selection
import common
import trib_area_analysis
import mesher
from graphics import preprocessing_3D
from itertools import count


# pylint: disable=unsubscriptable-object
# pylint: disable=invalid-name


@dataclass
class Model:
    """
    todo - in progress
    """
    levels: Levels = field(default_factory=Levels)
    sections: Sections = field(default_factory=Sections)
    materials: Materials = field(default_factory=Materials)
    selection: Selection = field(default_factory=Selection)
    line_connectivity: dict[tuple, int] = field(default_factory=dict)
    active_placement: str = field(default='centroid')
    active_angle: float = field(default=0.00)
    global_restraints: list = field(default_factory=list)

    def __post_init__(self):
        """
        todo - in progress
        """
        node.node_ids = count(0)
        section.section_ids = count(0)
        components.elem_ids = count(0)
        components.line_elem_seq_ids = count(0)
        self.update_required = False
        self.dct_beams = []
        self.dct_columns = []
        self.dct_braces = []
        self.dct_line_element_sequences = []
        self.dct_line_elements = []
        self.dct_end_releases = []
        self.dct_primary_nodes = []
        self.dct_parent_nodes = []
        self.dct_internal_nodes = []
        self.dct_all_nodes = []

    ###############################################
    # 'Add' methods - add objects to the building #
    ###############################################

    def add_node(self,
                 x: float,
                 y: float) -> list[node.Node]:
        """
        todo - in progress
        """
        added_nodes = []
        for levelname in self.levels.active:
            level = self.levels.registry[levelname]
            nd = node.Node(
                np.array([x, y, level.elevation]),
                level.restraint)
            level.nodes_primary.add(nd)
            added_nodes.append(nd)
            self.update_required = True
        return added_nodes

    def add_level(self,
                  name: str,
                  elevation: float,
                  restraint: str = "free"
                  ) -> Level:
        """
        todo - in progress
        """
        level = Level(name, elevation, restraint)
        self.levels.add(level)
        return level

    def add_sections_from_json(self,
                               filename: str,
                               sec_type: str,
                               labels: list[str]):
        """
        todo - in progress
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


    def add_column_at_point(self,
                            pt: np.ndarray,
                            n_sub=1,
                            model_as={'type': 'elastic'},
                            geom_transf='Linear', ends={'type': 'fixed'}) \
            -> ComponentAssembly:
        """
        todo - in progress
        """
        [x, y] = pt
        if not self.sections.active:
            raise ValueError("No active section")
        for levelname in self.levels.active:
            level = self.levels.registry[levelname]
            if level.previous_lvl:  # if previous level exists
                # check to see if top node exists
                top_node = level.look_for_node(x, y)
                # create it if it does not exist
                if not top_node:
                    top_node = node.Node(
                        np.array([x, y, level.elevation]), level.restraint)
                    level.nodes_primary.add(top_node)
                # check to see if bottom node exists
                bot_node = level.previous_lvl.look_for_node(
                    x, y)
                # create it if it does not exist
                if not bot_node:
                    bot_node = node.Node(
                        np.array([x, y, level.previous_lvl.elevation]),
                        level.previous_lvl.restraint)
                    level.previous_lvl.nodes_primary.add(bot_node)
                # check if column connecting the two nodes already exists
                node_ids_lst = [top_node.uid, bot_node.uid]
                node_ids_lst.sort()
                node_ids = (*node_ids_lst,)
                if node_ids in self.line_connectivity:
                    raise ValueError('Element already exists')
                # add the column connecting the two nodes
                column = ComponentAssembly(
                    node_i=top_node,
                    node_j=bot_node,
                    ang=self.active_angle,
                    offset_i=np.zeros(3),
                    offset_j=np.zeros(3),
                    n_sub=n_sub,
                    model_as=model_as,
                    geom_transf=geom_transf,
                    placement=self.active_placement,
                    ends=ends,
                    materials=self.materials,
                    sections=self.sections)
                level.columns.add(column)
                self.line_connectivity[node_ids] = column.uid
                top_node.column_below = column
                bot_node.column_above = column
            self.dct_update_required = True
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
                           geom_transf='Linear'):
        """
        todo - in progress
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
                    start_node = node.Node(
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
                    end_node = node.Node(
                        np.array([*end, level.elevation]), level.restraint)
                    level.nodes_primary.add(end_node)
                    connection_offset_j = np.zeros(3)
            # check for an existing ComponentAssembly connecting
            # the start and end nodes.
            node_ids_lst = [start_node.uid, end_node.uid]
            node_ids_lst.sort()
            node_ids = (*node_ids_lst,)
            if node_ids in self.line_connectivity:
                raise ValueError('Element already exists')

            # ---------------- #
            # element creation #
            # ---------------- #
            # add the beam connecting the two nodes
            beam = ComponentAssembly(
                node_i=start_node,
                node_j=end_node,
                ang=self.active_angle,
                offset_i=offset_i+connection_offset_i,
                offset_j=offset_j+connection_offset_j,
                n_sub=n_sub,
                model_as=model_as,
                geom_transf=geom_transf,
                placement=self.active_placement,
                ends=ends,
                materials=self.materials,
                sections=self.sections)
            level.beams.add(beam)
            beams.append(beam)
            self.line_connectivity[node_ids] = beam.uid
            start_node.beams.append(beam)
            end_node.beams.append(beam)
            self.dct_update_required = True
        return beams

    #############################################
    # Select methods - select objects           #
    #############################################

    def select_beam(self, uid):
        """
        todo - in progress
        """
        if uid in self.dct_beams:
            self.selection.beams.add(self.dct_beams[uid])

    def select_column(self, uid):
        """
        todo - in progress
        """
        if uid in self.dct_columns:
            self.selection.columns.add(self.dct_columns[uid])

    def select_all_at_level(self, lvl_name: str):
        """
        todo - in progress
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
        todo - in progress
        """
        for lvl in self.levels.registry.values():
            self.select_all_at_level(lvl)

    def select_perimeter_beams_story(self, lvl_name: str):
        """
        todo - in progress
        """
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
            halfedge_to_elem_map[h.uid] = edge_to_elem_map[h.edge.uid]
        loops = mesher.obtain_closed_loops(halfedges)
        external, _, trivial = mesher.orient_loops(loops)
        # Sanity checks.
        mesher.sanity_checks(external, trivial)
        loop = external[0]
        for h in loop:
            line_element = halfedge_to_elem_map[h.uid]
            self.selection.line_elements.append(line_element)

    def select_perimeter_beams_all(self):
        """
        todo - in progress
        """
        for lvl in self.levels.registry.values():
            self.select_perimeter_beams_story(lvl.name)

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

    def assign_surface_load(self,
                          load_per_area: float):
        """
        Assigns surface loads on the active levels
        """
        for levelname in self.levels.active:
            level = self.levels.registry[levelname]
            level.assign_surface_load(load_per_area)

    def assign_surface_load_massless(
            self,
            load_per_area: float):
        """
        Assigns surface loads on the active levels
        """
        for levelname in self.levels.active:
            level = self.levels.registry[levelname]
            level.assign_surface_load_massless(load_per_area)

    #########################
    # Preprocessing methods #
    #########################

    def _update_lists(self):
        """
        todo - in progress
        """
        self.dct_beams = {}
        self.dct_columns = {}
        self.dct_braces = {}
        self.dct_primary_nodes = {}
        self.dct_parent_nodes = {}
        self.dct_internal_nodes = {}
        for lvl in self.levels.registry.values():
            self.dct_beams.update(lvl.beams.registry)
            self.dct_columns.update(lvl.columns.registry)
            self.dct_braces.update(lvl.braces.registry)
            for nd in lvl.nodes_primary.registry.values():
                self.dct_primary_nodes[nd.uid] = nd
            if lvl.parent_node:
                self.dct_parent_nodes[lvl.parent_node.uid] = lvl.parent_node
        self.dct_line_element_sequences = {}
        self.dct_line_element_sequences.update(self.dct_beams)
        self.dct_line_element_sequences.update(self.dct_columns)
        self.dct_line_element_sequences.update(self.dct_braces)
        self.dct_line_elements = {}
        self.dct_end_releases = {}
        for sequence in self.dct_line_element_sequences.values():
            for elm in sequence.internal_line_elems():
                self.dct_line_elements[elm.uid] = elm
            for elm in sequence.internal_end_releases():
                self.dct_end_releases[elm.uid] = elm
            internal_nodes = sequence.internal_nodes()
            for nd in internal_nodes:
                if nd.uid not in self.dct_internal_nodes:
                    self.dct_internal_nodes[nd.uid] = nd
        self.dct_all_nodes = {}
        if self.dct_primary_nodes:
            self.dct_all_nodes.update(self.dct_primary_nodes)
        if self.dct_internal_nodes:
            self.dct_all_nodes.update(self.dct_internal_nodes)
        if self.dct_parent_nodes:
            self.dct_all_nodes.update(self.dct_parent_nodes)
        self.dct_update_required = False

    def list_of_beams(self):
        if self.dct_update_required:
            self._update_lists()
        return list(self.dct_beams.values())

    def list_of_columns(self):
        if self.dct_update_required:
            self._update_lists()
        return list(self.dct_columns.values())

    def list_of_braces(self):
        if self.dct_update_required:
            self._update_lists()
        return list(self.dct_braces.values())

    def list_of_line_element_sequences(self):
        if self.dct_update_required:
            self._update_lists()
        return list(self.dct_line_element_sequences.values())

    def list_of_line_elements(self):
        if self.dct_update_required:
            self._update_lists()
        return list(self.dct_line_elements.values())

    def list_of_endreleases(self):
        if self.dct_update_required:
            self._update_lists()
        return list(self.dct_end_releases.values())

    def list_of_primary_nodes(self):
        if self.dct_update_required:
            self._update_lists()
        return list(self.dct_primary_nodes.values())

    def list_of_parent_nodes(self):
        if self.dct_update_required:
            self._update_lists()
        return list(self.dct_parent_nodes.values())

    def list_of_internal_nodes(self):
        if self.dct_update_required:
            self._update_lists()
        return list(self.dct_internal_nodes.values())

    def list_of_all_nodes(self):
        if self.dct_update_required:
            self._update_lists()
        return list(self.dct_all_nodes.values())

    def list_of_steel_W_panel_zones(self):
        cols = self.list_of_columns()
        pzs = []
        for col in cols:
            end_sgmt = col.end_segment_i
            if isinstance(end_sgmt, EndSegment_Steel_W_PanelZone):
                pzs.append(col.end_segment_i)
            if isinstance(end_sgmt, EndSegment_Steel_W_PanelZone_IMK):
                pzs.append(col.end_segment_i)
        return pzs

    def retrieve_beam(self, uid: int) -> Optional[ComponentAssembly]:
        """
        todo - in progress
        """
        result = self.dct_beams[uid]
        return result

    def retrieve_column(self, uid: int) -> Optional[ComponentAssembly]:
        """
        todo - in progress
        """
        result = self.dct_columns[uid]
        return result

    def retrieve_node(self, uid: int) -> Optional[ComponentAssembly]:
        """
        todo - in progress
        """
        result = self.dct_all_nodes[uid]
        return result

    def reference_length(self):
        """
        Returns the largest dimension of the
        bounding box of the building
        (used in graphics)
        """
        p_min = np.full(3, np.inf)
        p_max = np.full(3, -np.inf)
        for nd in self.list_of_primary_nodes():
            p = np.array(nd.coords)
            p_min = np.minimum(p_min, p)
            p_max = np.maximum(p_max, p)
        ref_len = np.max(p_max - p_min)
        return ref_len

    def level_masses(self):
        """
        todo - in progress
        """
        lvls = self.levels.registry.values()
        n_lvls = len(lvls)
        level_masses = np.full(n_lvls, 0.00)
        for i, lvl in enumerate(lvls):
            total_mass = 0.00
            for nd in lvl.list_of_all_nodes():
                if nd.restraint_type == "free":
                    total_mass += nd.mass[0]
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
                               frame_axes=True,
                               camera=None):
        """
        todo - in progress
        """
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
            frame_axes=frame_axes,
            camera=camera)
