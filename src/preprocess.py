"""
Model Generator for OpenSees ~ preprocess
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
import numpy as np
import node
from components import MiddleSegment
from components import EndSegment_Fixed
from components import EndSegment_Steel_W_PanelZone
from components import EndSegment_Steel_W_PanelZone_IMK
from utility import common
from utility import trib_area_analysis
from utility import mesher


def diaphragms(mdl):
    """
    Applies rigid diaphragm constraints to all levels that contain
    beams.
    """
    for lvl in mdl.levels.registry.values():
        beams = []
        for seq in lvl.beams.registry.values():
            beams.extend(seq.internal_line_elems())
        if beams:
            lvl.diaphragm = True


def tributary_area_analysis(mdl):
    """
    Performs tributary area analysis to distribute floor loads to the
    supporting elements.
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
                        lvl.surface_load / line_elm.length_clear
                    line_elm.add_udl_glob(
                        np.array([0.00, 0.00, udlZ_val]),
                        ltype='floor')
                    udlZ_val_massless = - line_elm.tributary_area * \
                        lvl.surface_load_massless / line_elm.length_clear
                    line_elm.add_udl_glob(
                        np.array([0.00, 0.00, udlZ_val_massless]),
                        ltype='floor_massless'
                    )
            for nd in lvl.list_of_all_nodes():
                pZ_val = - nd.tributary_area * \
                    (lvl.surface_load + lvl.surface_load_massless)
                nd.load_fl += np.array((0.00, 0.00, -pZ_val,
                                        0.00, 0.00, 0.00))
    # ~~~

    mdl.update_required = True

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # tributary areas, weight and mass #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    for lvl in mdl.levels.registry.values():
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


def self_weight_and_mass(mdl):
    """
    TODO ~ docstring
    """
    # # frame element self-weight
    # if self_weight:
    #     for elm in mdl.list_of_line_element_sequences():
    #         elm.apply_self_weight_and_mass()
    # if assume_floor_slabs:
    #     for lvl in mdl.levels.registry.values():
    #         # accumulate all the mass at the parent nodes
    #         if lvl.diaphragm:
    #             properties = mesher.geometric_properties(
    #                 lvl.floor_coordinates)
    #             floor_mass = -lvl.surface_load * \
    #                 properties['area'] / common.G_CONST
    #             assert(floor_mass >= 0.00),\
    #                 "Error: floor area properties\n" + \
    #                 "Overall floor area should be negative" + \
    #                 " (by convention)."
    #             floor_centroid = properties['centroid']
    #             floor_mass_inertia = properties['inertia']['ir_mass']\
    #                 * floor_mass
    #             lvl.parent_node = node.Node(
    #                 np.array([floor_centroid[0], floor_centroid[1],
    #                           lvl.elevation]), "parent")
    #             lvl.parent_node.mass = np.array([floor_mass,
    #                                              floor_mass,
    #                                              0.,
    #                                              0., 0.,
    #                                              floor_mass_inertia])
    # frame element self-weight
    for elm in mdl.list_of_line_element_sequences():
        elm.apply_self_weight_and_mass()
    for lvl in mdl.levels.registry.values():
        if not lvl.diaphragm:
            continue
        if lvl.restraint != "free":
            continue
        # accumulate all the mass at the parent nodes
        properties = mesher.geometric_properties(lvl.floor_coordinates)
        floor_mass = -lvl.surface_load * \
            properties['area'] / common.G_CONST
        assert(floor_mass >= 0.00),\
            "Error: floor area properties\n" + \
            "Overall floor area should be negative (by convention)."
        floor_centroid = properties['centroid']
        floor_mass_inertia = properties['inertia']['ir_mass']\
            * floor_mass
        self_mass_centroid = np.array([0.00, 0.00])  # excluding floor
        total_self_mass = 0.00
        for nd in lvl.list_of_all_nodes():
            self_mass_centroid += nd.coords[0:2] * nd.mass[0]
            total_self_mass += nd.mass[0]
        self_mass_centroid = self_mass_centroid * \
            (1.00/total_self_mass)
        total_mass = total_self_mass + floor_mass
        # combined
        centroid = [
            (self_mass_centroid[0] * total_self_mass +
             floor_centroid[0] * floor_mass) / total_mass,
            (self_mass_centroid[1] * total_self_mass +
             floor_centroid[1] * floor_mass) / total_mass
        ]
        lvl.parent_node = node.Node(
            np.array([centroid[0], centroid[1],
                      lvl.elevation]), "parent")
        lvl.parent_node.mass = np.array([total_mass,
                                         total_mass,
                                         0.,
                                         0., 0., 0.])
        lvl.parent_node.mass[5] = floor_mass_inertia
        for nd in lvl.list_of_all_nodes():
            lvl.parent_node.mass[5] += nd.mass[0] * \
                np.linalg.norm(lvl.parent_node.coords - nd.coords)**2
            nd.mass[0] = common.EPSILON
            nd.mass[1] = common.EPSILON
            nd.mass[2] = common.EPSILON
    mdl.update_required = True


def model_steel_frame_panel_zones(mdl):
    """
    Introduce necessary elements and readjust connectivity to model
    steel frame panel zones
    """
    mdl._update_lists()
    mdl.update_required = True

    for lvl in mdl.levels.registry.values():
        columns = list(lvl.columns.registry.values())
        for col in columns:
            nd = col.node_i
            # get a list of all the connected beams
            beams = nd.beams
            # determine the ones that are properly connected
            # for panel zone modeling
            panel_beam_front = None
            panel_beam_back = None
            panel_beam_front_side = None
            panel_beam_back_side = None
            for bm in beams:
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
                if bm.node_i == nd:
                    bm_other_node = bm.node_j
                    this_side = 'i'
                else:
                    bm_other_node = bm.node_i
                    this_side = 'j'
                bm_vec = (bm_other_node.coords - nd.coords)
                if np.dot(bm_vec, col.y_axis) > 0.00:
                    panel_beam_front = bm
                    panel_beam_front_side = this_side
                else:
                    panel_beam_back = bm
                    panel_beam_back_side = this_side
            # check that the beams are connected at the face of the column
            if panel_beam_front:
                if panel_beam_front_side == 'i':
                    offset = panel_beam_front.offset_i
                else:
                    offset = panel_beam_front.offset_j
                assert np.abs(np.linalg.norm(
                    2. * offset[0:2]) -
                    col.section.properties['d']) \
                    < common.EPSILON, \
                    'Incorrect connectivity'
            if panel_beam_back:
                if panel_beam_back_side == 'i':
                    offset = panel_beam_back.offset_i
                else:
                    offset = panel_beam_back.offset_j
                assert np.abs(np.linalg.norm(
                    2. * offset[0:2]) -
                    col.section.properties['d']) \
                    < common.EPSILON, \
                    'Incorrect connectivity'

            if panel_beam_front:
                beam_depth = \
                    panel_beam_front.section.properties['d']
            elif panel_beam_back:
                beam_depth = \
                    panel_beam_back.section.properties['d']

            if panel_beam_front and panel_beam_back:
                d_f = panel_beam_front.section.properties['d']
                d_b = panel_beam_back.section.properties['d']
                beam_depth = max([d_f, d_b])

            if not (panel_beam_front or panel_beam_back):
                # Don't model panel zones if no beams are
                # connected
                continue

            col.ends['end_dist'] = beam_depth
            p_i = col.node_i.coords + col.offset_i
            start_loc = p_i + col.x_axis * col.ends['end_dist']
            col.n_i = node.Node(start_loc)
            col.middle_segment = MiddleSegment(
                col,
                np.zeros(3).copy(),
                np.zeros(3).copy(),
                0.00)
            # generate end segment i (panel zone)
            if col.ends['type'] in [
                    'steel_W_PZ', 'fixed', 'fixed-pinned']:
                col.end_segment_i = EndSegment_Steel_W_PanelZone(
                    col, col.node_i, col.n_i, "i",
                    mdl.sections.registry['rigid'],
                    mdl.materials.registry['fix'])
            elif col.ends['type'] == 'steel_W_PZ_IMK':
                col.end_segment_i = EndSegment_Steel_W_PanelZone_IMK(
                    col, col.node_i, col.n_i, "i",
                    mdl.sections.registry['rigid'],
                    mdl.materials.registry['fix'])
            else:
                raise ValueError('Invalid end type for W column')

            # modify beam connectivity
            panel_zone_segment = col.end_segment_i
            if panel_beam_front:
                if panel_beam_front_side == 'i':
                    # grab the end segment
                    sgm = panel_beam_front.end_segment_i
                    if isinstance(sgm, EndSegment_Fixed):
                        # modify the node
                        panel_beam_front.middle_segment.\
                            first_line_elem().node_i = \
                            panel_zone_segment.n_front
                        # modify the offset of the lineelement
                        panel_beam_front.middle_segment.\
                            first_line_elem().offset_i = \
                            np.zeros(3).copy()
                    else:
                        # modify the node
                        sgm.first_line_elem().node_i = \
                            panel_zone_segment.n_front
                        # modify the offset of the lineelement
                        sgm.first_line_elem().offset_i = \
                            np.zeros(3).copy()
                    # modify the node of the endsegment
                    sgm.n_external = \
                        panel_zone_segment.n_front
                    # modify the node and offset of the lineelementsequence
                    panel_beam_front.node_i = panel_zone_segment.n_front
                    panel_beam_front.offset_i = np.zeros(3).copy()
                elif panel_beam_front_side == 'j':
                    # grab the end segment
                    sgm = panel_beam_front.end_segment_j
                    # modify the offset of the endsegment
                    sgm.offset = np.zeros(3).copy()
                    if isinstance(sgm, EndSegment_Fixed):
                        # modify the node
                        panel_beam_front.middle_segment.\
                            last_line_elem().node_j = \
                            panel_zone_segment.n_front
                        # modify the offset of the lineelement
                        panel_beam_front.middle_segment.\
                            last_line_elem().offset_j = \
                            np.zeros(3).copy()
                    else:
                        # modify the node
                        sgm.last_line_elem().node_j = \
                            panel_zone_segment.n_front
                        # modify the offset of the lineelement
                        sgm.last_line_elem().offset_j = \
                            np.zeros(3).copy()
                    # modify the node of the endsegment
                    sgm.n_external = \
                        panel_zone_segment.n_front
                    # modify the node and offset of the lineelementsequence
                    panel_beam_front.node_j = panel_zone_segment.n_front
                    panel_beam_front.offset_j = np.zeros(3).copy()
                else:
                    raise ValueError('This should never happen!')

            if panel_beam_back:
                if panel_beam_back_side == 'i':
                    # grab the end segment
                    sgm = panel_beam_back.end_segment_i
                    if isinstance(sgm, EndSegment_Fixed):
                        # modify the node
                        panel_beam_back.middle_segment.\
                            first_line_elem().node_i = \
                            panel_zone_segment.n_back
                        panel_beam_back.n_i = panel_zone_segment.n_back
                        # modify the offset of the lineelement
                        panel_beam_back.middle_segment.\
                            first_line_elem().offset_i = \
                            np.zeros(3).copy()
                    else:
                        # modify the node
                        sgm.first_line_elem().node_i = \
                            panel_zone_segment.n_back
                        # modify the offset of the lineelement
                        sgm.first_line_elem().offset_i = \
                            np.zeros(3).copy()
                    # modify the node of the endsegment
                    sgm.n_external = \
                        panel_zone_segment.n_back
                    # modify the node and offset of the lineelementsequence
                    panel_beam_back.node_i = panel_zone_segment.n_back
                    panel_beam_back.offset_i = np.zeros(3).copy()
                elif panel_beam_back_side == 'j':
                    # grab the end segment
                    sgm = panel_beam_back.end_segment_j
                    if isinstance(sgm, EndSegment_Fixed):
                        # modify the node
                        panel_beam_back.middle_segment.\
                            last_line_elem().node_j = \
                            panel_zone_segment.n_back
                        # modify the offset of the lineelement
                        panel_beam_back.middle_segment.\
                            last_line_elem().offset_j = \
                            np.zeros(3).copy()
                    else:
                        # modify the node
                        sgm.last_line_elem().node_j = \
                            panel_zone_segment.n_back
                        # modify the offset of the lineelement
                        sgm.last_line_elem().offset_j = \
                            np.zeros(3).copy()
                    # modify the node of the endsegment
                    sgm.n_external = \
                        panel_zone_segment.n_back
                    # modify the node and offset of the lineelementsequence
                    panel_beam_back.node_j = panel_zone_segment.n_back
                    panel_beam_back.offset_j = np.zeros(3).copy()
                else:
                    raise ValueError('This should never happen!')


def elevate_steel_column_splices(mdl, relative_len):
    """
    Brings column splices higher than the level transition elevation.
    """
    mdl._update_lists()
    mdl.update_required = True
    for col in mdl.list_of_columns():
        # check to see if there is a column at the level below
        n_j = col.node_j
        if n_j.column_below:
            sec = n_j.column_below.section
            if sec is not col.section:
                for elm in col.end_segment_j.internal_line_elems.values():
                    elm.section = sec
                z_test = col.node_j.coords[2] + \
                    col.length_clear * relative_len
                for elm in col.middle_segment.internal_line_elems.values():
                    if elm.node_i.coords[2] < z_test:
                        elm.section = sec
