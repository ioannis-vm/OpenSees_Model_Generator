"""
The following utility functions are used for data visualization
https://plotly.com/python/reference/
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
from typing import Optional
import sys
import plotly.graph_objects as go  # type: ignore
import numpy as np
import numpy.typing as npt
from . import graphics_common, graphics_common_3d
if TYPE_CHECKING:
    from model import Model
    from ..load_case import LoadCase

nparr = npt.NDArray[np.float64]


def add_data__nodes(data_dict, mdl, load_case):
    """
    Adds a trace containing nodes
    Arguments:
      data_dict (dict): dictionary containing figure data
      mdl (Model): the model to be visualized
      load_case (LoadCase): the load_case to be visualized
    """
    # TODO: this was written very sloppily, because I was in a rush.
    # I need to improve this code, avoid code repetition
    # TODO: merge this method with the other methods that plot nodes.
    list_of_nodes = mdl.list_of_primary_nodes()
    x_list = [node.coords[0] for node in list_of_nodes]
    y_list = [node.coords[1] for node in list_of_nodes]
    z_list = [node.coords[2] for node in list_of_nodes]
    customdata_lst = []
    restraints = [node.restraint for node in list_of_nodes]
    restraint_symbols = []
    for node in list_of_nodes:
        if True in node.restraint:
            restraint_symbols.append("fixed")
        else:
            restraint_symbols.append("free")
    for node in list_of_nodes:
        if load_case:
            customdata_lst.append(
                (node.uid,
                 *load_case.node_mass[node.uid].val,
                 *load_case.node_loads[node.uid].val
                 )
            )
        else:
            customdata_lst.append(
                (node.uid,
                 )
            )
    customdata: nparr = np.array(customdata_lst, dtype='object')
    if load_case:
        data_dict.append({
            "name": "Primary nodes",
            "type": "scatter3d",
            "mode": "markers",
            "x": x_list,
            "y": y_list,
            "z": z_list,
            "customdata": customdata,
            "text": restraints,
            "hovertemplate": 'Coordinates: '
            '(%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
            'Restraint: %{text}<br>' +
            'Mass: (%{customdata[1]:.3g}, ' +
            '%{customdata[2]:.3g}, %{customdata[3]:.3g}, ' +
            '%{customdata[4]:.3g}, %{customdata[5]:.3g}, ' +
            '%{customdata[6]:.3g})<br>' +
            'Load: (%{customdata[7]:.3g}, ' +
            '%{customdata[8]:.3g}, %{customdata[9]:.3g}, ' +
            '%{customdata[10]:.3g}, %{customdata[11]:.3g}, ' +
            '%{customdata[12]:.3g})' +
            '<extra>Node: %{customdata[0]:d}</extra>',
            "marker": {
                "symbol": [graphics_common_3d.node_marker[sym][0]
                           for sym in restraint_symbols],
                "color": graphics_common.NODE_PRIMARY_COLOR,
                "size": [graphics_common_3d.node_marker[sym][1]
                         for sym in restraint_symbols],
                "line": {
                    "color": graphics_common.NODE_PRIMARY_COLOR,
                    "width": 4}
            }
        })
    else:
        data_dict.append({
            "name": "Primary nodes",
            "type": "scatter3d",
            "mode": "markers",
            "x": x_list,
            "y": y_list,
            "z": z_list,
            "customdata": customdata,
            "text": restraints,
            "hovertemplate": 'Coordinates: '
            '(%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
            'Restraint: %{text}<br>' +
            '<extra>Node: %{customdata[0]:d}</extra>',
            "marker": {
                "symbol": [graphics_common_3d.node_marker[sym][0]
                           for sym in restraint_symbols],
                "color": graphics_common.NODE_PRIMARY_COLOR,
                "size": [graphics_common_3d.node_marker[sym][1]
                         for sym in restraint_symbols],
                "line": {
                    "color": graphics_common.NODE_PRIMARY_COLOR,
                    "width": 4}
            }
        })


def add_data__parent_nodes(data_dict, load_case: LoadCase):
    """
    Adds a trace containing parent nodes
    Arguments:
      data_dict (dict): dictionary containing figure data
      mdl (Model): the model to be visualized
      load_case (LoadCase): the load_case to be visualized
    """
    list_of_nodes = load_case.parent_nodes.values()
    x_list = [node.coords[0] for node in list_of_nodes]
    y_list = [node.coords[1] for node in list_of_nodes]
    z_list = [node.coords[2] for node in list_of_nodes]
    customdata_list = []
    restraints = [node.restraint for node in list_of_nodes]
    for node in list_of_nodes:
        customdata_list.append(
            (node.uid,
             *load_case.node_mass[node.uid].val,
             *load_case.node_loads[node.uid].val
             )
        )

    customdata: nparr = np.array(customdata_list, dtype='object')
    data_dict.append({
        "type": "scatter3d",
        "mode": "markers",
        "x": x_list,
        "y": y_list,
        "z": z_list,
        "customdata": customdata,
        "text": restraints,
        "hovertemplate": 'Coordinates: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
        'Restraint: %{text}<br>' +
        'Mass: (%{customdata[1]:.3g}, ' +
        '%{customdata[2]:.3g}, %{customdata[3]:.3g}, ' +
        '%{customdata[4]:.3g}, %{customdata[5]:.3g}, ' +
        '%{customdata[6]:.3g})<br>' +
        'Load: (%{customdata[7]:.3g}, ' +
        '%{customdata[8]:.3g}, %{customdata[9]:.3g}, ' +
        '%{customdata[10]:.3g}, %{customdata[11]:.3g}, ' +
        '%{customdata[12]:.3g})' +
        '<extra>Parent Node: %{customdata[0]:d}</extra>',
        "marker": {
            "symbol": [graphics_common_3d.node_marker['parent'][0]
                       for node in list_of_nodes],
            "color": graphics_common.NODE_PRIMARY_COLOR,
            "size": [graphics_common_3d.node_marker['parent'][1]
                     for node in list_of_nodes],
            "line": {
                "color": graphics_common.NODE_PRIMARY_COLOR,
                "width": 4}
        }
    })


def add_data__internal_nodes(data_dict, mdl, load_case):
    """
    Adds a trace containing internal nodes
    Arguments:
      data_dict (dict): dictionary containing figure data
      mdl (Model): the model to be visualized
      load_case (LoadCase): the load_case to be visualized
    """
    list_of_nodes = mdl.list_of_internal_nodes()
    x_list = [node.coords[0] for node in list_of_nodes]
    y_list = [node.coords[1] for node in list_of_nodes]
    z_list = [node.coords[2] for node in list_of_nodes]
    customdata = []
    restraints = [node.restraint for node in list_of_nodes]
    restraint_symbols = []
    for node in list_of_nodes:
        if True in node.restraint:
            restraint_symbols.append("fixed")
        elif node.visibility.connected_to_zerolength:
            restraint_symbols.append("release")
        else:
            restraint_symbols.append("internal")
    for node in list_of_nodes:
        if load_case:
            customdata.append(
                (node.uid,
                 *load_case.node_mass[node.uid].val,
                 *load_case.node_loads[node.uid].val
                 )
            )
        else:
            customdata.append(
                (node.uid,
                 )
            )
    if load_case:
        data_dict.append({
            "name": "Internal nodes",
            "type": "scatter3d",
            "mode": "markers",
            "x": x_list,
            "y": y_list,
            "z": z_list,
            "customdata": customdata,
            "text": restraints,
            "hovertemplate": 'Coordinates: '
            '(%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
            'Restraint: %{text}<br>' +
            'Mass: (%{customdata[1]:.3g}, ' +
            '%{customdata[2]:.3g}, %{customdata[3]:.3g})<br>' +
            'Load: (%{customdata[4]:.3g}, ' +
            '%{customdata[5]:.3g}, %{customdata[6]:.3g})' +
            '<extra>Node: %{customdata[0]:d}</extra>',
            "marker": {
                "symbol": [graphics_common_3d.node_marker[sym][0]
                           for sym in restraint_symbols],
                "color": graphics_common.NODE_INTERNAL_COLOR,
                "size": [graphics_common_3d.node_marker[sym][1]
                         for sym in restraint_symbols],
                "line": {
                    "color": graphics_common.NODE_INTERNAL_COLOR,
                    "width": 2}
            }
        })
    else:
        data_dict.append({
            "name": "Internal nodes",
            "type": "scatter3d",
            "mode": "markers",
            "x": x_list,
            "y": y_list,
            "z": z_list,
            "customdata": customdata,
            "text": restraints,
            "hovertemplate": 'Coordinates: '
            '(%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
            'Restraint: %{text}<br>' +
            '<extra>Node: %{customdata[0]:d}</extra>',
            "marker": {
                "symbol": [graphics_common_3d.node_marker[sym][0]
                           for sym in restraint_symbols],
                "color": graphics_common.NODE_INTERNAL_COLOR,
                "size": [graphics_common_3d.node_marker[sym][1]
                         for sym in restraint_symbols],
                "line": {
                    "color": graphics_common.NODE_INTERNAL_COLOR,
                    "width": 2}
            }
        })


def add_data__release_nodes(data_dict, list_of_nodes):
    """
    Adds a trace containing release nodes
    Arguments:
      data_dict (dict): dictionary containing figure data
      mdl (Model): the model to be visualized
      load_case (LoadCase): the load_case to be visualized
    """
    x_list = [node.coords[0] for node in list_of_nodes]
    y_list = [node.coords[1] for node in list_of_nodes]
    z_list = [node.coords[2] for node in list_of_nodes]
    customdata = []
    for node in list_of_nodes:
        customdata.append(
            (node.uid,
             *node.mass,
             *node.load_total()
             )
        )
    data_dict.append({
        "type": "scatter3d",
        "mode": "markers",
        "x": x_list,
        "y": y_list,
        "z": z_list,
        "hoverinfo": "skip",
        "marker": {
            "symbol": graphics_common_3d.node_marker['pinned'][0],
            "color": graphics_common.NODE_INTERNAL_COLOR,
            "size": graphics_common_3d.node_marker['pinned'][1],
            "line": {
                "color": graphics_common.NODE_INTERNAL_COLOR,
                "width": 2}
        }
    })


# def add_data__diaphragm_lines(data_dict, lvl):
#     """
#     Adds a trace containing rigid diaphragm indicator lines
#     Arguments:
#       data_dict (dict): dictionary containing figure data
#       lvl (Level): a level
#     """
#     if not lvl.parent_node:
#         return
#     parent_node = lvl.parent_node
#     x_list: list[Optional[float]] = []
#     y_list: list[Optional[float]] = []
#     z_list: list[Optional[float]] = []
#     for node in lvl.nodes_primary.values():
#         x_list.extend(
#             (node.coords[0], parent_node.coords[0], None)
#         )
#         y_list.extend(
#             (node.coords[1], parent_node.coords[1], None)
#         )
#         z_list.extend(
#             (node.coords[2], parent_node.coords[2], None)
#         )
#     data_dict.append({
#         "type": "scatter3d",
#         "mode": "lines",
#         "x": x_list,
#         "y": y_list,
#         "z": z_list,
#         "hoverinfo": "skip",
#         "line": {
#             "width": 4,
#             "dash": "dash",
#             "color": graphics_common.GRID_COLOR
#         }
#     })


# def add_data__bisector_lines(data_dict, lvl):
#     """
#     Adds a trace containing tributary area bisector lines
#     Arguments:
#       data_dict (dict): dictionary containing figure data
#       lvl (Level): a level
#     """
#     if not lvl.parent_node:
#         return
#     x_list: list[Optional[float]] = []
#     y_list: list[Optional[float]] = []
#     z_list: list[Optional[float]] = []
#     for line in lvl.floor_bisector_lines:
#         p_1 = line[0]
#         p_2 = line[1]
#         x_list.extend(
#             (p_1[0], p_2[0], None)
#         )
#         y_list.extend(
#             (p_1[1], p_2[1], None)
#         )
#         z_list.extend(
#             (lvl.elevation, lvl.elevation, None)
#         )
#     data_dict.append({
#         "type": "scatter3d",
#         "mode": "lines",
#         "x": x_list,
#         "y": y_list,
#         "z": z_list,
#         "hoverinfo": "skip",
#         "line": {
#             "width": 4,
#             "color": graphics_common.BISECTOR_COLOR
#         }
#     })


def add_data__frames(data_dict, mdl, load_case):
    """
    Adds a trace containing frame element centroidal axis lines
    Arguments:
      data_dict (dict): dictionary containing figure data
      mdl (Model): the model to be visualized
      load_case (LoadCase): the load_case to be visualized
    """
    beamcolumn_elems = mdl.list_of_beamcolumn_elements()
    if not beamcolumn_elems:
        return
    x_list: list[Optional[float]] = []
    y_list: list[Optional[float]] = []
    z_list: list[Optional[float]] = []
    customdata_list = []
    section_names = []
    for elm in beamcolumn_elems:
        if elm.visibility.hidden_at_line_plots:
            continue
        p_i = np.array(elm.nodes[0].coords) + elm.geomtransf.offset_i
        p_j = np.array(elm.nodes[1].coords) + elm.geomtransf.offset_j
        section_names.extend([elm.section.name]*3)
        x_list.extend(
            (p_i[0], p_j[0], None)
        )
        y_list.extend(
            (p_i[1], p_j[1], None)
        )
        z_list.extend(
            (p_i[2], p_j[2], None)
        )
        if load_case:
            customdata_list.append(
                (elm.uid,
                 *load_case.line_element_udl[elm.uid].val,
                 elm.nodes[0].uid,
                 elm.parent_component.uid)
            )
            customdata_list.append(
                (elm.uid,
                 *load_case.line_element_udl[elm.uid].val,
                 elm.nodes[1].uid,
                 elm.parent_component.uid)
            )
            customdata_list.append(
                (None,)*6
            )
        else:
            customdata_list.append(
                (elm.uid,
                 elm.nodes[0].uid,
                 elm.parent_component.uid)
            )
            customdata_list.append(
                (elm.uid,
                 elm.nodes[1].uid,
                 elm.parent_component.uid)
            )
            customdata_list.append(
                (None,)*3
            )

    if load_case:
        customdata: nparr = np.array(customdata_list, dtype='object')
        data_dict.append({
            "name": "Frame elements",
            "type": "scatter3d",
            "mode": "lines",
            "x": x_list,
            "y": y_list,
            "z": z_list,
            "text": section_names,
            "customdata": customdata,
            "hovertemplate": 'Section: %{text}<br>' +
            'UDL (local): (%{customdata[1]:.3g}, ' +
            '%{customdata[2]:.3g}, %{customdata[3]:.3g})' +
            '<extra>Element: %{customdata[0]:d}<br>' +
            'Node @ this end: %{customdata[4]:d}<br>'
            'Parent: %{customdata[5]}</extra>',
            "line": {
                "width": 5,
                "color": graphics_common.FRAME_COLOR
            }
        })
    else:
        customdata = np.array(customdata_list, dtype='object')
        data_dict.append({
            "name": "Frame elements",
            "type": "scatter3d",
            "mode": "lines",
            "x": x_list,
            "y": y_list,
            "z": z_list,
            "text": section_names,
            "customdata": customdata,
            "hovertemplate": 'Section: %{text}<br>' +
            '<extra>Element: %{customdata[0]:d}<br>' +
            'Node @ this end: %{customdata[1]:d}<br>'
            'Parent: %{customdata[2]}</extra>',
            "line": {
                "width": 5,
                "color": graphics_common.FRAME_COLOR
            }
        })


def add_data__twonodelinks(data_dict, mdl):
    """
    Adds a trace containing twonodelink elements
    Arguments:
      data_dict (dict): dictionary containing figure data
      mdl (Model): the model to be visualized
    """
    link_elems = mdl.list_of_twonodelink_elements()
    if not link_elems:
        return
    x_list: list[Optional[float]] = []
    y_list: list[Optional[float]] = []
    z_list: list[Optional[float]] = []
    customdata_list = []
    for elm in link_elems:
        p_i: nparr = np.array(elm.nodes[0].coords)
        p_j: nparr = np.array(elm.nodes[1].coords)
        x_list.extend(
            (p_i[0], p_j[0], None)
        )
        y_list.extend(
            (p_i[1], p_j[1], None)
        )
        z_list.extend(
            (p_i[2], p_j[2], None)
        )
        customdata_list.append(
            (elm.uid,
             elm.nodes[0].uid,
             elm.parent_component.uid)
        )
        customdata_list.append(
            (elm.uid,
             elm.nodes[1].uid,
             elm.parent_component.uid)
        )
        customdata_list.append(
            (None,)*3
        )

    customdata: nparr = np.array(customdata_list, dtype='object')
    data_dict.append({
        "name": "TwoNodeLink elements",
        "type": "scatter3d",
        "mode": "lines",
        "x": x_list,
        "y": y_list,
        "z": z_list,
        # "text": section_names,
        "customdata": customdata,
        "hovertemplate": 'Section: %{text}<br>' +
        '<extra>Element: %{customdata[0]:d}<br>' +
        'Node @ this end: %{customdata[1]:d}<br>'
        'Parent: %{customdata[2]}</extra>',
        "line": {
            "width": 5,
            "color": graphics_common.LINK_COLOR
        }
    })


def add_data__frame_offsets(data_dict, mdl):
    """
    Adds a trace containing frame element rigid offset lines
    Arguments:
      data_dict (dict): dictionary containing figure data
      mdl (Model): the model to be visualized
    """
    beamcolumn_elems = mdl.list_of_beamcolumn_elements()
    if not beamcolumn_elems:
        return

    x_list: list[Optional[float]] = []
    y_list: list[Optional[float]] = []
    z_list: list[Optional[float]] = []

    for elm in beamcolumn_elems:
        p_i: nparr = np.array(elm.nodes[0].coords)
        p_io: nparr = (np.array(elm.nodes[0].coords)
                       + elm.geomtransf.offset_i)
        p_j: nparr = np.array(elm.nodes[1].coords)
        p_jo: nparr = (np.array(elm.nodes[1].coords)
                       + elm.geomtransf.offset_j)

        x_list.extend((p_i[0], p_io[0], None))
        y_list.extend((p_i[1], p_io[1], None))
        z_list.extend((p_i[2], p_io[2], None))
        x_list.extend((p_j[0], p_jo[0], None))
        y_list.extend((p_j[1], p_jo[1], None))
        z_list.extend((p_j[2], p_jo[2], None))

    data_dict.append({
        "name": "Rigid offsets",
        "type": "scatter3d",
        "mode": "lines",
        "x": x_list,
        "y": y_list,
        "z": z_list,
        "hoverinfo": "skip",
        "line": {
            "width": 8,
            "color": graphics_common.OFFSET_COLOR
        }
    })


def add_data__frame_axes(data_dict, mdl, ref_len):
    """
    Adds a trace containing frame element local axis lines
    Arguments:
      data_dict (dict): dictionary containing figure data
      mdl (Model): the model to be visualized
      ref_len (float): model reference length to scale the axes
    """
    beamcolumn_elems = mdl.list_of_beamcolumn_elements()
    if not beamcolumn_elems:
        return
    scaling = ref_len * 0.025
    x_list: list[Optional[float]] = []
    y_list: list[Optional[float]] = []
    z_list: list[Optional[float]] = []
    colors: list[Optional[str]] = []
    for elm in beamcolumn_elems:
        if elm.visibility.hidden_at_line_plots:
            continue
        x_vec = elm.geomtransf.x_axis
        y_vec = elm.geomtransf.y_axis
        z_vec = elm.geomtransf.z_axis
        l_clear = elm.clear_length()
        i_pos = np.array(elm.nodes[0].coords) + elm.geomtransf.offset_i
        mid_pos = i_pos + x_vec * l_clear/2.00
        x_list.extend((mid_pos[0], mid_pos[0]+x_vec[0]*scaling, None))
        y_list.extend((mid_pos[1], mid_pos[1]+x_vec[1]*scaling, None))
        z_list.extend((mid_pos[2], mid_pos[2]+x_vec[2]*scaling, None))
        colors.extend(["red"]*3)
        x_list.extend((mid_pos[0], mid_pos[0]+y_vec[0]*scaling, None))
        y_list.extend((mid_pos[1], mid_pos[1]+y_vec[1]*scaling, None))
        z_list.extend((mid_pos[2], mid_pos[2]+y_vec[2]*scaling, None))
        colors.extend(["green"]*3)
        x_list.extend((mid_pos[0], mid_pos[0]+z_vec[0]*scaling, None))
        y_list.extend((mid_pos[1], mid_pos[1]+z_vec[1]*scaling, None))
        z_list.extend((mid_pos[2], mid_pos[2]+z_vec[2]*scaling, None))
        colors.extend(["blue"]*3)
    data_dict.append({
        "name": "Frame axes",
        "type": "scatter3d",
        "mode": "lines",
        "x": x_list,
        "y": y_list,
        "z": z_list,
        "hoverinfo": "skip",
        "line": {
            "width": 8,
            "color": colors
        }
    })


def add_data__zerolength_axes(data_dict, mdl, ref_len):
    """
    Adds a trace containing zerolength element local axis lines
    Arguments:
      data_dict (dict): dictionary containing figure data
      mdl (Model): the model to be visualized
      ref_len (float): model reference length to scale the axes
    """
    zerolength_elements = mdl.list_of_zerolength_elements()
    if not zerolength_elements:
        return
    scaling = ref_len * 0.025
    x_list: list[Optional[float]] = []
    y_list: list[Optional[float]] = []
    z_list: list[Optional[float]] = []
    colors: list[Optional[str]] = []
    for elm in zerolength_elements:
        x_vec: nparr = elm.vecx
        y_vec: nparr = elm.vecyp
        z_vec: nparr = np.cross(x_vec, y_vec)
        mid_pos = np.array(elm.nodes[0].coords)
        x_list.extend((mid_pos[0], mid_pos[0]+x_vec[0]*scaling, None))
        y_list.extend((mid_pos[1], mid_pos[1]+x_vec[1]*scaling, None))
        z_list.extend((mid_pos[2], mid_pos[2]+x_vec[2]*scaling, None))
        colors.extend(["red"]*3)
        x_list.extend((mid_pos[0], mid_pos[0]+y_vec[0]*scaling, None))
        y_list.extend((mid_pos[1], mid_pos[1]+y_vec[1]*scaling, None))
        z_list.extend((mid_pos[2], mid_pos[2]+y_vec[2]*scaling, None))
        colors.extend(["green"]*3)
        x_list.extend((mid_pos[0], mid_pos[0]+z_vec[0]*scaling, None))
        y_list.extend((mid_pos[1], mid_pos[1]+z_vec[1]*scaling, None))
        z_list.extend((mid_pos[2], mid_pos[2]+z_vec[2]*scaling, None))
        colors.extend(["blue"]*3)
    data_dict.append({
        "name": "Zerolength axes",
        "type": "scatter3d",
        "mode": "lines",
        "x": x_list,
        "y": y_list,
        "z": z_list,
        "hoverinfo": "skip",
        "line": {
            "width": 8,
            "color": colors
        }
    })


def add_data__global_axes(data_dict, ref_len):
    """
    Adds a trace containing global axes
    Arguments:
      data_dict (dict): dictionary containing figure data
      ref_len (float): model reference length to scale the axes
    """
    scaling = ref_len
    x_list: list[Optional[float]] = []
    y_list: list[Optional[float]] = []
    z_list: list[Optional[float]] = []
    colors: list[Optional[str]] = []
    x_vec: nparr = np.array([1.00, 0.00, 0.00])
    y_vec: nparr = np.array([0.00, 1.00, 0.00])
    z_vec: nparr = np.array([0.00, 0.00, 1.00])

    x_list.extend((0.00, x_vec[0]*scaling, None))
    y_list.extend((0.00, x_vec[1]*scaling, None))
    z_list.extend((0.00, x_vec[2]*scaling, None))
    x_list.extend((0.00, y_vec[0]*scaling, None))
    y_list.extend((0.00, y_vec[1]*scaling, None))
    z_list.extend((0.00, y_vec[2]*scaling, None))
    x_list.extend((0.00, z_vec[0]*scaling, None))
    y_list.extend((0.00, z_vec[1]*scaling, None))
    z_list.extend((0.00, z_vec[2]*scaling, None))
    colors.extend(["red"]*3)
    colors.extend(["green"]*3)
    colors.extend(["blue"]*3)
    # we add it twice in order for the animations to work
    # see https://plotly.com/python/animations/
    data_dict.extend([{
        "type": "scatter3d",
        "mode": "lines",
        "x": x_list,
        "y": y_list,
        "z": z_list,
        "hoverinfo": "skip",
        "line": {
            "width": 3,
            "color": colors
        }
    }]*2)


def add_data__extruded_frames_mesh(data_dict, mdl):
    """
    Adds a trace containing frame element extrusion mesh
    Arguments:
      data_dict (dict): dictionary containing figure data
      mdl (Model): the model to be visualized
    """
    beamcolumn_elems = mdl.list_of_beamcolumn_elements()
    if not beamcolumn_elems:
        return
    x_list: list[Optional[float]] = []
    y_list: list[Optional[float]] = []
    z_list: list[Optional[float]] = []
    i_list: list[Optional[int]] = []
    j_list: list[Optional[int]] = []
    k_list: list[Optional[int]] = []
    index = 0
    for elm in beamcolumn_elems:
        if elm.visibility.hidden_when_extruded:
            continue
        side_a = np.array(elm.nodes[0].coords) + elm.geomtransf.offset_i
        side_b = np.array(elm.nodes[1].coords) + elm.geomtransf.offset_j
        y_vec = elm.geomtransf.y_axis
        z_vec = elm.geomtransf.z_axis
        if not elm.section.outside_shape:
            continue
        loop = elm.section.outside_shape.halfedges
        for halfedge in loop:
            loc0 = halfedge.vertex.coords[0]*z_vec +\
                halfedge.vertex.coords[1]*y_vec + side_a
            loc1 = halfedge.vertex.coords[0]*z_vec +\
                halfedge.vertex.coords[1]*y_vec + side_b
            loc2 = halfedge.nxt.vertex.coords[0]*z_vec +\
                halfedge.nxt.vertex.coords[1]*y_vec + side_b
            loc3 = halfedge.nxt.vertex.coords[0]*z_vec +\
                halfedge.nxt.vertex.coords[1]*y_vec + side_a
            x_list.append(loc0[0])
            y_list.append(loc0[1])
            z_list.append(loc0[2])
            x_list.append(loc1[0])
            y_list.append(loc1[1])
            z_list.append(loc1[2])
            x_list.append(loc2[0])
            y_list.append(loc2[1])
            z_list.append(loc2[2])
            x_list.append(loc3[0])
            y_list.append(loc3[1])
            z_list.append(loc3[2])
            i_list.append(index + 0)
            j_list.append(index + 1)
            k_list.append(index + 2)
            i_list.append(index + 0)
            j_list.append(index + 2)
            k_list.append(index + 3)
            index += 4
    data_dict.append({
        "type": "mesh3d",
        "x": x_list,
        "y": y_list,
        "z": z_list,
        "i": i_list,
        "j": j_list,
        "k": k_list,
        "hoverinfo": "skip",
        "color": graphics_common.BEAM_MESH_COLOR,
        "opacity": 0.30
    })


def show(mdl: Model,
         load_case: Optional[LoadCase] = None,
         extrude=False,
         offsets=True,
         global_axes=True,
         parent_nodes=True,
         frame_axes=False,
         zerolength_axes=False,
         camera=None
         ):
    """
    Visualize the model
    Arguments:
      mdl (Model): the model to be visualized
      load_case (LoadCase): the load_case to be visualized
      extrude (bool): wether to extrude frame elements
      offsets (bool): whether to show frame element rigid offsets
      global_axes (bool): whether to show global axes
      diaphragm_lines (bool): whether to show lines indicating rigid
        diaphragm extent
      tributary_areas (bool): whether to show tributary area
        boundaries
      parent_nodes (bool): whether to plot parent nodes
      frame_axes (bool): whether to show the local axes of frame
        elements
      zerolength_axes (bool): whether to show the local axes of
        zerolength elements
      camera (dict): custom positioning of the camera
    """

    layout = graphics_common_3d.global_layout(mdl, camera)
    data_dict: list[dict[str, object]] = []

    ref_len = mdl.reference_length()

    # plot the nodes

    add_data__nodes(data_dict, mdl, load_case)
    add_data__internal_nodes(data_dict, mdl, load_case)

    # global axes
    if global_axes:
        add_data__global_axes(data_dict, ref_len)

    # # diaphgragm lines
    # if diaphragm_lines:
    #     for lvl in mdl.levels.values():
    #         add_data__diaphragm_lines(data_dict, lvl)

    # # bisector lines
    # if tributary_areas:
    #     for lvl in mdl.levels.values():
    #         add_data__bisector_lines(data_dict, lvl)

    # plot beamcolumn elements
    add_data__frames(data_dict, mdl, load_case)
    if frame_axes:
        add_data__frame_axes(data_dict, mdl, ref_len)
    if zerolength_axes:
        add_data__zerolength_axes(data_dict, mdl, ref_len)
    if extrude:
        add_data__extruded_frames_mesh(
            data_dict, mdl)
    # plot the rigid offsets
    if offsets:
        add_data__frame_offsets(data_dict, mdl)
    add_data__twonodelinks(data_dict, mdl)
    # plot the parent nodes
    if parent_nodes:
        if load_case:
            add_data__parent_nodes(data_dict, load_case)

    fig_datastructure = dict(data=data_dict, layout=layout)
    fig = go.Figure(fig_datastructure)

    if "pytest" not in sys.modules:
        fig.show()
