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
from typing import Any
import sys
import plotly.graph_objects as go  # type: ignore
import numpy as np
import numpy.typing as npt
from . import graphics_common, graphics_common_3D
from ..ops.element import elasticBeamColumn
from ..ops.element import dispBeamColumn
if TYPE_CHECKING:
    from model import Model
    from ..load_case import LoadCase


def add_data__nodes(dt, mdl, load_case):
    # Todo: this was written very sloppily, because I was in a rush.
    # improve this code, avoid code repetition
    list_of_nodes = mdl.list_of_primary_nodes()
    x = [node.coords[0] for node in list_of_nodes]
    y = [node.coords[1] for node in list_of_nodes]
    z = [node.coords[2] for node in list_of_nodes]
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
                 *load_case.node_mass.registry[node.uid].total(),
                 *load_case.node_loads.registry[node.uid].total()
                 )
            )
        else:
            customdata_lst.append(
                (node.uid,
                 )
            )
    customdata: Any = np.array(customdata_lst, dtype='object')
    if load_case:
        dt.append({
            "name": "Primary nodes",
            "type": "scatter3d",
            "mode": "markers",
            "x": x,
            "y": y,
            "z": z,
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
            '<extra>Node: %{customdata[0]:d}</extra>',
            "marker": {
                "symbol": [graphics_common_3D.node_marker[sym][0]
                           for sym in restraint_symbols],
                "color": graphics_common.NODE_PRIMARY_COLOR,
                "size": [graphics_common_3D.node_marker[sym][1]
                         for sym in restraint_symbols],
                "line": {
                    "color": graphics_common.NODE_PRIMARY_COLOR,
                    "width": 4}
            }
        })
    else:
        dt.append({
            "name": "Primary nodes",
            "type": "scatter3d",
            "mode": "markers",
            "x": x,
            "y": y,
            "z": z,
            "customdata": customdata,
            "text": restraints,
            "hovertemplate": 'Coordinates: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
            'Restraint: %{text}<br>' +
            '<extra>Node: %{customdata[0]:d}</extra>',
            "marker": {
                "symbol": [graphics_common_3D.node_marker[sym][0]
                           for sym in restraint_symbols],
                "color": graphics_common.NODE_PRIMARY_COLOR,
                "size": [graphics_common_3D.node_marker[sym][1]
                         for sym in restraint_symbols],
                "line": {
                    "color": graphics_common.NODE_PRIMARY_COLOR,
                    "width": 4}
            }
        })
        


def add_data__parent_nodes(dt, list_of_nodes):

    x = [node.coords[0] for node in list_of_nodes]
    y = [node.coords[1] for node in list_of_nodes]
    z = [node.coords[2] for node in list_of_nodes]
    customdata = []
    restraint_types = [node.restraint_type for node in list_of_nodes]
    for node in list_of_nodes:
        customdata.append(
            (node.uid,
             *load_case.node_mass.registry[node.uid].total(),
             *node.load_total()
             )
        )

    customdata = np.array(customdata, dtype='object')
    dt.append({
        "type": "scatter3d",
        "mode": "markers",
        "x": x,
        "y": y,
        "z": z,
        "customdata": customdata,
        "text": restraint_types,
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
            "symbol": [graphics_common_3D.node_marker[node.restraint_type][0]
                       for node in list_of_nodes],
            "color": graphics_common.NODE_PRIMARY_COLOR,
            "size": [graphics_common_3D.node_marker[node.restraint_type][1]
                     for node in list_of_nodes],
            "line": {
                "color": graphics_common.NODE_PRIMARY_COLOR,
                "width": 4}
        }
    })


def add_data__internal_nodes(dt, mdl, load_case):
    list_of_nodes = mdl.list_of_internal_nodes()
    x = [node.coords[0] for node in list_of_nodes]
    y = [node.coords[1] for node in list_of_nodes]
    z = [node.coords[2] for node in list_of_nodes]
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
                 *load_case.node_mass.registry[node.uid].total(),
                 *load_case.node_loads.registry[node.uid].total()
                 )
            )
        else:
            customdata.append(
                (node.uid,
                 )
            )
    if load_case:
        dt.append({
            "name": "Internal nodes",
            "type": "scatter3d",
            "mode": "markers",
            "x": x,
            "y": y,
            "z": z,
            "customdata": customdata,
            "text": restraints,
            "hovertemplate": 'Coordinates: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
            'Restraint: %{text}<br>' +
            'Mass: (%{customdata[1]:.3g}, ' +
            '%{customdata[2]:.3g}, %{customdata[3]:.3g})<br>' +
            'Load: (%{customdata[4]:.3g}, ' +
            '%{customdata[5]:.3g}, %{customdata[6]:.3g})' +
            '<extra>Node: %{customdata[0]:d}</extra>',
            "marker": {
                "symbol": [graphics_common_3D.node_marker[sym][0]
                           for sym in restraint_symbols],
                "color": graphics_common.NODE_INTERNAL_COLOR,
                "size": [graphics_common_3D.node_marker[sym][1]
                           for sym in restraint_symbols],
                "line": {
                    "color": graphics_common.NODE_INTERNAL_COLOR,
                    "width": 2}
            }
        })
    else:
        dt.append({
            "name": "Internal nodes",
            "type": "scatter3d",
            "mode": "markers",
            "x": x,
            "y": y,
            "z": z,
            "customdata": customdata,
            "text": restraints,
            "hovertemplate": 'Coordinates: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
            'Restraint: %{text}<br>' +
            '<extra>Node: %{customdata[0]:d}</extra>',
            "marker": {
                "symbol": [graphics_common_3D.node_marker[sym][0]
                           for sym in restraint_symbols],
                "color": graphics_common.NODE_INTERNAL_COLOR,
                "size": [graphics_common_3D.node_marker[sym][1]
                           for sym in restraint_symbols],
                "line": {
                    "color": graphics_common.NODE_INTERNAL_COLOR,
                    "width": 2}
            }
        })
        


def add_data__release_nodes(dt, list_of_nodes):
    x = [node.coords[0] for node in list_of_nodes]
    y = [node.coords[1] for node in list_of_nodes]
    z = [node.coords[2] for node in list_of_nodes]
    customdata = []
    restraint_types = [node.restraint_type for node in list_of_nodes]
    for node in list_of_nodes:
        customdata.append(
            (node.uid,
             *node.mass,
             *node.load_total()
             )
        )
    dt.append({
        "type": "scatter3d",
        "mode": "markers",
        "x": x,
        "y": y,
        "z": z,
        "hoverinfo": "skip",
        "marker": {
            "symbol": graphics_common_3D.node_marker['pinned'][0],
            "color": graphics_common.NODE_INTERNAL_COLOR,
            "size": graphics_common_3D.node_marker['pinned'][1],
            "line": {
                "color": graphics_common.NODE_INTERNAL_COLOR,
                "width": 2}
        }
    })


def add_data__diaphragm_lines(dt, lvl):
    if not lvl.parent_node:
        return
    mnode = lvl.parent_node
    x = []
    y = []
    z = []
    for node in lvl.nodes_primary.registry.values():
        x.extend(
            (node.coords[0], mnode.coords[0], None)
        )
        y.extend(
            (node.coords[1], mnode.coords[1], None)
        )
        z.extend(
            (node.coords[2], mnode.coords[2], None)
        )
    dt.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "hoverinfo": "skip",
        "line": {
            "width": 4,
            "dash": "dash",
            "color": graphics_common.GRID_COLOR
        }
    })


def add_data__bisector_lines(dt, lvl):
    if not lvl.parent_node:
        return
    x = []
    y = []
    z = []
    for line in lvl.floor_bisector_lines:
        p1 = line[0]
        p2 = line[1]
        x.extend(
            (p1[0], p2[0], None)
        )
        y.extend(
            (p1[1], p2[1], None)
        )
        z.extend(
            (lvl.elevation, lvl.elevation, None)
        )
    dt.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "hoverinfo": "skip",
        "line": {
            "width": 4,
            "color": graphics_common.BISECTOR_COLOR
        }
    })


def add_data__frames(dt, mdl, load_case):
    beamcolumn_elems = mdl.list_of_beamcolumn_elements()
    if not beamcolumn_elems:
        return
    x = []
    y = []
    z = []
    customdata = []
    section_names = []
    for elm in beamcolumn_elems:
        if elm.visibility.hidden_at_line_plots:
            continue
        p_i = np.array(elm.eleNodes[0].coords) + elm.geomtransf.offset_i
        p_j = np.array(elm.eleNodes[1].coords) + elm.geomtransf.offset_j
        section_names.extend([elm.section.name]*3)
        x.extend(
            (p_i[0], p_j[0], None)
        )
        y.extend(
            (p_i[1], p_j[1], None)
        )
        z.extend(
            (p_i[2], p_j[2], None)
        )
        if load_case:
            customdata.append(
                (elm.uid,
                 *load_case.line_element_udl.registry[elm.uid].total(),
                 elm.eleNodes[0].uid,
                 elm.parent_component.uid)
            )
            customdata.append(
                (elm.uid,
                 *load_case.line_element_udl.registry[elm.uid].total(),
                 elm.eleNodes[1].uid,
                 elm.parent_component.uid)
            )
            customdata.append(
                [None]*6
            )
        else:
            customdata.append(
                (elm.uid,
                 elm.eleNodes[0].uid,
                 elm.parent_component.uid)
            )
            customdata.append(
                (elm.uid,
                 elm.eleNodes[1].uid,
                 elm.parent_component.uid)
            )
            customdata.append(
                [None]*3
            )

    if load_case:
        customdata = np.array(customdata, dtype='object')
        dt.append({
            "name": "Frame elements",
            "type": "scatter3d",
            "mode": "lines",
            "x": x,
            "y": y,
            "z": z,
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
        customdata = np.array(customdata, dtype='object')
        dt.append({
            "name": "Frame elements",
            "type": "scatter3d",
            "mode": "lines",
            "x": x,
            "y": y,
            "z": z,
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
        

def add_data__frame_offsets(dt, mdl):
    beamcolumn_elems = mdl.list_of_beamcolumn_elements()
    if not beamcolumn_elems:
        return

    x = []
    y = []
    z = []

    for elm in beamcolumn_elems:
        p_i = np.array(elm.eleNodes[0].coords)
        p_io = np.array(elm.eleNodes[0].coords) + elm.geomtransf.offset_i
        p_j = np.array(elm.eleNodes[1].coords)
        p_jo = np.array(elm.eleNodes[1].coords) + elm.geomtransf.offset_j

        x.extend((p_i[0], p_io[0], None))
        y.extend((p_i[1], p_io[1], None))
        z.extend((p_i[2], p_io[2], None))
        x.extend((p_j[0], p_jo[0], None))
        y.extend((p_j[1], p_jo[1], None))
        z.extend((p_j[2], p_jo[2], None))

    dt.append({
        "name": "Rigid offsets",
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "hoverinfo": "skip",
        "line": {
            "width": 8,
            "color": graphics_common.OFFSET_COLOR
        }
    })


def add_data__frame_axes(dt, mdl, ref_len):
    beamcolumn_elems = mdl.list_of_beamcolumn_elements()
    if not beamcolumn_elems:
        return
    s = ref_len * 0.025
    x = []
    y = []
    z = []
    colors = []
    for elm in beamcolumn_elems:
        if elm.visibility.hidden_at_line_plots:
            continue
        x_vec = elm.geomtransf.x_axis
        y_vec = elm.geomtransf.y_axis
        z_vec = elm.geomtransf.z_axis
        l_clear = elm.clear_length()
        i_pos = np.array(elm.eleNodes[0].coords) + elm.geomtransf.offset_i
        mid_pos = i_pos + x_vec * l_clear/2.00
        x.extend((mid_pos[0], mid_pos[0]+x_vec[0]*s, None))
        y.extend((mid_pos[1], mid_pos[1]+x_vec[1]*s, None))
        z.extend((mid_pos[2], mid_pos[2]+x_vec[2]*s, None))
        colors.extend(["red"]*3)
        x.extend((mid_pos[0], mid_pos[0]+y_vec[0]*s, None))
        y.extend((mid_pos[1], mid_pos[1]+y_vec[1]*s, None))
        z.extend((mid_pos[2], mid_pos[2]+y_vec[2]*s, None))
        colors.extend(["green"]*3)
        x.extend((mid_pos[0], mid_pos[0]+z_vec[0]*s, None))
        y.extend((mid_pos[1], mid_pos[1]+z_vec[1]*s, None))
        z.extend((mid_pos[2], mid_pos[2]+z_vec[2]*s, None))
        colors.extend(["blue"]*3)
    dt.append({
        "name": "Frame axes",
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "hoverinfo": "skip",
        "line": {
            "width": 8,
            "color": colors
        }
    })


def add_data__zerolength_axes(dt, mdl, ref_len):
    zerolength_elements = mdl.list_of_zerolength_elements()
    if not zerolength_elements:
        return
    s = ref_len * 0.025
    x = []
    y = []
    z = []
    colors = []
    for elm in zerolength_elements:
        x_vec = elm.vecx
        y_vec = elm.vecyp
        z_vec = np.cross(x_vec, y_vec)
        mid_pos = np.array(elm.eleNodes[0].coords)
        x.extend((mid_pos[0], mid_pos[0]+x_vec[0]*s, None))
        y.extend((mid_pos[1], mid_pos[1]+x_vec[1]*s, None))
        z.extend((mid_pos[2], mid_pos[2]+x_vec[2]*s, None))
        colors.extend(["red"]*3)
        x.extend((mid_pos[0], mid_pos[0]+y_vec[0]*s, None))
        y.extend((mid_pos[1], mid_pos[1]+y_vec[1]*s, None))
        z.extend((mid_pos[2], mid_pos[2]+y_vec[2]*s, None))
        colors.extend(["green"]*3)
        x.extend((mid_pos[0], mid_pos[0]+z_vec[0]*s, None))
        y.extend((mid_pos[1], mid_pos[1]+z_vec[1]*s, None))
        z.extend((mid_pos[2], mid_pos[2]+z_vec[2]*s, None))
        colors.extend(["blue"]*3)
    dt.append({
        "name": "Zerolength axes",
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "hoverinfo": "skip",
        "line": {
            "width": 8,
            "color": colors
        }
    })


def add_data__global_axes(dt, ref_len):

    s = ref_len
    x = []
    y = []
    z = []
    colors = []
    x_vec = np.array([1.00, 0.00, 0.00])
    y_vec = np.array([0.00, 1.00, 0.00])
    z_vec = np.array([0.00, 0.00, 1.00])

    x.extend((0.00, x_vec[0]*s, None))
    y.extend((0.00, x_vec[1]*s, None))
    z.extend((0.00, x_vec[2]*s, None))
    x.extend((0.00, y_vec[0]*s, None))
    y.extend((0.00, y_vec[1]*s, None))
    z.extend((0.00, y_vec[2]*s, None))
    x.extend((0.00, z_vec[0]*s, None))
    y.extend((0.00, z_vec[1]*s, None))
    z.extend((0.00, z_vec[2]*s, None))
    colors.extend(["red"]*3)
    colors.extend(["green"]*3)
    colors.extend(["blue"]*3)
    dt.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "hoverinfo": "skip",
        "line": {
            "width": 3,
            "color": colors
        }
    })


def add_data__extruded_frames_mesh(dt, mdl):
    beamcolumn_elems = mdl.list_of_beamcolumn_elements()
    if not beamcolumn_elems:
        return
    x_list = []
    y_list = []
    z_list = []
    i_list = []
    j_list = []
    k_list = []
    index = 0
    for elm in beamcolumn_elems:
        if elm.visibility.hidden_when_extruded:
            continue
        side_a = np.array(elm.eleNodes[0].coords) + elm.geomtransf.offset_i
        side_b = np.array(elm.eleNodes[1].coords) + elm.geomtransf.offset_j
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
    dt.append({
        "type": "mesh3d",
        "x": x_list,
        "y": y_list,
        "z": z_list,
        "i": i_list,
        "j": j_list,
        "k": k_list,
        "hoverinfo": "skip",
        "color": graphics_common.BEAM_MESH_COLOR,
        "opacity": 0.65
    })


def show(mdl: Model,
         load_case: LoadCase=None,
         extrude=False,
         offsets=True,
         global_axes=True,
         diaphragm_lines=True,
         tributary_areas=True,
         parent_nodes=True,
         frame_axes=False,
         zerolength_axes=False,
         camera=None
         ):

    layout = graphics_common_3D.global_layout(camera)
    dt: list[dict] = []

    ref_len = mdl.reference_length()

    # plot the nodes
    
    add_data__nodes(dt, mdl, load_case)
    if not extrude:
        add_data__internal_nodes(dt, mdl, load_case)

    # # global axes
    # if global_axes:
    #     add_data__global_axes(dt, ref_len)

    # # diaphgragm lines
    # if diaphragm_lines:
    #     for lvl in mdl.levels.registry.values():
    #         add_data__diaphragm_lines(dt, lvl)

    # # bisector lines
    # if tributary_areas:
    #     for lvl in mdl.levels.registry.values():
    #         add_data__bisector_lines(dt, lvl)

    # plot beamcolumn elements

    if extrude:
        add_data__extruded_frames_mesh(
            dt, mdl)
        # add_data__extruded_steel_W_PZ_mesh(
        #     dt, list_of_steel_W_panel_zones)
        pass
    else:
        add_data__frames(dt, mdl, load_case)
        if frame_axes:
            add_data__frame_axes(dt, mdl, ref_len)
        if zerolength_axes:
            add_data__zerolength_axes(dt, mdl, ref_len)
    # plot the rigid offsets
    if offsets:
        add_data__frame_offsets(dt, mdl)

    # # plot the parent nodes
    # if parent_nodes:
    #     add_data__parent_nodes(dt, mdl.list_of_parent_nodes())

    fig_datastructure = dict(data=dt, layout=layout)
    fig = go.Figure(fig_datastructure)

    if not "pytest" in sys.modules:
        fig.show()
