"""
The following utility functions are used for data visualization
https://plotly.com/python/reference/
"""
#   __                 UC eley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   April 2021
#
# https://github.com/ioannis-vm/OpenSeesPy_Building_Modeler/

import plotly.graph_objects as go
import numpy as np
from utility.graphics import common, common_3D


def add_data__grids(dt, building):
    for lvl in building.levels.level_list:
        elevation = lvl.elevation
        for grid in building.gridsystem.grids:
            dt.append({
                "type": "scatter3d",
                "mode": "lines",
                "x": [grid.start[0], grid.end[0]],
                "y": [grid.start[1], grid.end[1]],
                "z": [elevation]*2,
                "hoverinfo": "skip",
                "line": {
                    "width": 4,
                    "color": common.GRID_COLOR
                }
            })


def add_data__nodes(dt, list_of_nodes):
    x = [node.coordinates[0] for node in list_of_nodes]
    y = [node.coordinates[1] for node in list_of_nodes]
    z = [node.coordinates[2] for node in list_of_nodes]
    customdata = []
    restraint_types = [node.restraint_type for node in list_of_nodes]
    for node in list_of_nodes:
        customdata.append(
            (node.uniq_id,
             *node.mass.value,
             *node.load.value
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
        '%{customdata[2]:.3g}, %{customdata[3]:.3g})<br>' +
        'Load: (%{customdata[4]:.3g}, ' +
        '%{customdata[5]:.3g}, %{customdata[6]:.3g})' +
        '<extra>Node: %{customdata[0]:d}</extra>',
        "marker": {
            "symbol": [common_3D.node_marker[node.restraint_type][0]
                       for node in list_of_nodes],
            "color": common.NODE_PRIMARY_COLOR,
            "size": [common_3D.node_marker[node.restraint_type][1]
                     for node in list_of_nodes],
            "line": {
                "color": common.NODE_PRIMARY_COLOR,
                "width": 4}
        }
    })


def add_data__master_nodes(dt, list_of_nodes):

    x = [node.coordinates[0] for node in list_of_nodes]
    y = [node.coordinates[1] for node in list_of_nodes]
    z = [node.coordinates[2] for node in list_of_nodes]
    customdata = []
    restraint_types = [node.restraint_type for node in list_of_nodes]
    for node in list_of_nodes:
        customdata.append(
            (node.uniq_id,
             *node.mass.value,
             *node.load.value
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
        '<extra>Master Node: %{customdata[0]:d}</extra>',
        "marker": {
            "symbol": [common_3D.node_marker[node.restraint_type][0]
                       for node in list_of_nodes],
            "color": common.NODE_PRIMARY_COLOR,
            "size": [common_3D.node_marker[node.restraint_type][1]
                     for node in list_of_nodes],
            "line": {
                "color": common.NODE_PRIMARY_COLOR,
                "width": 4}
        }
    })


def add_data__internal_nodes(dt, list_of_nodes):
    x = [node.coordinates[0] for node in list_of_nodes]
    y = [node.coordinates[1] for node in list_of_nodes]
    z = [node.coordinates[2] for node in list_of_nodes]
    customdata = []
    restraint_types = [node.restraint_type for node in list_of_nodes]
    for node in list_of_nodes:
        customdata.append(
            (node.uniq_id,
             *node.mass.value,
             *node.load.value
             )
        )
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
        '%{customdata[2]:.3g}, %{customdata[3]:.3g})<br>' +
        'Load: (%{customdata[4]:.3g}, ' +
        '%{customdata[5]:.3g}, %{customdata[6]:.3g})' +
        '<extra>Node: %{customdata[0]:d}</extra>',
        "marker": {
            "symbol": common_3D.node_marker['internal'][0],
            "color": common.NODE_INTERNAL_COLOR,
            "size": common_3D.node_marker['internal'][1],
            "line": {
                "color": common.NODE_INTERNAL_COLOR,
                "width": 2}
        }
    })


def add_data__diaphragm_lines(dt, lvl):
    if not lvl.master_node:
        return
    mnode = lvl.master_node
    x = []
    y = []
    z = []
    for node in lvl.nodes_primary.node_list:
        x.extend(
            (node.coordinates[0], mnode.coordinates[0], None)
        )
        y.extend(
            (node.coordinates[1], mnode.coordinates[1], None)
        )
        z.extend(
            (node.coordinates[2], mnode.coordinates[2], None)
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
            "color": common.GRID_COLOR
        }
    })


def add_data__frames(dt, list_of_frames):
    x = []
    y = []
    z = []
    customdata = []
    section_names = []
    for elm in list_of_frames:
        section_names.extend([elm.section.name]*3)
        x.extend(
            (elm.node_i.coordinates[0], elm.node_j.coordinates[0], None)
        )
        y.extend(
            (elm.node_i.coordinates[1], elm.node_j.coordinates[1], None)
        )
        z.extend(
            (elm.node_i.coordinates[2], elm.node_j.coordinates[2], None)
        )
        customdata.append(
            (elm.uniq_id,
             *elm.udl.value,
             elm.node_i.uniq_id)
        )
        customdata.append(
            (elm.uniq_id,
             *elm.udl.value,
             elm.node_j.uniq_id)
        )
        customdata.append(
            [None]*6
        )

    customdata = np.array(customdata, dtype='object')
    dt.append({
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
        'Node @ this end: %{customdata[4]:d}</extra>',
        "line": {
            "width": 5,
            "color": common.FRAME_COLOR
        }
    })


def add_data__frame_axes(dt, list_of_frames, ref_len):
    if not list_of_frames:
        return
    s = ref_len * 0.025
    x = []
    y = []
    z = []
    colors = []
    for elm in list_of_frames:
        x_vec = elm.local_x_axis_vector()
        y_vec = elm.local_y_axis_vector()
        z_vec = elm.local_z_axis_vector()
        l = elm.length()
        i_pos = np.array(elm.node_i.coordinates)
        mid_pos = i_pos + x_vec * l/2.00
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


def add_data__extruded_frames_mesh(dt, list_of_frames):
    if not list_of_frames:
        return
    x_list = []
    y_list = []
    z_list = []
    i_list = []
    j_list = []
    k_list = []
    index = 0
    for elm in list_of_frames:
        side_a = np.array(elm.node_i.coordinates)
        side_b = np.array(elm.node_j.coordinates)
        y_vec = elm.local_y_axis_vector()
        z_vec = elm.local_z_axis_vector()
        loop = elm.section.mesh.halfedges
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
        "color": common.BEAM_MESH_COLOR,
        "opacity": 0.65
    })


def plot_building_geometry(building: 'Building', extrude_frames=False):

    layout = common_3D.global_layout()
    dt = []

    # plot the grids
    add_data__grids(dt, building)

    # plot the diaphgragm-lines
    for lvl in building.levels.level_list:
        add_data__diaphragm_lines(dt, lvl)

    # plot the internal nodes
    if not extrude_frames:
        add_data__internal_nodes(dt, building.list_of_internal_nodes())

    # plot the nodes
    add_data__nodes(dt, building.list_of_primary_nodes())

    # plot the master nodes
    add_data__master_nodes(dt, building.list_of_master_nodes())

    # plot the columns and beams (if any)
    if extrude_frames:
        add_data__extruded_frames_mesh(
            dt, building.list_of_internal_elems())
    else:
        add_data__frames(dt, building.list_of_internal_elems())
        add_data__frame_axes(dt, building.list_of_internal_elems(),
                             building.reference_length())

    fig_datastructure = dict(data=dt, layout=layout)
    fig = go.Figure(fig_datastructure)
    fig.show()
