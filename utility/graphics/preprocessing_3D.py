"""
The following utility functions are used for data visualization
https://plotly.com/python/reference/
"""
#   __                 UC Berkeley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   April 2021
#
# https://github.com/ioannis-vm/OpenSeesPy_Building_Modeler/

import plotly.graph_objects as go
import numpy as np
from utility.graphics import common, common_3D


def add_data__grids(dt, list_of_grids, elevation):
    for grid in list_of_grids:
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


def add_data__master_node(dt, list_of_nodes):

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


def add_data__diaphragm_lines(dt, lvl):
    if not lvl.master_node:
        return
    mnode = lvl.master_node
    x = []
    y = []
    z = []
    for node in lvl.nodes.node_list:
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
        section_names.extend([elm.section.name]*2)
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
            "color": common.COLUMN_COLOR
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
        "hoverinfo": "none",
        "color": common.BEAM_MESH_COLOR,
        "opacity": 0.65
    })


def level_geometry(building: 'Building', lvlname: str, extrude_frames=False):

    level = building.levels.get(lvlname)

    dt = []

    # draw the grids
    add_data__grids(dt, building.gridsystem.grids, level.elevation)

    # draw the diaphgragm-lines
    for lvl in building.levels.level_list:
        add_data__diaphragm_lines(dt, lvl)

    # draw the nodes
    add_data__nodes(dt, level.nodes.node_list)

    # draw the master nodes
    if level.slab_data:
        add_data__master_node(dt, [level.master_node])

    # draw the columns and beams (if any)
    list_of_frame_elems = level.beams.beam_list + level.columns.column_list
    if extrude_frames:
        add_data__extruded_frames_mesh(dt, list_of_frame_elems)
    else:
        add_data__frames(dt, list_of_frame_elems)

    return dt


def draw_level_geometry(building: 'Building', lvlname: str,
                        extrude_frames=False):

    dt = level_geometry(building, lvlname, extrude_frames)
    layout = common_3D.global_layout()
    fig_datastructure = dict(data=dt, layout=layout)
    fig = go.Figure(fig_datastructure)

    fig.show()


def draw_building_geometry(building: 'Building', extrude_frames=False):
    layout = common_3D.global_layout()
    dt = []
    for lvl in building.levels.level_list:
        dt.append(
            level_geometry(building, lvl.name, extrude_frames)
        )

    def dt_flat(dt): return [item for sublist in dt for item in sublist]
    fig_datastructure = dict(data=dt_flat(dt), layout=layout)
    fig = go.Figure(fig_datastructure)
    fig.show()
