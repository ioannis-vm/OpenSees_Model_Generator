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
                "width": 2,
                "color": common.GRID_COLOR
            }
        })


def add_data__nodes(dt, list_of_nodes):
    dt.append({
        "type": "scatter3d",
        "mode": "markers",
        "x": [node.coordinates[0] for node in list_of_nodes],
        "y": [node.coordinates[1] for node in list_of_nodes],
        "z": [node.coordinates[2] for node in list_of_nodes],
        # "hoverinfo": "text",
        # "hovertext": ["node" + str(node.uniq_id)
        #               for node in level.nodes.node_list],
        "marker": {
            "symbol": [common_3D.node_marker[node.restraint_type][0]
                       for node in list_of_nodes],
            "color": common.NODE_PRIMARY_COLOR,
            "size": [common_3D.node_marker[node.restraint_type][1]
                     for node in list_of_nodes]
        }
    })


def add_data__master_node(dt, master_node):
    dt.append({
        "type": "scatter3d",
        "mode": "markers",
        "x": [master_node.coordinates[0]],
        "y": [master_node.coordinates[1]],
        "z": [master_node.coordinates[2]],
        "hoverinfo": "text",
        # "hovertext": ["node" + str(node.uniq_id)
        #               for node in level.nodes.node_list],
        "marker": {
            "symbol": 'circle-open',
            "color": common.NODE_PRIMARY_COLOR,
            "size": 8
        }
    })


def add_data__frames(dt, list_of_frames):
    for elm in list_of_frames:
        dt.append({
            "type": "scatter3d",
            "mode": "lines",
            "x": [elm.node_i.coordinates[0], elm.node_j.coordinates[0]],
            "y": [elm.node_i.coordinates[1], elm.node_j.coordinates[1]],
            "z": [elm.node_i.coordinates[2], elm.node_j.coordinates[2]],
            "hoverinfo": "text",
            "hovertext": "column " + str(elm.uniq_id),
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
            loc0 = halfedge.vertex.coords[0]*z_vec + \
                halfedge.vertex.coords[1]*y_vec + side_a
            loc1 = halfedge.vertex.coords[0]*z_vec + \
                halfedge.vertex.coords[1]*y_vec + side_b
            loc2 = halfedge.nxt.vertex.coords[0]*z_vec + \
                halfedge.nxt.vertex.coords[1]*y_vec + side_b
            loc3 = halfedge.nxt.vertex.coords[0]*z_vec + \
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

    # draw the nodes
    add_data__nodes(dt, level.nodes.node_list)

    # draw the center of mass
    if level.slab_data:
        add_data__master_node(dt, level.master_node)

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
