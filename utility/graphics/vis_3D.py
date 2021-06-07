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
import plotly.io as pio
import numpy as np
from modeler import Building

pio.renderers.default = 'browser'

# https://www.google.com/search?q=color+picker
GRID_COLOR = '#d1d1d1'
NODE_PRIMARY_COLOR = '#7ac4b7'
COLUMN_COLOR = '#0f24db'
BEAM_COLOR = '#0f24db'
BEAM_MESH_COLOR = '#6d7aed'


def global_layout():
    """
    Some general definitions needed often
    """
    return go.Layout(
        scene=dict(aspectmode='data',
                   xaxis_visible=False,
                   yaxis_visible=False,
                   zaxis_visible=False,
                   ),
        showlegend=False
    )


def level_geometry(building: Building, lvlname: str, extrude_frames=False):

    level = building.levels.get(lvlname)

    # draw the grids
    dt = []
    for grid in building.gridsystem.grids:
        dt.append({
            "type": "scatter3d",
            "mode": "lines",
            "x": [grid.start[0], grid.end[0]],
            "y": [grid.start[1], grid.end[1]],
            "z": [level.elevation]*2,
            "hoverinfo": "skip",
            "line": {
                "width": 2,
                "color": GRID_COLOR
            }
        })

    mark = {}
    mark['fixed'] = ("square", 10)
    mark['pinned'] = ("circle-open", 10)
    mark['free'] = ("circle", 5)

    # draw the nodes
    dt.append({
        "type": "scatter3d",
        "mode": "markers",
        "x": [node.coordinates[0] for node in level.nodes.node_list],
        "y": [node.coordinates[1] for node in level.nodes.node_list],
        "z": [node.coordinates[2] for node in level.nodes.node_list],
        # "hoverinfo": "text",
        # "hovertext": ["node" + str(node.uniq_id)
        #               for node in level.nodes.node_list],
        "marker": {
            "symbol": [mark[node.restraint_type][0]
                       for node in level.nodes.node_list],
            "color": NODE_PRIMARY_COLOR,
            "size": [mark[node.restraint_type][1]
                     for node in level.nodes.node_list]
        }
    })

    # draw the center of mass
    if level.slab_data:
        dt.append({
            "type": "scatter3d",
            "mode": "markers",
            "x": [level.slab_data['properties']['centroid'][0]],
            "y": [level.slab_data['properties']['centroid'][1]],
            "z": [level.elevation],
            "hoverinfo": "text",
            "hovertext": ["node" + str(node.uniq_id)
                          for node in level.nodes.node_list],
            "marker": {
                "symbol": 'circle-open',
                "color": NODE_PRIMARY_COLOR,
                "size": 10
            }
        })

    # draw the columns and beams
    if extrude_frames:
        x_list = []
        y_list = []
        z_list = []
        i_list = []
        j_list = []
        k_list = []
        index = 0
        for elm in level.beams.beam_list+level.columns.column_list:
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
        if x_list:
            dt.append({
                "type": "mesh3d",
                "x": x_list,
                "y": y_list,
                "z": z_list,
                "i": i_list,
                "j": j_list,
                "k": k_list,
                "hoverinfo": "none",
                "color": BEAM_MESH_COLOR,
                "opacity": 0.65
            })
    else:
        for col in level.columns.column_list:
            dt.append({
                "type": "scatter3d",
                "mode": "lines",
                "x": [col.node_i.coordinates[0], col.node_j.coordinates[0]],
                "y": [col.node_i.coordinates[1], col.node_j.coordinates[1]],
                "z": [level.elevation, level.previous_lvl.elevation],
                "hoverinfo": "text",
                "hovertext": "column " + str(col.uniq_id),
                "line": {
                    "width": 5,
                    "color": COLUMN_COLOR
                }
            })
        for beam in level.beams.beam_list:
            dt.append({
                "type": "scatter3d",
                "mode": "lines",
                "x": [beam.node_i.coordinates[0], beam.node_j.coordinates[0]],
                "y": [beam.node_i.coordinates[1], beam.node_j.coordinates[1]],
                "z": [level.elevation, level.elevation],
                "hoverinfo": "text",
                "hovertext": "beam " + str(beam.uniq_id),
                "line": {
                    "width": 5,
                    "color": BEAM_COLOR
                }
            })

    layout = global_layout()
    fig_datastructure = {
        "data": dt,
        "layout": layout
    }

    return fig_datastructure


def draw_level_geometry(building: Building, lvlname: str,
                        extrude_frames=False):

    fig_datastructure = level_geometry(building, lvlname, extrude_frames)
    fig = go.Figure(fig_datastructure)

    fig.show()


def draw_building_geometry(building: Building, extrude_frames=False):
    layout = global_layout()
    dt = []
    for lvl in building.levels.level_list:
        dt.append(
            level_geometry(building, lvl.name, extrude_frames)["data"]
        )

    def dt_flat(dt): return [item for sublist in dt for item in sublist]
    fig_datastructure = dict(data=dt_flat(dt), layout=layout)
    fig = go.Figure(fig_datastructure)
    fig.show()
