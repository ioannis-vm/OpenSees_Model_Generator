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
# https://github.com/ioannis-vm/OpenSeesPy_Building_Modeler/blob/main/modeler.py

from modeler import Building
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

pio.renderers.default = 'browser'

# https://www.google.com/search?q=color+picker
GRID_COLOR = '#d1d1d1'
NODE_PRIMARY_COLOR = '#7ac4b7'
COLUMN_COLOR = '#0f24db'
BEAM_COLOR = '#0f24db'


def global_layout():
    return go.Layout(
        scene=dict(aspectmode='data',
                   aspectratio=dict(x=1, y=1, z=1),  # TODO
                   xaxis_visible=False,
                   yaxis_visible=False,
                   zaxis_visible=False,
                   ),
        showlegend=False
    )


def level_geometry(building: Building, lvlname: str):

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

    if level.restraint == "fixed":
        mark = "square"
        size = 10
    elif level.restraint == "pinned":
        mark = "circle-open"
        size = 10
    else:
        mark = "circle"
        size = 5

    # draw the nodes
    dt.append({
        "type": "scatter3d",
        "mode": "markers",
        "x": [node.x_coord for node in level.nodes.node_list],
        "y": [node.y_coord for node in level.nodes.node_list],
        "z": [level.elevation]*len(level.nodes.node_list),
        "hoverinfo": "text",
        "hovertext": "lala",
        "marker": {
            "symbol": mark,
            "color": NODE_PRIMARY_COLOR,
            "size": size
        }
    })

    # ["node " + str(node.uniq_id) for node in level.nodes.node_list]

    # draw the columns
    for col in level.columns.column_list:
        dt.append({
            "type": "scatter3d",
            "mode": "lines",
            "x": [col.node_i.x_coord, col.node_j.x_coord],
            "y": [col.node_i.y_coord, col.node_j.y_coord],
            "z": [level.elevation, level.previous_lvl.elevation],
            "hoverinfo": "text",
            "hovertext": "column " + str(col.uniq_id),
            "line": {
                "width": 5,
                "color": COLUMN_COLOR
            }
        })

    # draw the beams
    for beam in level.beams.beam_list:
        dt.append({
            "type": "scatter3d",
            "mode": "lines",
            "x": [beam.node_i.x_coord, beam.node_j.x_coord],
            "y": [beam.node_i.y_coord, beam.node_j.y_coord],
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


def draw_level_geometry(building: Building, lvlname: str):

    fig_datastructure = level_geometry(building, lvlname)
    fig = go.Figure(fig_datastructure)

    fig.show()


def draw_building_geometry(building: Building):
    layout = global_layout()
    dt = []
    for lvl in building.levels.level_list:
        dt.append(
            level_geometry(building, lvl.name)["data"]
        )

    def dt_flat(dt): return [item for sublist in dt for item in sublist]
    fig_datastructure = dict(data=dt_flat(dt), layout=layout)
    fig = go.Figure(fig_datastructure)
    fig.show()
