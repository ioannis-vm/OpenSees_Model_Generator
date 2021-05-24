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

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from modeler import Building

# https://www.google.com/search?q=color+picker
GRID_COLOR = '#d1d1d1'
NODE_PRIMARY_COLOR = '#7ac4b7'
COLUMN_COLOR = '#0f24db'
BEAM_COLOR = '#0f24db'
FLOOR_COLOR = '#c4994f'


def draw_level_geometry(building: Building, lvlname: str):
    """
    Shows the grids, beams, floor perimeters and UDLs
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    # retrieved the specified level
    level = building.levels.get(lvlname)

    # draw the floor perimeter
    xy = np.array(level.perimeter.points)
    polygon = Polygon(xy, True)
    patches = [polygon]
    collection = PatchCollection(
        patches, alpha=0.25, facecolors=FLOOR_COLOR, edgecolors=("black",))
    ax.add_collection(collection)

    ax.margins(0.10)
    fig.show()

    # draw the beams
