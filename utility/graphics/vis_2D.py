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
import skgeom as sg
from modeler import Building

# https://www.google.com/search?q=color+picker
GRID_COLOR = '#d1d1d1'
NODE_PRIMARY_COLOR = '#7ac4b7'
COLUMN_COLOR = '#0f24db'
BEAM_COLOR = '#0f24db'
FLOOR_COLOR = '#c4994f'


def draw_level_geometry(building: Building, lvlname: str, extrude_frames=False):
    """
    TODO Shows the grids, beams, floor perimeters, tributary areas
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    # retrieve the specified level
    level = building.levels.get(lvlname)

    # draw the gridlines
    for grd in building.gridsystem.grids:
        line = [
            grd.start,
            grd.end
        ]
        line_np = np.array(line)
        ax.plot(line_np[:, 0], line_np[:, 1], color=GRID_COLOR)

    # draw the floor slabs and tributary areas
    for loop in level.slab_data['loops']:
        coords = [h.vertex.coords for h in loop]
        poly = sg.Polygon(coords)
        skel = sg.skeleton.create_interior_straight_skeleton(poly)
        for h in skel.halfedges:
            if h.is_bisector:
                p1 = h.vertex.point
                p2 = h.opposite.vertex.point
                ax.plot([p1.x(), p2.x()], [p1.y(), p2.y()], 'r-', lw=1)
        plot_polygon = Polygon(coords, True)
        patches = [plot_polygon]
        collection = PatchCollection(patches,
                                     alpha=0.25,
                                     facecolors=FLOOR_COLOR,
                                     edgecolors=FLOOR_COLOR)
        ax.add_collection(collection)

    # draw floor center of mass
    centroid = level.slab_data['properties']['centroid']
    ax.scatter(centroid[0], centroid[1], s=150,
               facecolors='none', edgecolors='black', lw=2)

    # draw the beam column elements
    if extrude_frames:
        patches = []
        for bm in level.beams.beam_list:
            bbox = bm.section.mesh.bounding_box()
            sec_b = bbox[1, 0] - bbox[0, 0]
            p0 = bm.node_i.coordinates[0:2] + \
                bm.local_y_axis_vector()[0:2]*sec_b/2
            p1 = bm.node_j.coordinates[0:2] + \
                bm.local_y_axis_vector()[0:2]*sec_b/2
            p2 = bm.node_j.coordinates[0:2] - \
                bm.local_y_axis_vector()[0:2]*sec_b/2
            p3 = bm.node_i.coordinates[0:2] - \
                bm.local_y_axis_vector()[0:2]*sec_b/2
            coords = np.vstack((p0, p1, p2, p3))
            patches.append(Polygon(coords, True))
        for col in level.columns.column_list:
            coords = np.array(
                [h.vertex.coords for h in col.section.mesh.halfedges])
            coords += col.node_i.coordinates[0:2]
            patches.append(Polygon(coords, True))
        collection = PatchCollection(
            patches, facecolors=BEAM_COLOR,
            edgecolors=BEAM_COLOR, alpha=0.35)
        ax.add_collection(collection)

    else:
        for bm in level.beams.beam_list:
            line = [
                bm.node_i.coordinates[0:2],
                bm.node_j.coordinates[0:2]
            ]
            line_np = np.array(line)
            ax.plot(line_np[:, 0], line_np[:, 1], color=BEAM_COLOR)

    # draw the nodes
    points = []
    if level.nodes.node_list:
        for nd in level.nodes.node_list:
            points.append(nd.coordinates)
        points_np = np.array(points)
        ax.scatter(points_np[:, 0], points_np[:, 1], color=NODE_PRIMARY_COLOR)

    ax.margins(0.10)
    fig.show()
