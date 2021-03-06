"""The following utility functions are used for data visualization
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

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from utility.graphics import common


def plot_2D_level_geometry(building: 'Model',
                           lvlname: str,
                           extrude_frames=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    # retrieve the specified level
    level = building.levels.retrieve_by_name(lvlname)

    # draw the gridlines
    for grd in building.gridsystem.grids:
        line = [
            grd.start,
            grd.end
        ]
        line_np = np.array(line)
        ax.plot(line_np[:, 0], line_np[:, 1], color=common.GRID_COLOR)

    # draw the floor slabs and tributary areas
    if level.parent_node:
        coords = level.floor_coordinates
        for bisector in level.floor_bisector_lines:
            ax.plot(bisector[:, 0], bisector[:, 1], 'r-', lw=1)

        plot_polygon = Polygon(coords, True)
        patches = [plot_polygon]
        collection = PatchCollection(patches,
                                     alpha=0.25,
                                     facecolors=common.FLOOR_COLOR,
                                     edgecolors=common.FLOOR_COLOR)
        ax.add_collection(collection)

        # draw floor center of mass
        centroid = level.parent_node.coords
        ax.scatter(centroid[0], centroid[1], s=150,
                   facecolors='none', edgecolors='black', lw=2)

    # draw the beam column elements
    if extrude_frames:
        patches = []
        for bm in level.beams.registry.values():
            for elm in bm.internal_elems():
                bbox = elm.section.mesh.bounding_box()
                sec_b = bbox[1, 0] - bbox[0, 0]
                p0 = elm.internal_pt_i[0:2] + \
                    elm.z_axis[0:2]*sec_b/2
                p1 = elm.internal_pt_j[0:2] + \
                    elm.z_axis[0:2]*sec_b/2
                p2 = elm.internal_pt_j[0:2] - \
                    elm.z_axis[0:2]*sec_b/2
                p3 = elm.internal_pt_i[0:2] - \
                    elm.z_axis[0:2]*sec_b/2
                coords = np.vstack((p0, p1, p2, p3))
                patches.append(Polygon(coords, True))
        for col in level.columns.registry.values():
            coords = np.array(
                [h.vertex.coords for h in col.section.mesh.halfedges])
            ang = col.ang
            rot_mat = np.array([
                [np.cos(ang), -np.sin(ang)],
                [np.sin(ang), np.cos(ang)]
            ])
            coords = (rot_mat @ coords.T).T
            coords += col.internal_elems[0].internal_pt_i[0:2]
            patches.append(Polygon(coords, True))
        collection = PatchCollection(
            patches, facecolors=common.FRAME_COLOR,
            edgecolors=common.FRAME_COLOR, alpha=0.35)
        ax.add_collection(collection)

    else:
        for bm in level.beams.registry.values():
            for elm in bm.internal_elems:
                line = [
                    elm.internal_pt_i[0:2],
                    elm.internal_pt_j[0:2]
                ]
                line_np = np.array(line)
                ax.plot(line_np[:, 0], line_np[:, 1], color=common.FRAME_COLOR)

    # draw the nodes
    points = []
    if level.nodes_primary.registry.values():
        for nd in level.nodes_primary.registry.values():
            points.append(nd.coords)
        points_np = np.array(points)
        ax.scatter(points_np[:, 0], points_np[:, 1],
                   color=common.NODE_PRIMARY_COLOR)

    ax.margins(0.10)
    fig.show()
