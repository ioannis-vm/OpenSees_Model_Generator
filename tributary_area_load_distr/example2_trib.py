import matplotlib.pyplot as plt
import numpy as np
from skgeom.draw import draw
import skgeom as sg
from tributary_area_load_distr.mesher import Vertex, Edge, define_halfedges
import pandas as pd
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def accumulate_areas(areas, index, value):
    """
    Using a dictionary (`areas`) to accumulate the areas
    associated with each vertex.
    `index` corresponds to the unique id of the vertex.
    `value` is the value to be added at the givn index.
    """
    try:
        # assuming the key already exists
        areas[index] = areas[index] + value
    except KeyError:
        # create it if it doesn't already exist
        areas[index] = value


def is_in_some_miniloop(halfedge, loops):
    for loop in loops:
        for other_halfedge in loop:
            if (other_halfedge.vertex.point == halfedge.vertex.point and
                    other_halfedge.next.vertex.point == halfedge.next.vertex.point):
                return True
    return False


def area_from_vertices(vertices):
    return sg.Polygon(vertices).area()


# define the vertices
vertices = [
    Vertex((0., 0.)),
    Vertex((1., 0.)),
    Vertex((2., 0.)),
    Vertex((0., 1.)),
    Vertex((1., 1.)),
    Vertex((0., 2.)),
    Vertex((1., 2.)),
    Vertex((2., 2.)),
]


# define the edges
edges = [
    Edge(vertices[0], vertices[1]),
    Edge(vertices[1], vertices[2]),
    Edge(vertices[0], vertices[3]),
    Edge(vertices[0], vertices[4]),
    Edge(vertices[1], vertices[4]),
    Edge(vertices[2], vertices[7]),
    Edge(vertices[3], vertices[4]),
    Edge(vertices[3], vertices[5]),
    Edge(vertices[4], vertices[5]),
    Edge(vertices[4], vertices[6]),
    Edge(vertices[4], vertices[7]),
    Edge(vertices[5], vertices[6]),
    Edge(vertices[6], vertices[7]),
]


# # # # # # # # # # #

define_halfedges(edges)

halfedges = []
for edge in edges:
    if edge.h_i not in halfedges:
        halfedges.append(edge.h_i)
    if edge.h_j not in halfedges:
        halfedges.append(edge.h_j)

results = {
    'halfedge': [],
    'vertex': [],
    'edge': [],
    'next': [],
}

for h in halfedges:
    results['halfedge'].append(h)
    results['vertex'].append(h.vertex)
    results['edge'].append(h.edge)
    results['next'].append(h.nxt)


df = pd.DataFrame(results)
print(df)


def is_in_some_loop(halfedge, loops):
    for loop in loops:
        if halfedge in loop:
            return True
    return False


loops = []
for halfedge in halfedges:
    if loops:
        if is_in_some_loop(halfedge, loops):
            continue
    loop = [halfedge]
    nxt = halfedge.nxt
    while(nxt != halfedge):
        loop.append(nxt)
        nxt = nxt.nxt
    loops.append(loop)

print('Formed the following loops:')
for loop in loops:
    print(loop)


# remove the largest loop (that corresponds to the exterior halfedges)
loop_areas = [area_from_vertices([h.vertex.coords
                                  for h in loop])
              for loop in loops]
outer = min(loop_areas)
index = loop_areas.index(outer)
del loops[index]
del loop_areas[index]


def draw_skeleton(fig, polygon, skeleton, show_time=False):

    coord_array = poly.coords
    coord_array = np.vstack((coord_array, coord_array[0, :]))

    plt.plot(coord_array[:, 0], coord_array[:, 1], 'black', lw=2.0)

    for h in skeleton.halfedges:
        if h.is_bisector:
            p1 = h.vertex.point
            p2 = h.opposite.vertex.point
            plt.plot([p1.x(), p2.x()], [p1.y(), p2.y()], 'r-', lw=1.5)

    if show_time:
        for v in skeleton.vertices:
            plt.gcf().gca().add_artist(plt.Circle(
                (v.point.x(), v.point.y()),
                v.time, color='grey', fill=False, lw=0.20))


# fig = plt.figure()

# for loop in loops:
#     poly = sg.Polygon([h.vertex.coords for h in loop])
#     skel = sg.skeleton.create_interior_straight_skeleton(poly)
#     draw_skeleton(fig, poly, skel, show_time=True)
# fig.show()


# accumulate area part
areas = {}

for loop in loops:

    poly = sg.Polygon([h.vertex.coords for h in loop])
    skel = sg.skeleton.create_interior_straight_skeleton(poly)

    miniloops = []
    for halfedge in skel.halfedges:
        if miniloops:
            if is_in_some_miniloop(halfedge, miniloops):
                continue
        miniloop = [halfedge]
        nxt = halfedge.next
        while(nxt.vertex.point != halfedge.vertex.point):
            miniloop.append(nxt)
            nxt = nxt.next
        miniloops.append(miniloop)

    miniloop_areas = [area_from_vertices(
        [h.vertex.point for h in miniloop]) for miniloop in miniloops]
    outer = min(miniloop_areas)
    index = miniloop_areas.index(outer)
    del miniloops[index]
    del miniloop_areas[index]

    for i, miniloop in enumerate(miniloops):
        area = miniloop_areas[i]
        loop_edges = [h.edge for h in loop]
        for halfedge in miniloop:
            for edge in loop_edges:
                v_i = sg.Point2(*edge.v_i.coords)
                v_j = sg.Point2(*edge.v_j.coords)
                pt_1 = halfedge.vertex.point
                pt_2 = halfedge.next.vertex.point
                if ((pt_1 == v_i and pt_2 == v_j) or (pt_1 == v_j and pt_2 == v_i)):
                    accumulate_areas(areas, edge.nid, area)


for k, v in areas.items():
    print(k, v)
