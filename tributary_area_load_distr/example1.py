from tributary_area_load_distr.mesher import Vertex, Edge, traverse
import pandas as pd
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# define the vertices
vertices = [
    Vertex((5.00, 0.00)),
    Vertex((0.00, 5.00)),
    Vertex((5.00, 5.00)),
    Vertex((0.00, 10.00)),
    Vertex((5.00, 10.00)),
]


# define the edges
edges = [
    Edge(vertices[0], vertices[1]),
    Edge(vertices[0], vertices[2]),
    Edge(vertices[1], vertices[2]),
    Edge(vertices[1], vertices[3]),
    Edge(vertices[2], vertices[4]),
    Edge(vertices[3], vertices[4]),
]


e_start = edges[0]
v_start = e_start.v_i

h_start = e_start.define_halfedge(v_start)

v_current = v_start
e_current = e_start
h_current = h_start

traverse(h_start, v_start, v_current, e_current, h_current)


# # # # # # # # # # #
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
