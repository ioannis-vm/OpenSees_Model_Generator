from tributary_area_load_distr.mesher import Vertex, Edge, tributary_areas

# define the vertices
my_vertices = [
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
my_edges = [
    Edge(my_vertices[0], my_vertices[1]),
    Edge(my_vertices[1], my_vertices[2]),
    Edge(my_vertices[0], my_vertices[3]),
    Edge(my_vertices[0], my_vertices[4]),
    Edge(my_vertices[1], my_vertices[4]),
    Edge(my_vertices[2], my_vertices[7]),
    Edge(my_vertices[3], my_vertices[4]),
    Edge(my_vertices[3], my_vertices[5]),
    Edge(my_vertices[4], my_vertices[5]),
    Edge(my_vertices[4], my_vertices[6]),
    Edge(my_vertices[4], my_vertices[7]),
    Edge(my_vertices[5], my_vertices[6]),
    Edge(my_vertices[6], my_vertices[7]),
]


areas = tributary_areas(my_edges, print_halfedges=True)
