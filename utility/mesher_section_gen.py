from utility.mesher import Vertex, Edge, Halfedge, Mesh, define_halfedges, obtain_closed_loops, plot_mesh, print_halfedge_results


def w_mesh(b, h, tw, tf):
    """
    Defines a loop of counterclockwise halfedges
    that form the shape of the W section with
    the specified parameters.
    The origin coincides with the centroid.
    Input:
        b: total width
        h: total height
        tw: web thickness
        tf: flange thickness
    """
    vertices = [
        Vertex((b/2., h/2.)),
        Vertex((-b/2., h/2.)),
        Vertex((-b/2., h/2.-tf)),
        Vertex((-tw/2., h/2.-tf)),
        Vertex((-tw/2., -(h/2.-tf))),
        Vertex((-b/2., -(h/2.-tf))),
        Vertex((-b/2., -h/2.)),
        Vertex((b/2., -h/2.)),
        Vertex((b/2., -(h/2-tf))),
        Vertex((tw/2., -(h/2-tf))),
        Vertex((tw/2., h/2.-tf)),
        Vertex((b/2., h/2.-tf))
    ]
    edges = [
        Edge(vertices[0], vertices[1]),
        Edge(vertices[1], vertices[2]),
        Edge(vertices[2], vertices[3]),
        Edge(vertices[3], vertices[4]),
        Edge(vertices[4], vertices[5]),
        Edge(vertices[5], vertices[6]),
        Edge(vertices[6], vertices[7]),
        Edge(vertices[7], vertices[8]),
        Edge(vertices[8], vertices[9]),
        Edge(vertices[9], vertices[10]),
        Edge(vertices[10], vertices[11]),
        Edge(vertices[11], vertices[0])
    ]
    halfedges = define_halfedges(edges)
    loop = obtain_closed_loops(halfedges)[1][0]
    return(Mesh(loop))


def rect_mesh(b, h):
    """
    Defines a loop of counterclockwise halfedges
    that form the shape of the rectangular section with
    the specified parameters.
    The origin coincides with the centroid.
    Input:
        b: total width
        h: total height
    """
    vertices = [
        Vertex((b/2., h/2.)),
        Vertex((-b/2., h/2.)),
        Vertex((-b/2., -h/2.)),
        Vertex((b/2., -h/2.))
    ]
    edges = [
        Edge(vertices[0], vertices[1]),
        Edge(vertices[1], vertices[2]),
        Edge(vertices[2], vertices[3]),
        Edge(vertices[3], vertices[0]),
    ]
    halfedges = define_halfedges(edges)
    loop = obtain_closed_loops(halfedges)[1][0]
    return(Mesh(loop))
