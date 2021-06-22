from utility import mesher


def generate(edges):
    halfedges = mesher.define_halfedges(edges)
    loops = mesher.obtain_closed_loops(halfedges)
    external, _, trivial = mesher.orient_loops(loops)
    mesher.sanity_checks(external, trivial)
    return mesher.Mesh(external[0])


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
        mesher.Vertex((b/2., h/2.)),
        mesher.Vertex((-b/2., h/2.)),
        mesher.Vertex((-b/2., h/2.-tf)),
        mesher.Vertex((-tw/2., h/2.-tf)),
        mesher.Vertex((-tw/2., -(h/2.-tf))),
        mesher.Vertex((-b/2., -(h/2.-tf))),
        mesher.Vertex((-b/2., -h/2.)),
        mesher.Vertex((b/2., -h/2.)),
        mesher.Vertex((b/2., -(h/2-tf))),
        mesher.Vertex((tw/2., -(h/2-tf))),
        mesher.Vertex((tw/2., h/2.-tf)),
        mesher.Vertex((b/2., h/2.-tf))
    ]
    edges = [
        mesher.Edge(vertices[0], vertices[1]),
        mesher.Edge(vertices[1], vertices[2]),
        mesher.Edge(vertices[2], vertices[3]),
        mesher.Edge(vertices[3], vertices[4]),
        mesher.Edge(vertices[4], vertices[5]),
        mesher.Edge(vertices[5], vertices[6]),
        mesher.Edge(vertices[6], vertices[7]),
        mesher.Edge(vertices[7], vertices[8]),
        mesher.Edge(vertices[8], vertices[9]),
        mesher.Edge(vertices[9], vertices[10]),
        mesher.Edge(vertices[10], vertices[11]),
        mesher.Edge(vertices[11], vertices[0])
    ]
    return generate(edges)


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
        mesher.Vertex((b/2., h/2.)),
        mesher.Vertex((-b/2., h/2.)),
        mesher.Vertex((-b/2., -h/2.)),
        mesher.Vertex((b/2., -h/2.))
    ]
    edges = [
        mesher.Edge(vertices[0], vertices[1]),
        mesher.Edge(vertices[1], vertices[2]),
        mesher.Edge(vertices[2], vertices[3]),
        mesher.Edge(vertices[3], vertices[0]),
    ]
    return generate(edges)
