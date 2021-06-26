import skgeom as sg
import numpy as np
from utility import mesher


def list_of_beams_to_mesh_edges_external(beams):
    """
    Defines `Edge` elements that have the same starting
    and ending point as the primary nodes of the provided
    BeamColumn elements, and maps those edges to their
    generatory BeamColumn elements.
    Args:
        beams (list[BeamColumn]): All BeamColumn elements
              that correspond to the beams of a single
              level for which tributary area analysis is
              performed.
    Returns:
        edges (list[Edge]): Generated edges.
        edge_to_beam_map (dict): Mapping between edges and beams.
    """
    vertices = []
    edges = []
    coordinate_list = []

    for beam in beams:
        i_coords = list(beam.node_i.coords[0:2])
        if i_coords not in coordinate_list:
            coordinate_list.append(i_coords)
        j_coords = list(beam.node_j.coords[0:2])
        if j_coords not in coordinate_list:
            coordinate_list.append(j_coords)
    # define Vertices
    for coord in coordinate_list:
        vertices.append(mesher.Vertex(coord))
    # define edges
    edge_to_beam_map = {}
    for beam in beams:
        i_coords = list(beam.node_i.coords[0:2])
        i_index = coordinate_list.index(i_coords)
        j_coords = list(beam.node_j.coords[0:2])
        j_index = coordinate_list.index(j_coords)
        edge = mesher.Edge(vertices[i_index], vertices[j_index])
        edges.append(edge)
        edge_to_beam_map[edge.uniq_id] = beam
    return edges, edge_to_beam_map


def closed_beam_sequences(beams):
    """
    Generates sequences of beams that form closed
    regions in plan view. Floor plans are assumed to
    occupy these regions. Subsequent functions use
    those sequences of beams to obtain their tributary
    areas. This function also returns an matrix of
    coordinates of the obtained overall floor perimeter.
    Args:
        beams (list[BeamColumn]): All BeamColumn elements
              that correspond to the beams of a single
              level for which tributary area analysis is
              performed.
    Returns:
        sequences (list[list[BeamColumn]]):
                  Sequences of beams that form closed
                  regions in plan view.
        coords (np.ndarray):
               Coordinates of the floor perimeter.
    """
    if not beams:
        return
    edges, edge_to_beam_map = \
        list_of_beams_to_mesh_edges_external(beams)

    halfedges = mesher.define_halfedges(edges)
    halfedge_to_beam_map = {}
    for h in halfedges:
        halfedge_to_beam_map[h.uniq_id] = \
            edge_to_beam_map[h.edge.uniq_id]
    loops = mesher.obtain_closed_loops(halfedges)
    external, internal, trivial = \
        mesher.orient_loops(loops)
    # Sanity checks.
    mesher.sanity_checks(external, trivial)
    coords = np.array(
        [h.vertex.coords for h in external[0]])
    # Gather sequences of beams that form closed regions
    sequences_of_beams = []
    for loop in internal:
        sequence = []
        for h in loop:
            sequence.append(halfedge_to_beam_map[
                h.uniq_id
            ])
        sequences_of_beams.append(sequence)
    return sequences_of_beams, coords


def list_of_beams_to_mesh_edges_internal(beams):
    """
    TODO (explain)
    Args:
        beams (list[BeamColumn]): TODO.
    Returns:
        edges (list[Edge]): Generated edges.
        edge_map (dict): Mapping between edges and beams or nodes.
    """

    # we have an ordered list of beams
    # that are connected in a closed sequence
    # however the orientation of the beams is
    # unknown. i.e. If we are currently at node
    # j of the current beam, we don't know if
    # that corresponds to node i or node j of the
    # next beam. (referring to the primary nodes).
    # We figure that out with the following code.

    flip = []
    for i in range(len(beams)):
        current_beam = beams[i]
        if i+1 == len(beams):
            i = -1
        next_beam = beams[i+1]
        if (current_beam.node_j == next_beam.node_i) or \
           (current_beam.node_j == next_beam.node_j):
            flip.append(False)
        elif (current_beam.node_i == next_beam.node_i) or \
             (current_beam.node_i == next_beam.node_j):
            flip.append(True)
        else:
            # By design, this should never happen
            raise ValueError("Beams are not connected")

    # create a list containing the coordinates
    # of all unique points
    vertices = []
    edges = []
    coordinate_list = []
    for i in range(len(beams)):
        i_coords = list(beams[i].internal_pt_i[0:2])
        j_coords = list(beams[i].internal_pt_j[0:2])
        if i_coords not in coordinate_list:
            coordinate_list.append(i_coords)
        if j_coords not in coordinate_list:
            coordinate_list.append(j_coords)

    # define Vertices
    for coord in coordinate_list:
        vertices.append(mesher.Vertex(coord))
    # define edges

    edge_to_beam_map = {}
    for i in range(len(beams)):
        if i+1 == len(beams):
            inext = 0
        else:
            inext = i+1
        # clear span
        i_coords = list(beams[i].internal_pt_i[0:2])
        i_index = coordinate_list.index(i_coords)
        j_coords = list(beams[i].internal_pt_j[0:2])
        j_index = coordinate_list.index(j_coords)
        # offset (internal-to-internal between
        #         previous and next beams)
        if flip[inext]:
            next_coords = list(beams[inext].internal_pt_j[0:2])
        else:
            next_coords = list(beams[inext].internal_pt_i[0:2])
        next_index = coordinate_list.index(next_coords)

        edge_span = mesher.Edge(
            vertices[i_index], vertices[j_index])
        edges.append(edge_span)

        # note: it might be the case that the projection of the
        #       offset to the x-y plane is such that next_coords
        #       coincides with the previous point, which would
        #       result in the definition of a trivial edge
        #       (an edge that is connected to the same
        #        vertex at both ends). This would break the
        #       algorithm that defines the halfedges.
        #       We avoid it by taking care not to define
        #       such trivial edges.
        if flip[i]:
            if i_index != next_index:
                edge_offset = mesher.Edge(
                    vertices[i_index], vertices[next_index])
                edges.append(edge_offset)
            else:
                edge_offset = None
        else:
            if j_index != next_index:
                edge_offset = mesher.Edge(
                    vertices[j_index], vertices[next_index])
            else:
                edge_offset = None

        # the flags represent what the edge
        #     corresponds to
        # 0 means clear length of a beam
        # 1 means offset at node i
        # 2 means offset at node j
        edge_to_beam_map[edge_span.uniq_id] = \
            (beams[i], 0)
        if edge_offset:
            if flip[i]:
                edge_to_beam_map[edge_offset.uniq_id] = \
                    (beams[i], 1)
            else:
                edge_to_beam_map[edge_offset.uniq_id] = \
                    (beams[i], 2)

    return edges, edge_to_beam_map


def calculate_tributary_areas(
        beams: list['BeamColumn']) -> list[np.ndarray]:
    """
    TODO - docstring
    """

    def is_in_some_subloop(halfedge, loops):
        for loop in loops:
            for other_halfedge in loop:
                if (other_halfedge.vertex.point ==
                    halfedge.vertex.point and
                        other_halfedge.next.vertex.point ==
                        halfedge.next.vertex.point):
                    return True
        return False

    bisectors = []

    sequences_of_beams, coords = \
        closed_beam_sequences(beams)

    for sequence in sequences_of_beams:

        edges, edge_to_beam_map = \
            list_of_beams_to_mesh_edges_internal(
                sequence)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # for edge in edges:
        #     p1 = edge.v_i.coords
        #     p2 = edge.v_j.coords
        #     coords = np.row_stack((p1, p2))
        #     plt.plot(coords[:, 0], coords[:, 1])
        # plt.show()

        # for edge in edges:
        #     p1 = np.array(edge.v_i.coords)
        #     p2 = np.array(edge.v_j.coords)
        #     print(np.linalg.norm(p2-p1))

        halfedges = mesher.define_halfedges(edges)

        halfedge_to_beam_map = {}
        for h in halfedges:
            halfedge_to_beam_map[h.uniq_id] = edge_to_beam_map[h.edge.uniq_id]
        loops = mesher.obtain_closed_loops(halfedges)
        external, internal, trivial = mesher.orient_loops(loops)
        # Sanity checks.
        mesher.sanity_checks(external, trivial)
        assert(len(internal) == 1)

        poly = sg.Polygon([h.vertex.coords for h in internal[0]])
        skel = sg.skeleton.create_interior_straight_skeleton(poly)

        subloops = []
        for halfedge in skel.halfedges:
            if subloops:
                if is_in_some_subloop(halfedge, subloops):
                    continue
            subloop = [halfedge]
            nxt = halfedge.next
            while(nxt.vertex.point != halfedge.vertex.point):
                subloop.append(nxt)
                nxt = nxt.next
            subloops.append(subloop)

        subloop_areas = [float(sg.Polygon(
            [h.vertex.point for h in subloop]).area())
            for subloop in subloops]
        outer = min(subloop_areas)  # Remove the exterior loop
        index = subloop_areas.index(outer)
        del subloops[index]
        del subloop_areas[index]

        for h in skel.halfedges:
            if h.is_bisector:
                p1 = h.vertex.point
                p1c = np.array((float(p1.x()), float(p1.y())))
                p2 = h.opposite.vertex.point
                p2c = np.array((float(p2.x()), float(p2.y())))
                pt = np.row_stack((p1c, p2c))
                bisectors.append(pt)

        # DEBUG
        #
        # fig = plt.figure()
        # for h in skel.halfedges:
        #     if h.is_bisector:
        #         p1 = h.vertex.point
        #         p2 = h.opposite.vertex.point
        #         plt.plot([p1.x(), p2.x()], [p1.y(), p2.y()], 'r-', lw=2)
        # n = len(internal[0])
        # coords = np.full((n+1, 2), 0.00)
        # for i, h in enumerate(internal[0]):
        #     coords[i, :] = h.vertex.coords
        # coords[-1, :] = coords[0, :]
        # plt.plot(coords[:, 0], coords[:, 1])
        # plt.scatter(coords[:, 0], coords[:, 1])
        # fig.show()

        for i, subloop in enumerate(subloops):
            area = subloop_areas[i]
            loop_edges = [h.edge for h in internal[0]]
            for halfedge in subloop:
                for edge in loop_edges:
                    v_i = sg.Point2(*edge.v_i.coords)
                    v_j = sg.Point2(*edge.v_j.coords)
                    pt_1 = halfedge.vertex.point
                    pt_2 = halfedge.next.vertex.point
                    if ((pt_1 == v_i and pt_2 == v_j) or
                            (pt_1 == v_j and pt_2 == v_i)):
                        beam, flag = edge_to_beam_map[edge.uniq_id]
                        if flag == 0:
                            # 0 means clear length of a beam
                            beam.tributary_area += area
                        elif flag == 1:
                            # 1 means offset at node i
                            beam.node_i.tributary_area += area
                        else:
                            # 2 means offset at node j
                            beam.node_j.tributary_area += area
    return coords, bisectors
