import skgeom as sg
import numpy as np
from utility.mesher import Vertex, Edge
from utility.mesher import define_halfedges
from utility.mesher import print_halfedge_results
from utility.mesher import obtain_closed_loops
from utility.mesher import geometric_properties


def calculate_tributary_areas_from_loops(loops):

    def accumulate_areas(areas, index, value):
        """
        Using a dictionary (`areas`) to accumulate the areas
        associated with each vertex for tributary-area load
        distribution.
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
                if (other_halfedge.vertex.point ==
                    halfedge.vertex.point and
                        other_halfedge.next.vertex.point ==
                        halfedge.next.vertex.point):
                    return True
        return False

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

        miniloop_areas = [float(sg.Polygon(
            [h.vertex.point for h in miniloop]).area())
            for miniloop in miniloops]
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
    return areas


def generate_floor_slab_data(lvl):
    """
    Used after locking the model.
    Generates floor slabs data, containing:
    TODO describe conents
    """
    def convert_list_of_beams_to_mesh_edges(beams):
        """
        As the name suggests.
        Uses the beams as a mesh
        and returns the edge elements.
        """
        vertices = []
        edges = []
        coordinate_list = []
        for beam in beams:
            i_coords = beam.node_i.coordinates[0:2]
            if i_coords not in coordinate_list:
                coordinate_list.append(i_coords)
            j_coords = beam.node_j.coordinates[0:2]
            if j_coords not in coordinate_list:
                coordinate_list.append(j_coords)
        # define Vertices
        for coord in coordinate_list:
            vertices.append(Vertex(coord))
        # define edges
        beam_to_edge_map = {}
        for beam in beams:
            i_coords = beam.node_i.coordinates[0:2]
            i_index = coordinate_list.index(i_coords)
            j_coords = beam.node_j.coordinates[0:2]
            j_index = coordinate_list.index(j_coords)
            edge = Edge(vertices[i_index], vertices[j_index])
            edges.append(edge)
            beam_to_edge_map[beam.uniq_id] = edge.nid
        return edges, beam_to_edge_map

    beams = lvl.beams.element_list
    if beams:
        edges, beam_to_edge_map = convert_list_of_beams_to_mesh_edges(beams)
        halfedges = define_halfedges(edges)
        external_loop, loops, loop_areas = obtain_closed_loops(halfedges)
        coords = np.array([h.vertex.coords for h in external_loop])
        properties = geometric_properties(coords)
        lvl.slab_data = {
            'external_loop': external_loop,
            'loops': loops,
            'beam_to_edge_map': beam_to_edge_map,
            'loop_areas': loop_areas,
            'properties': properties
        }
    else:
        lvl.slab_data = None
