"""
Model Generator for OpenSees ~ split component
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

import numpy as np
from ..line import Line
from ..ops.element import ElasticBeamColumn
from ..ops.element import DispBeamColumn
from ..ops.element import GeomTransf
from ..ops.node import Node
from .. import common
from ..ops.element import Lobatto


def split_component(component, point):
    """
    Splits a beam-functioning component assembly to
    accomodate for a node required to connect
    another component assembly
    """
    elms = []
    elms.extend(component.elastic_beamcolumn_elements.values())
    elms.extend(component.disp_beamcolumn_elements.values())
    distances = np.zeros(len(elms))
    for i, elm in enumerate(elms):
        p_i = (np.array(elm.nodes[0].coords)
               + elm.geomtransf.offset_i)
        p_j = (np.array(elm.nodes[1].coords)
               + elm.geomtransf.offset_j)
        line = Line('', p_i, p_j)
        dist = line.point_distance(point)
        distances[i] = dist
    np.nan_to_num(distances, copy=False, nan=np.inf)
    i_min = np.argmin(distances)
    closest_elm = elms[i_min]
    p_i = (np.array(closest_elm.nodes[0].coords)
           + closest_elm.geomtransf.offset_i)
    p_j = (np.array(closest_elm.nodes[1].coords)
           + closest_elm.geomtransf.offset_j)
    line = Line('', p_i, p_j)
    split_point = line.project(point)
    assert split_point is not None  # check if it exists

    # first check if a node already exists there
    inodes = component.internal_nodes.values()
    for inode in inodes:
        if np.linalg.norm(
                np.array(inode.coords) - split_point) < common.EPSILON:
            avail_node = inode
            offset = point - np.array(avail_node.coords)
            return avail_node, offset

    # otherwise:

    # remove existing line element
    node_i = closest_elm.nodes[0]
    node_j = closest_elm.nodes[1]
    prev_section = closest_elm.section
    prev_gtransf = closest_elm.geomtransf
    if isinstance(closest_elm, ElasticBeamColumn):
        component.elastic_beamcolumn_elements.pop(closest_elm.uid)
    elif isinstance(closest_elm, DispBeamColumn):
        component.disp_beamcolumn_elements.pop(closest_elm.uid)
    else:
        raise ValueError('Unsupported element type')

    # add split node
    middle_node = Node(
        component.parent_collection.parent
        .parent_model.uid_generator.new('node'),
        list(split_point))
    component.internal_nodes.add(middle_node)
    # add two new line elements
    # part i
    o_i = prev_gtransf.offset_i
    o_j = np.zeros(3)
    n_i = node_i
    n_j = middle_node
    transf_i = GeomTransf(
        prev_gtransf.transf_type,
        component.parent_collection.parent.parent_model
        .uid_generator.new('transformation'),
        o_i,
        o_j,
        prev_gtransf.x_axis,
        prev_gtransf.y_axis,
        prev_gtransf.z_axis
    )
    if isinstance(closest_elm, ElasticBeamColumn):
        elm_i = ElasticBeamColumn(
            component,
            component.parent_collection.parent.parent_model
            .uid_generator.new('element'),
            [n_i, n_j],
            prev_section,
            transf_i
        )
        component.elastic_beamcolumn_elements.add(elm_i)
    elif isinstance(closest_elm, DispBeamColumn):
        assert isinstance(closest_elm.integration, Lobatto)
        beam_integration = Lobatto(
            uid=component.parent_collection.parent.parent_model
            .uid_generator.new('beam integration'),
            parent_section=prev_section,
            n_p=closest_elm.integration.n_p
        )
        elm_i = DispBeamColumn(  # type: ignore
            component,
            component.parent_collection.parent.parent_model
            .uid_generator.new('element'),
            [n_i, n_j],
            prev_section,
            transf_i,
            beam_integration
        )
        component.disp_beamcolumn_elements.add(elm_i)
    # part j
    o_i = np.zeros(3)
    o_j = prev_gtransf.offset_j
    n_i = middle_node
    n_j = node_j
    transf_j = GeomTransf(
        prev_gtransf.transf_type,
        component.parent_collection.parent.parent_model
        .uid_generator.new('transformation'),
        o_i,
        o_j,
        prev_gtransf.x_axis,
        prev_gtransf.y_axis,
        prev_gtransf.z_axis
    )
    if isinstance(closest_elm, ElasticBeamColumn):
        elm_j = ElasticBeamColumn(
            component,
            component.parent_collection.parent.parent_model
            .uid_generator.new('element'),
            [n_i, n_j],
            prev_section,
            transf_j
        )
        component.elastic_beamcolumn_elements.add(elm_j)
    elif isinstance(closest_elm, DispBeamColumn):
        assert isinstance(closest_elm.integration, Lobatto)
        beam_integration = Lobatto(
            uid=component.parent_collection.parent.parent_model.
            uid_generator.new('beam integration'),
            parent_section=prev_section,
            n_p=closest_elm.integration.n_p
        )
        elm_j = DispBeamColumn(  # type: ignore
            component,
            component.parent_collection.parent.parent_model
            .uid_generator.new('element'),
            [n_i, n_j],
            prev_section,
            transf_j,
            beam_integration
        )
        component.disp_beamcolumn_elements.add(elm_j)

    # calculate offset and return
    offset = point - np.array(middle_node.coords)
    return middle_node, offset
