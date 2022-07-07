import numpy as np
from ..line import Line
from ..ops.element import elasticBeamColumn
from ..ops.element import dispBeamColumn
from ..ops.element import geomTransf
from ..ops.node import Node
from .. import common
from ..ops.element import Lobatto


def split_component(component, point):

    elms = []
    elms.extend(component.elastic_beamcolumn_elements.registry.values())
    elms.extend(component.force_beamcolumn_elements.registry.values())
    split_elm = None
    distances = np.zeros(len(elms))
    for i, elm in enumerate(elms):
        p_i = (np.array(elm.eleNodes[0].coords)
               + elm.geomtransf.offset_i)
        p_j = (np.array(elm.eleNodes[1].coords)
               + elm.geomtransf.offset_j)
        line = Line('', p_i, p_j)
        dist = line.point_distance(point)
        distances[i] = dist
    np.nan_to_num(distances, copy=False, nan=np.inf)
    i_min = np.argmin(distances)
    closest_elm = elms[i_min]
    p_i = (np.array(closest_elm.eleNodes[0].coords)
           + closest_elm.geomtransf.offset_i)
    p_j = (np.array(closest_elm.eleNodes[1].coords)
           + closest_elm.geomtransf.offset_j)
    line = Line('', p_i, p_j)
    split_point = line.project(point)

    # first check if a node already exists there
    inodes = component.internal_nodes.registry.values()
    for inode in inodes:
        if np.linalg.norm(np.array(inode.coords) - split_point) < common.EPSILON:
            avail_node = inode
            offset = point - np.array(avail_node.coords)
            return avail_node, offset

    # otherwise:

    # remove existing line element
    element_class = closest_elm.__class__
    node_i = closest_elm.eleNodes[0]
    node_j = closest_elm.eleNodes[1]
    prev_section = closest_elm.section
    prev_gtransf = closest_elm.geomtransf
    if isinstance(closest_elm, elasticBeamColumn):
        component.elastic_beamcolumn_elements.registry.pop(closest_elm.uid)
    elif isinstance(closest_elm, dispBeamColumn):
        component.force_beamcolumn_elements.registry.pop(closest_elm.uid)
    else:
        raise ValueError('Unsupported element type')
    
    # add split node
    middle_node = Node(
        component.parent_collection.parent.parent_model.uid_generator.new('node'),
        [*split_point])
    component.internal_nodes.add(middle_node)
    # add two new line elements
    # part i
    o_i = prev_gtransf.offset_i
    o_j = np.zeros(3)
    n_i = node_i
    n_j = middle_node
    conn_i_list = [n_i.uid, n_j.uid]
    conn_i_list.sort()
    conn_i = (*conn_i_list,)
    transf_i = geomTransf(
        prev_gtransf.transfType,
        component.parent_collection.parent.parent_model.uid_generator.new('transformation'),
        o_i,
        o_j,
        prev_gtransf.x_axis,
        prev_gtransf.y_axis,
        prev_gtransf.z_axis
    )
    if isinstance(closest_elm, elasticBeamColumn):
        elm_i = elasticBeamColumn(
            component,
            component.parent_collection.parent.parent_model.uid_generator.new('element'),
            [n_i, n_j],
            prev_section,
            transf_i
        )
        component.elastic_beamcolumn_elements.add(elm_i)
    elif isinstance(closest_elm, dispBeamColumn):
        beam_integration = Lobatto(
            uid=component.parent_collection.parent.parent_model.uid_generator.new('beam integration'),
            parent_section=prev_section,
            n_p=closest_elm.integration.n_p
        )
        elm_i = dispBeamColumn(
            component,
            component.parent_collection.parent.parent_model.uid_generator.new('element'),
            [n_i, n_j],
            prev_section,
            transf_i,
            beam_integration
        )
        component.force_beamcolumn_elements.add(elm_i)
    # part j
    o_i = np.zeros(3)
    o_j = prev_gtransf.offset_j
    n_i = middle_node
    n_j = node_j
    conn_j_list = [n_i.uid, n_j.uid]
    conn_j_list.sort()
    conn_j = (*conn_j_list,)
    transf_j = geomTransf(
        prev_gtransf.transfType,
        component.parent_collection.parent.parent_model.uid_generator.new('transformation'),
        o_i,
        o_j,
        prev_gtransf.x_axis,
        prev_gtransf.y_axis,
        prev_gtransf.z_axis
    )
    if isinstance(closest_elm, elasticBeamColumn):
        elm_j = elasticBeamColumn(
            component,
            component.parent_collection.parent.parent_model.uid_generator.new('element'),
            [n_i, n_j],
            prev_section,
            transf_j
        )
        component.elastic_beamcolumn_elements.add(elm_j)
    elif isinstance(closest_elm, dispBeamColumn):
        beam_integration = Lobatto(
            uid=component.parent_collection.parent.parent_model.uid_generator.new('beam integration'),
            parent_section=prev_section,
            n_p=closest_elm.integration.n_p
        )
        elm_j = dispBeamColumn(
            component,
            component.parent_collection.parent.parent_model.uid_generator.new('element'),
            [n_i, n_j],
            prev_section,
            transf_j,
            beam_integration
        )
        component.force_beamcolumn_elements.add(elm_j)
    
    
    # reconfigure connectivity
    uid_list = [node_i.uid, node_j.uid]
    uid_list.sort()
    uid_tuple = (*uid_list,)
    component.element_connectivity.pop(uid_tuple)
    component.element_connectivity[conn_i] = elm_i
    component.element_connectivity[conn_j] = elm_j

    # calculate offset and return
    offset = point - np.array(middle_node.coords)
    return middle_node, offset
