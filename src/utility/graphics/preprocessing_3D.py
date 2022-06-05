"""
The following utility functions are used for data visualization
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

import plotly.graph_objects as go
import numpy as np
from utility.graphics import common, common_3D


def add_data__grids(dt, building):
    for lvl in building.levels.registry.values():
        elevation = lvl.elevation
        for grid in building.gridsystem.grids:
            dt.append({
                "type": "scatter3d",
                "mode": "lines",
                "x": [grid.start[0], grid.end[0]],
                "y": [grid.start[1], grid.end[1]],
                "z": [elevation]*2,
                "hoverinfo": "skip",
                "line": {
                    "width": 4,
                    "color": common.GRID_COLOR
                }
            })


def add_data__nodes(dt, list_of_nodes):
    x = [node.coords[0] for node in list_of_nodes]
    y = [node.coords[1] for node in list_of_nodes]
    z = [node.coords[2] for node in list_of_nodes]
    customdata = []
    restraint_types = [node.restraint_type for node in list_of_nodes]
    for node in list_of_nodes:
        customdata.append(
            (node.uid,
             *node.mass,
             *node.load_total()
             )
        )

    customdata = np.array(customdata, dtype='object')

    dt.append({
        "type": "scatter3d",
        "mode": "markers",
        "x": x,
        "y": y,
        "z": z,
        "customdata": customdata,
        "text": restraint_types,
        "hovertemplate": 'Coordinates: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
        'Restraint: %{text}<br>' +
        'Mass: (%{customdata[1]:.3g}, ' +
        '%{customdata[2]:.3g}, %{customdata[3]:.3g})<br>' +
        'Load: (%{customdata[4]:.3g}, ' +
        '%{customdata[5]:.3g}, %{customdata[6]:.3g})' +
        '<extra>Node: %{customdata[0]:d}</extra>',
        "marker": {
            "symbol": [common_3D.node_marker[node.restraint_type][0]
                       for node in list_of_nodes],
            "color": common.NODE_PRIMARY_COLOR,
            "size": [common_3D.node_marker[node.restraint_type][1]
                     for node in list_of_nodes],
            "line": {
                "color": common.NODE_PRIMARY_COLOR,
                "width": 4}
        }
    })


def add_data__parent_nodes(dt, list_of_nodes):

    x = [node.coords[0] for node in list_of_nodes]
    y = [node.coords[1] for node in list_of_nodes]
    z = [node.coords[2] for node in list_of_nodes]
    customdata = []
    restraint_types = [node.restraint_type for node in list_of_nodes]
    for node in list_of_nodes:
        customdata.append(
            (node.uid,
             *node.mass,
             *node.load_total()
             )
        )

    customdata = np.array(customdata, dtype='object')
    dt.append({
        "type": "scatter3d",
        "mode": "markers",
        "x": x,
        "y": y,
        "z": z,
        "customdata": customdata,
        "text": restraint_types,
        "hovertemplate": 'Coordinates: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
        'Restraint: %{text}<br>' +
        'Mass: (%{customdata[1]:.3g}, ' +
        '%{customdata[2]:.3g}, %{customdata[3]:.3g}, ' +
        '%{customdata[4]:.3g}, %{customdata[5]:.3g}, ' +
        '%{customdata[6]:.3g})<br>' +
        'Load: (%{customdata[7]:.3g}, ' +
        '%{customdata[8]:.3g}, %{customdata[9]:.3g}, ' +
        '%{customdata[10]:.3g}, %{customdata[11]:.3g}, ' +
        '%{customdata[12]:.3g})' +
        '<extra>Parent Node: %{customdata[0]:d}</extra>',
        "marker": {
            "symbol": [common_3D.node_marker[node.restraint_type][0]
                       for node in list_of_nodes],
            "color": common.NODE_PRIMARY_COLOR,
            "size": [common_3D.node_marker[node.restraint_type][1]
                     for node in list_of_nodes],
            "line": {
                "color": common.NODE_PRIMARY_COLOR,
                "width": 4}
        }
    })


def add_data__internal_nodes(dt, list_of_nodes):
    x = [node.coords[0] for node in list_of_nodes]
    y = [node.coords[1] for node in list_of_nodes]
    z = [node.coords[2] for node in list_of_nodes]
    customdata = []
    restraint_types = [node.restraint_type for node in list_of_nodes]
    for node in list_of_nodes:
        customdata.append(
            (node.uid,
             *node.mass,
             *node.load_total()
             )
        )
    dt.append({
        "type": "scatter3d",
        "mode": "markers",
        "x": x,
        "y": y,
        "z": z,
        "customdata": customdata,
        "text": restraint_types,
        "hovertemplate": 'Coordinates: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
        'Restraint: %{text}<br>' +
        'Mass: (%{customdata[1]:.3g}, ' +
        '%{customdata[2]:.3g}, %{customdata[3]:.3g})<br>' +
        'Load: (%{customdata[4]:.3g}, ' +
        '%{customdata[5]:.3g}, %{customdata[6]:.3g})' +
        '<extra>Node: %{customdata[0]:d}</extra>',
        "marker": {
            "symbol": common_3D.node_marker['internal'][0],
            "color": common.NODE_INTERNAL_COLOR,
            "size": common_3D.node_marker['internal'][1],
            "line": {
                "color": common.NODE_INTERNAL_COLOR,
                "width": 2}
        }
    })


def add_data__release_nodes(dt, list_of_nodes):
    x = [node.coords[0] for node in list_of_nodes]
    y = [node.coords[1] for node in list_of_nodes]
    z = [node.coords[2] for node in list_of_nodes]
    customdata = []
    restraint_types = [node.restraint_type for node in list_of_nodes]
    for node in list_of_nodes:
        customdata.append(
            (node.uid,
             *node.mass,
             *node.load_total()
             )
        )
    dt.append({
        "type": "scatter3d",
        "mode": "markers",
        "x": x,
        "y": y,
        "z": z,
        "hoverinfo": "skip",
        "marker": {
            "symbol": common_3D.node_marker['pinned'][0],
            "color": common.NODE_INTERNAL_COLOR,
            "size": common_3D.node_marker['pinned'][1],
            "line": {
                "color": common.NODE_INTERNAL_COLOR,
                "width": 2}
        }
    })


def add_data__diaphragm_lines(dt, lvl):
    if not lvl.parent_node:
        return
    mnode = lvl.parent_node
    x = []
    y = []
    z = []
    for node in lvl.nodes_primary.registry.values():
        x.extend(
            (node.coords[0], mnode.coords[0], None)
        )
        y.extend(
            (node.coords[1], mnode.coords[1], None)
        )
        z.extend(
            (node.coords[2], mnode.coords[2], None)
        )
    dt.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "hoverinfo": "skip",
        "line": {
            "width": 4,
            "dash": "dash",
            "color": common.GRID_COLOR
        }
    })


def add_data__bisector_lines(dt, lvl):
    if not lvl.parent_node:
        return
    x = []
    y = []
    z = []
    for line in lvl.floor_bisector_lines:
        p1 = line[0]
        p2 = line[1]
        x.extend(
            (p1[0], p2[0], None)
        )
        y.extend(
            (p1[1], p2[1], None)
        )
        z.extend(
            (lvl.elevation, lvl.elevation, None)
        )
    dt.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "hoverinfo": "skip",
        "line": {
            "width": 4,
            "color": common.BISECTOR_COLOR
        }
    })


def add_data__frames(dt, list_of_frames):
    x = []
    y = []
    z = []
    customdata = []
    section_names = []
    for elm in list_of_frames:
        if elm.hidden_as_line:
            continue
        section_names.extend([elm.section.name]*3)
        x.extend(
            (elm.internal_pt_i[0], elm.internal_pt_j[0], None)
        )
        y.extend(
            (elm.internal_pt_i[1], elm.internal_pt_j[1], None)
        )
        z.extend(
            (elm.internal_pt_i[2], elm.internal_pt_j[2], None)
        )
        customdata.append(
            (elm.uid,
             *elm.udl_total(),
             elm.node_i.uid,
             elm.parent.uid)
        )
        customdata.append(
            (elm.uid,
             *elm.udl_total(),
             elm.node_j.uid,
             elm.parent.uid)
        )
        customdata.append(
            [None]*6
        )

    customdata = np.array(customdata, dtype='object')
    dt.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "text": section_names,
        "customdata": customdata,
        "hovertemplate": 'Section: %{text}<br>' +
        'UDL (local): (%{customdata[1]:.3g}, ' +
        '%{customdata[2]:.3g}, %{customdata[3]:.3g})' +
        '<extra>Element: %{customdata[0]:d}<br>' +
        'Node @ this end: %{customdata[4]:d}<br>'
        'Parent: %{customdata[5]}</extra>',
        "line": {
            "width": 5,
            "color": common.FRAME_COLOR
        }
    })


def add_data__frame_offsets(dt, list_of_elems):
    if not list_of_elems:
        return

    x = []
    y = []
    z = []

    for elm in list_of_elems:
        p_i = elm.node_i.coords
        p_io = elm.end_segment_i.internal_pt
        p_j = elm.node_j.coords
        p_jo = elm.end_segment_j.internal_pt

        x.extend((p_i[0], p_io[0], None))
        y.extend((p_i[1], p_io[1], None))
        z.extend((p_i[2], p_io[2], None))
        x.extend((p_j[0], p_jo[0], None))
        y.extend((p_j[1], p_jo[1], None))
        z.extend((p_j[2], p_jo[2], None))

    dt.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "hoverinfo": "skip",
        "line": {
            "width": 8,
            "color": common.OFFSET_COLOR
        }
    })


def add_data__frame_axes(dt, list_of_frames, ref_len):
    if not list_of_frames:
        return
    s = ref_len * 0.025
    x = []
    y = []
    z = []
    colors = []
    for elm in list_of_frames:
        if elm.hidden_as_line:
            continue
        x_vec = elm.x_axis
        y_vec = elm.y_axis
        z_vec = elm.z_axis
        l_clear = elm.length_clear
        i_pos = np.array(elm.internal_pt_i)
        mid_pos = i_pos + x_vec * l_clear/2.00
        x.extend((mid_pos[0], mid_pos[0]+x_vec[0]*s, None))
        y.extend((mid_pos[1], mid_pos[1]+x_vec[1]*s, None))
        z.extend((mid_pos[2], mid_pos[2]+x_vec[2]*s, None))
        colors.extend(["red"]*3)
        x.extend((mid_pos[0], mid_pos[0]+y_vec[0]*s, None))
        y.extend((mid_pos[1], mid_pos[1]+y_vec[1]*s, None))
        z.extend((mid_pos[2], mid_pos[2]+y_vec[2]*s, None))
        colors.extend(["green"]*3)
        x.extend((mid_pos[0], mid_pos[0]+z_vec[0]*s, None))
        y.extend((mid_pos[1], mid_pos[1]+z_vec[1]*s, None))
        z.extend((mid_pos[2], mid_pos[2]+z_vec[2]*s, None))
        colors.extend(["blue"]*3)
    dt.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "hoverinfo": "skip",
        "line": {
            "width": 8,
            "color": colors
        }
    })


def add_data__global_axes(dt, ref_len):

    s = ref_len
    x = []
    y = []
    z = []
    colors = []
    x_vec = np.array([1.00, 0.00, 0.00])
    y_vec = np.array([0.00, 1.00, 0.00])
    z_vec = np.array([0.00, 0.00, 1.00])

    x.extend((0.00, x_vec[0]*s, None))
    y.extend((0.00, x_vec[1]*s, None))
    z.extend((0.00, x_vec[2]*s, None))
    x.extend((0.00, y_vec[0]*s, None))
    y.extend((0.00, y_vec[1]*s, None))
    z.extend((0.00, y_vec[2]*s, None))
    x.extend((0.00, z_vec[0]*s, None))
    y.extend((0.00, z_vec[1]*s, None))
    z.extend((0.00, z_vec[2]*s, None))
    colors.extend(["red"]*3)
    colors.extend(["green"]*3)
    colors.extend(["blue"]*3)
    dt.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "hoverinfo": "skip",
        "line": {
            "width": 3,
            "color": colors
        }
    })


def add_data__extruded_frames_mesh(dt, list_of_frames):
    if not list_of_frames:
        return
    x_list = []
    y_list = []
    z_list = []
    i_list = []
    j_list = []
    k_list = []
    index = 0
    for elm in list_of_frames:
        if elm.hidden_when_extruded:
            continue
        side_a = np.array(elm.internal_pt_i)
        side_b = np.array(elm.internal_pt_j)
        y_vec = elm.y_axis
        z_vec = elm.z_axis
        loop = elm.section.mesh.halfedges
        for halfedge in loop:
            loc0 = halfedge.vertex.coords[0]*z_vec +\
                halfedge.vertex.coords[1]*y_vec + side_a
            loc1 = halfedge.vertex.coords[0]*z_vec +\
                halfedge.vertex.coords[1]*y_vec + side_b
            loc2 = halfedge.nxt.vertex.coords[0]*z_vec +\
                halfedge.nxt.vertex.coords[1]*y_vec + side_b
            loc3 = halfedge.nxt.vertex.coords[0]*z_vec +\
                halfedge.nxt.vertex.coords[1]*y_vec + side_a
            x_list.append(loc0[0])
            y_list.append(loc0[1])
            z_list.append(loc0[2])
            x_list.append(loc1[0])
            y_list.append(loc1[1])
            z_list.append(loc1[2])
            x_list.append(loc2[0])
            y_list.append(loc2[1])
            z_list.append(loc2[2])
            x_list.append(loc3[0])
            y_list.append(loc3[1])
            z_list.append(loc3[2])
            i_list.append(index + 0)
            j_list.append(index + 1)
            k_list.append(index + 2)
            i_list.append(index + 0)
            j_list.append(index + 2)
            k_list.append(index + 3)
            index += 4
    dt.append({
        "type": "mesh3d",
        "x": x_list,
        "y": y_list,
        "z": z_list,
        "i": i_list,
        "j": j_list,
        "k": k_list,
        "hoverinfo": "skip",
        "color": common.BEAM_MESH_COLOR,
        "opacity": 0.65
    })


def add_data__extruded_steel_W_PZ_mesh(dt, list_of_endsegments):

    if not list_of_endsegments:
        return
    x_list = []
    y_list = []
    z_list = []
    i_list = []
    j_list = []
    k_list = []
    index = 0

    for elm in list_of_endsegments:

        side_a = np.array(elm.internal_pt_i)
        side_b = np.array(elm.internal_pt_j)
        x_vec = elm.parent.x_axis
        y_vec = elm.parent.y_axis
        z_vec = np.cross(x_vec, y_vec)
        loop = elm.parent.section.mesh.halfedges

        for halfedge in loop:
            loc0 = halfedge.vertex.coords[0]*z_vec +\
                halfedge.vertex.coords[1]*y_vec + side_a
            loc1 = halfedge.vertex.coords[0]*z_vec +\
                halfedge.vertex.coords[1]*y_vec + side_b
            loc2 = halfedge.nxt.vertex.coords[0]*z_vec +\
                halfedge.nxt.vertex.coords[1]*y_vec + side_b
            loc3 = halfedge.nxt.vertex.coords[0]*z_vec +\
                halfedge.nxt.vertex.coords[1]*y_vec + side_a
            x_list.append(loc0[0])
            y_list.append(loc0[1])
            z_list.append(loc0[2])
            x_list.append(loc1[0])
            y_list.append(loc1[1])
            z_list.append(loc1[2])
            x_list.append(loc2[0])
            y_list.append(loc2[1])
            z_list.append(loc2[2])
            x_list.append(loc3[0])
            y_list.append(loc3[1])
            z_list.append(loc3[2])
            i_list.append(index + 0)
            j_list.append(index + 1)
            k_list.append(index + 2)
            i_list.append(index + 0)
            j_list.append(index + 2)
            k_list.append(index + 3)
            index += 4
    dt.append({
        "type": "mesh3d",
        "x": x_list,
        "y": y_list,
        "z": z_list,
        "i": i_list,
        "j": j_list,
        "k": k_list,
        "hoverinfo": "skip",
        "color": common.BEAM_MESH_COLOR,
        "opacity": 0.65
    })


def plot_building_geometry(building: 'Model',
                           extrude_frames=False,
                           offsets=True,
                           gridlines=True,
                           global_axes=True,
                           diaphragm_lines=True,
                           tributary_areas=True,
                           just_selection=False,
                           parent_nodes=True,
                           frame_axes=True,
                           camera=None
                           ):

    layout = common_3D.global_layout(camera)
    dt = []

    ref_len = building.reference_length()

    # plot the nodes
    if not just_selection:
        nodes_primary = building.list_of_primary_nodes()
        nodes_internal = building.list_of_internal_nodes()
        releases = [x.node_i for x in building.list_of_endreleases()]
    else:
        nodes_primary = building.selection.list_of_primary_nodes()
        nodes_internal = building.selection.list_of_internal_nodes()
    add_data__nodes(dt, nodes_primary)
    if not extrude_frames:
        add_data__internal_nodes(dt, nodes_internal)
        add_data__release_nodes(dt, releases)

    # grid lines
    if gridlines:
        add_data__grids(dt, building)

    # global axes
    if global_axes:
        add_data__global_axes(dt, ref_len)

    # diaphgragm lines
    if diaphragm_lines:
        for lvl in building.levels.registry.values():
            add_data__diaphragm_lines(dt, lvl)

    # bisector lines
    if tributary_areas:
        for lvl in building.levels.registry.values():
            add_data__bisector_lines(dt, lvl)

    # plot the linear elements
    if just_selection:
        line_element_sequences = \
            building.selection.list_of_line_element_sequences()
        line_elems = building.selection.list_of_line_elements()
    else:
        line_element_sequences = building.list_of_line_element_sequences()
        line_elems = building.list_of_line_elements()
        list_of_steel_W_panel_zones = building.list_of_steel_W_panel_zones()

    if extrude_frames:
        add_data__extruded_frames_mesh(
            dt, line_elems)
        add_data__extruded_steel_W_PZ_mesh(
            dt, list_of_steel_W_panel_zones)
    else:
        add_data__frames(dt, line_elems)
        if frame_axes:
            add_data__frame_axes(dt, line_elems,
                                 building.reference_length())
    # plot the rigid offsets
    if offsets:
        add_data__frame_offsets(dt, line_element_sequences)

    # plot the parent nodes
    if parent_nodes:
        add_data__parent_nodes(dt, building.list_of_parent_nodes())

    fig_datastructure = dict(data=dt, layout=layout)
    fig = go.Figure(fig_datastructure)
    fig.show()
