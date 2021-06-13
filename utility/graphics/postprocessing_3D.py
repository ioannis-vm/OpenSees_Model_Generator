"""
The following utility functions are used for data visualization
https://plotly.com/python/reference/
"""
#   __                 UC Berkeley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   April 2021
#
# https://github.com/ioannis-vm/OpenSeesPy_Building_Modeler/

import plotly.graph_objects as go
import numpy as np
from utility.graphics import common, common_3D


def interp3D_deformation(element, u_i, r_i, u_j, r_j, num_points):
    x_vec = element.local_x_axis_vector()
    y_vec = element.local_y_axis_vector()
    z_vec = element.local_z_axis_vector()

    # global -> local transformation matrix
    T_global2local = np.vstack((x_vec, y_vec, z_vec))
    T_local2global = T_global2local.T

    u_i_global = u_i
    r_i_global = r_i
    u_j_global = u_j
    r_j_global = r_j

    u_i_local = T_global2local @ u_i_global
    r_i_local = T_global2local @ r_i_global
    u_j_local = T_global2local @ u_j_global
    r_j_local = T_global2local @ r_j_global

    # discrete sample location parameter
    t = np.linspace(0.00, 1.00, num=num_points)
    l = element.length()

    # shape function matrices
    Nx_mat = np.column_stack((
        1. - t,
        t
    ))
    Nyz_mat = np.column_stack((
        1. - 3. * t**2 + 2. * t**3,
        (t - 2. * t**2 + t**3) * l,
        3. * t**2 - 2. * t**3,
        (-t**2 + t**3) * l
    ))
    Nyz_derivative_mat = np.column_stack((
        - 6. * t + 6. * t**2,
        (1 - 4. * t + 3. * t**2) * l,
        6. * t - 6. * t**2,
        (-2. * t + 3. * t**2) * l
    ))

    # axial deformation
    d_x_local = Nx_mat @ np.array([u_i_local[0], u_j_local[0]])

    # bending deformation along the local xy plane
    d_y_local = Nyz_mat @ np.array([
        u_i_local[1],
        r_i_local[2],
        u_j_local[1],
        r_j_local[2]
    ])

    # bending deformation along the local xz plane
    d_z_local = Nyz_mat @ np.array([
        u_i_local[2],
        -r_i_local[1],
        u_j_local[2],
        -r_j_local[1]
    ])

    # torsional deformation
    r_x_local = Nx_mat @ np.array([r_i_local[0], r_j_local[0]])

    # bending rotation around the local z axis
    r_z_local = Nyz_derivative_mat @ np.array([
        u_i_local[1],
        r_i_local[2],
        u_j_local[1],
        r_j_local[2]
    ]) / l

    # bending rotation around the local y axis
    r_y_local = Nyz_derivative_mat @ np.array([
        u_i_local[2],
        r_i_local[1],
        u_j_local[2],
        r_j_local[1]
    ]) / l

    # all deformations
    d_local = np.column_stack((d_x_local, d_y_local, d_z_local))

    # all rotations
    r_local = np.column_stack((r_x_local, r_y_local, r_z_local))

    d_global = (T_local2global @ d_local.T).T

    return d_global, r_local


def interp3D_points(element, d_global, r_local, num_points, scaling):

    element_point_samples = np.column_stack((
        np.linspace(element.node_i.coordinates[0],
                    element.node_j.coordinates[0], num=num_points),
        np.linspace(element.node_i.coordinates[1],
                    element.node_j.coordinates[1], num=num_points),
        np.linspace(element.node_i.coordinates[2],
                    element.node_j.coordinates[2], num=num_points),
    ))

    interpolation_points = element_point_samples + d_global * scaling

    return interpolation_points


def add_data__extruded_frames_deformed_mesh(analysis,
                                            dt,
                                            list_of_frames,
                                            step,
                                            num_points,
                                            scaling):
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
        u_i = analysis.node_displacements[elm.node_i.uniq_id][step][0:3]
        r_i = analysis.node_displacements[elm.node_i.uniq_id][step][3:6]
        u_j = analysis.node_displacements[elm.node_j.uniq_id][step][0:3]
        r_j = analysis.node_displacements[elm.node_j.uniq_id][step][3:6]

        d_global, r_local = interp3D_deformation(
            elm, u_i, r_i, u_j, r_j, num_points)

        interpolation_points = interp3D_points(
            elm, d_global, r_local, num_points, scaling)
        x_vec = elm.local_x_axis_vector()
        y_vec = elm.local_y_axis_vector()
        z_vec = elm.local_z_axis_vector()
        for i in range(num_points-1):
            loc_i_global = interpolation_points[i, :]
            loc_j_global = interpolation_points[i+1, :]
            rot_i_local = r_local[i, :]
            rot_j_local = r_local[i+1, :]

            loop = elm.section.mesh.halfedges
            for halfedge in loop:

                za = halfedge.vertex.coords[0]
                ya = halfedge.vertex.coords[1]
                zb = halfedge.nxt.vertex.coords[0]
                yb = halfedge.nxt.vertex.coords[1]
                defo_ia_global = za * z_vec + ya * y_vec + \
                    (- rot_i_local[2] * ya * x_vec +
                     rot_i_local[1] * za * x_vec
                     + rot_i_local[0] * ya * z_vec
                     - rot_i_local[0] * za * y_vec
                     )*scaling
                defo_ja_global = za * z_vec + ya * y_vec + \
                    (- rot_j_local[2] * ya * x_vec +
                     rot_j_local[1] * za * x_vec
                     + rot_j_local[0] * ya * z_vec
                     - rot_j_local[0] * za * y_vec
                     )*scaling
                defo_ib_global = zb * z_vec + yb * y_vec + \
                    (- rot_i_local[2] * yb * x_vec +
                     rot_i_local[1] * zb * x_vec
                     + rot_i_local[0] * yb * z_vec
                     - rot_i_local[0] * zb * y_vec
                     )*scaling
                defo_jb_global = zb * z_vec + yb * y_vec + \
                    (- rot_j_local[2] * yb * x_vec +
                     rot_i_local[1] * zb * x_vec
                     + rot_j_local[0] * yb * z_vec
                     - rot_j_local[0] * zb * y_vec
                     )*scaling
                loc0 = loc_i_global + defo_ia_global
                loc1 = loc_j_global + defo_ja_global
                loc2 = loc_j_global + defo_jb_global
                loc3 = loc_i_global + defo_ib_global
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
        "hoverinfo": "none",
        "color": common.BEAM_MESH_COLOR,
        "opacity": 0.65
    })


def add_data__nodes_deformed(analysis, dt, list_of_nodes, step, scaling):
    ids = [node.uniq_id for node in list_of_nodes]
    location_data = np.full((len(list_of_nodes), 3), 0.00)
    displacement_data = np.full((len(list_of_nodes), 6), 0.00)
    for i, node in enumerate(list_of_nodes):
        location_data[i, :] = node.coordinates
        displacement_data[i, :] = \
            analysis.node_displacements[node.uniq_id][step]
    r = np.sqrt(displacement_data[:, 0]**2 +
                displacement_data[:, 1]**2 +
                displacement_data[:, 2]**2)
    r = np.reshape(r, (-1, 1))
    ids = np.reshape(np.array(ids), (-1, 1))
    displacement_data = np.concatenate((displacement_data, r, ids), 1)
    dt.append({
        "type": "scatter3d",
        "mode": "markers",
        "x": location_data[:, 0] + displacement_data[:, 0] * scaling,
        "y": location_data[:, 1] + displacement_data[:, 1] * scaling,
        "z": location_data[:, 2] + displacement_data[:, 2] * scaling,
        "customdata": displacement_data,
        "hovertemplate": 'ux: %{customdata[0]:.6g}<br>' +
        'uy: %{customdata[1]:.6g}<br>' +
        'uz: %{customdata[2]:.6g}<br>' +
        'combined: %{customdata[6]:.6g}<br>' +
        'rz: %{customdata[5]:.6g} (rad)<br>' +
        '<extra>Node %{customdata[7]:d}</extra>',
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


def get_auto_scaling(analysis, step):
    """
    Automatically calculate a scaling value that
    makes the maximum displacement appear approximately
    10% of the largest dimention of the building's bounding box
    """
    # building's reference length
    p_min = np.full(3, np.inf)
    p_max = np.full(3, -np.inf)
    for lvl in analysis.building.levels.level_list:
        for node in lvl.nodes.node_list:
            p = np.array(node.coordinates)
            p_min = np.minimum(p_min, p)
            p_max = np.maximum(p_max, p)
    ref_len = np.max(p_max - p_min)
    # maximum displacement
    max_d = 0.00
    for lvl in analysis.building.levels.level_list:
        for elm in lvl.beams.beam_list:
            u_i = analysis.node_displacements[
                elm.node_i.uniq_id][step][0:3]
            r_i = analysis.node_displacements[
                elm.node_i.uniq_id][step][3:6]
            u_j = analysis.node_displacements[
                elm.node_j.uniq_id][step][0:3]
            r_j = analysis.node_displacements[
                elm.node_j.uniq_id][step][3:6]
            d_global, r_local = interp3D_deformation(
                elm, u_i, r_i, u_j, r_j, 3)
            max_d = np.maximum(max_d, np.max(np.abs(d_global)))
    # scaling factor: max_d scaled = 10% of the reference length
    scaling = ref_len / max_d * 0.1
    # never scale things down
    # (usually when this is required, things have gone bad
    #  and we should be able to realize that immediately)
    if scaling < 1.00:
        scaling = 1.00
    return scaling


def deformed_shape(analysis: 'Analysis',
                   step,
                   scaling,
                   extrude_frames):

    # calculate a nice scaling factor if 0.00 is passed
    if scaling == 0:
        scaling = get_auto_scaling(analysis, step)

    layout = common_3D.global_layout()
    dt = []

    list_of_frames = analysis.building.list_of_frames()
    list_of_nodes = analysis.building.list_of_nodes()

    if analysis.building.list_of_master_nodes():
        list_of_nodes.extend(analysis.building.list_of_master_nodes())

    # draw the nodes
    add_data__nodes_deformed(analysis, dt, list_of_nodes, step, scaling)

    # draw the frames
    add_data__extruded_frames_deformed_mesh(
        analysis, dt, list_of_frames, step, 25, scaling)

    fig_datastructure = dict(data=dt, layout=layout)
    fig = go.Figure(fig_datastructure)
    fig.show()

    metadata = {'scaling': scaling}
    return metadata
