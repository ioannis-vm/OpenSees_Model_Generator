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
from utility import transformations


def force_scaling_factor(ref_len, fmax, factor):
    if fmax == 0.00:
        result = 0.00
    else:
        result = ref_len / fmax * factor
    return result


def interp3D_deformation(element, u_i, r_i, u_j, r_j, num_points):
    x_vec = element.x_axis
    y_vec = element.y_axis
    z_vec = element.z_axis

    # global -> local transformation matrix
    T_global2local = \
        transformations.transformation_matrix(x_vec, y_vec, z_vec)
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
    l = element.length_clear()

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
        -u_i_local[2],
        r_i_local[1],
        -u_j_local[2],
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
        np.linspace(element.internal_pt_i[0],
                    element.internal_pt_j[0], num=num_points),
        np.linspace(element.internal_pt_i[1],
                    element.internal_pt_j[1], num=num_points),
        np.linspace(element.internal_pt_i[2],
                    element.internal_pt_j[2], num=num_points),
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

        # translations and rotations at the offset ends
        u_i = analysis.node_displacements[elm.node_i.uniq_id][step][0:3]
        r_i = analysis.node_displacements[elm.node_i.uniq_id][step][3:6]
        u_j = analysis.node_displacements[elm.node_j.uniq_id][step][0:3]
        r_j = analysis.node_displacements[elm.node_j.uniq_id][step][3:6]

        # transferring them to the clear element ends
        offset_i = elm.offset_i
        offset_j = elm.offset_j
        u_i_o = transformations.offset_transformation(offset_i, u_i, r_i)
        u_j_o = transformations.offset_transformation(offset_j, u_j, r_j)

        d_global, r_local = interp3D_deformation(
            elm, u_i_o, r_i, u_j_o, r_j, num_points)

        interpolation_points = interp3D_points(
            elm, d_global, r_local, num_points, scaling)
        x_vec = elm.x_axis
        y_vec = elm.y_axis
        z_vec = elm.z_axis
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
        "hoverinfo": "skip",
        "color": common.BEAM_MESH_COLOR,
        "opacity": 0.65
    })


def add_data__frames_deformed(analysis,
                              dt,
                              list_of_frames,
                              step,
                              num_points,
                              scaling):
    if not list_of_frames:
        return
    x = []
    y = []
    z = []
    for elm in list_of_frames:
        u_i = analysis.node_displacements[elm.node_i.uniq_id][step][0:3]
        r_i = analysis.node_displacements[elm.node_i.uniq_id][step][3:6]
        u_j = analysis.node_displacements[elm.node_j.uniq_id][step][0:3]
        r_j = analysis.node_displacements[elm.node_j.uniq_id][step][3:6]
        # transferring them to the clear element ends
        offset_i = elm.offset_i
        offset_j = elm.offset_j
        u_i_o = transformations.offset_transformation(offset_i, u_i, r_i)
        u_j_o = transformations.offset_transformation(offset_j, u_j, r_j)
        d_global, r_local = interp3D_deformation(
            elm, u_i_o, r_i, u_j_o, r_j, num_points)
        interpolation_points = interp3D_points(
            elm, d_global, r_local, num_points, scaling)
        for i in range(len(interpolation_points)-1):
            x.extend((interpolation_points[i, 0],
                      interpolation_points[i+1, 0],
                      None))
            y.extend((interpolation_points[i, 1],
                      interpolation_points[i+1, 1],
                      None))
            z.extend((interpolation_points[i, 2],
                      interpolation_points[i+1, 2],
                      None))

    dt.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": x,
        "y": y,
        "z": z,
        "hoverinfo": "skip",
        "line": {
            "width": 5,
            "color": common.BEAM_MESH_COLOR
        }
    })


def add_data__frames_offsets_deformed(analysis,
                                      dt,
                                      list_of_beamcolumn_elems,
                                      step,
                                      scaling):
    if not list_of_beamcolumn_elems:
        return
    x = []
    y = []
    z = []
    for elm in list_of_beamcolumn_elems:
        p_i = np.array(elm.node_i.coords)
        p_io = np.array(elm.internal_pt_i)
        offset_i = elm.offset_i
        u_i = np.array(
            analysis.node_displacements[elm.node_i.uniq_id][step][0:3])
        r_i = np.array(
            analysis.node_displacements[elm.node_i.uniq_id][step][3:6])
        u_io = transformations.offset_transformation(offset_i, u_i, r_i)

        p_j = np.array(elm.node_j.coords)
        p_jo = np.array(elm.internal_pt_j)
        offset_j = elm.offset_j
        u_j = np.array(
            analysis.node_displacements[elm.node_j.uniq_id][step][0:3])
        r_j = np.array(
            analysis.node_displacements[elm.node_j.uniq_id][step][3:6])
        u_jo = transformations.offset_transformation(offset_j, u_j, r_j)

        x_i = p_i + u_i * scaling
        x_io = p_io + u_io * scaling
        x_j = p_j + u_j * scaling
        x_jo = p_jo + u_jo * scaling

        x.extend((x_i[0], x_io[0], None))
        y.extend((x_i[1], x_io[1], None))
        z.extend((x_i[2], x_io[2], None))
        x.extend((x_j[0], x_jo[0], None))
        y.extend((x_j[1], x_jo[1], None))
        z.extend((x_j[2], x_jo[2], None))

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


def add_data__frames_undeformed(dt, list_of_frames):
    x = []
    y = []
    z = []
    for elm in list_of_frames:
        x.extend(
            (elm.internal_pt_i[0], elm.internal_pt_j[0], None)
        )
        y.extend(
            (elm.internal_pt_i[1], elm.internal_pt_j[1], None)
        )
        z.extend(
            (elm.internal_pt_i[2], elm.internal_pt_j[2], None)
        )
        dt.append({
            "type": "scatter3d",
            "mode": "lines",
            "x": x,
            "y": y,
            "z": z,
            "hoverinfo": "skip",
            "line": {
                "width": 5,
                "color": common.BEAM_MESH_COLOR
            }
        })


def add_data__nodes_deformed(analysis, dt, list_of_nodes, step, scaling):
    ids = [node.uniq_id for node in list_of_nodes]
    location_data = np.full((len(list_of_nodes), 3), 0.00)
    displacement_data = np.full((len(list_of_nodes), 6), 0.00)
    for i, node in enumerate(list_of_nodes):
        location_data[i, :] = node.coords
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
        'rx: %{customdata[3]:.6g} (rad)<br>' +
        'ry: %{customdata[4]:.6g} (rad)<br>' +
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


def add_data__nodes_undeformed(dt, list_of_nodes):
    x = [node.coords[0] for node in list_of_nodes]
    y = [node.coords[1] for node in list_of_nodes]
    z = [node.coords[2] for node in list_of_nodes]

    dt.append({
        "type": "scatter3d",
        "mode": "markers",
        "x": x,
        "y": y,
        "z": z,
        "hoverinfo": "skip",
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


def get_auto_scaling_deformation(analysis, step):
    """
    Automatically calculate a scaling value that
    makes the maximum displacement appear approximately
    10% of the largest dimention of the building's bounding box
    """
    ref_len = analysis.building.reference_length()
    # maximum displacement
    max_d = 0.00
    for elm in analysis.building.list_of_internal_elems():
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
    if max_d > 1.00e-14:
        scaling = ref_len / max_d * 0.1
    else:
        # no infinite scaling, thank you
        scaling = 1.00

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
        scaling = get_auto_scaling_deformation(analysis, step)

    layout = common_3D.global_layout()
    dt = []

    list_of_frames = \
        analysis.building.list_of_internal_elems()
    list_of_nodes = analysis.building.list_of_primary_nodes() + \
        analysis.building.list_of_internal_nodes()
    list_of_parent_nodes = analysis.building.list_of_parent_nodes()

    if list_of_parent_nodes:
        list_of_nodes.extend(analysis.building.list_of_parent_nodes())

    # draw the nodes
    add_data__nodes_deformed(analysis, dt, list_of_nodes, step, scaling)

    # draw the frames
    add_data__frames_offsets_deformed(
        analysis, dt, list_of_frames, step, scaling)
    if extrude_frames:
        add_data__extruded_frames_deformed_mesh(
            analysis, dt, list_of_frames, step, 15, scaling)
    else:
        add_data__frames_deformed(
            analysis, dt, list_of_frames, step, 15, scaling)

    fig_datastructure = dict(data=dt, layout=layout)
    fig = go.Figure(fig_datastructure)
    fig.show()

    metadata = {'scaling': scaling}
    return metadata


def basic_forces(analysis: 'Analysis',
                 step,
                 scaling_n,
                 scaling_q,
                 scaling_m,
                 scaling_t,
                 num_points):

    layout = common_3D.global_layout()
    dt = []

    list_of_frames = analysis.building.list_of_internal_elems()
    list_of_nodes = analysis.building.list_of_all_nodes()

    # draw the nodes0
    add_data__nodes_undeformed(dt, list_of_nodes)
    # draw the frames
    add_data__frames_undeformed(
        dt, list_of_frames)

    # For the main lines: 1
    # For the fill lines: 2
    # Plot options:
    # a: axial + torsion
    # b: shear in local Y and Z
    # c: moment in Y and Z
    # d: shear combined
    # e: moment combined
    x1_a = []
    y1_a = []
    z1_a = []
    colors1_a = []
    customdata_a = []
    x2_a = []
    y2_a = []
    z2_a = []
    colors2_a = []
    x1_b = []
    y1_b = []
    z1_b = []
    colors1_b = []
    customdata_b = []
    x2_b = []
    y2_b = []
    z2_b = []
    colors2_b = []
    x1_c = []
    y1_c = []
    z1_c = []
    colors1_c = []
    customdata_c = []
    x2_c = []
    y2_c = []
    z2_c = []
    colors2_c = []
    x1_d = []
    y1_d = []
    z1_d = []
    colors1_d = []
    customdata_d = []
    x2_d = []
    y2_d = []
    z2_d = []
    colors2_d = []
    x1_e = []
    y1_e = []
    z1_e = []
    colors1_e = []
    customdata_e = []
    x2_e = []
    y2_e = []
    z2_e = []
    colors2_e = []

    # preallocate memory
    #
    # (we do this to determine the internal forces
    #  for all elements before we even start plotting
    #  them, to be able to compute a nice scaling factor
    #  without having to then recalculate the basic forces)
    # (We store the discretized basic force vectors in a
    #  linear fashion, element-wise)
    num_elems = len(analysis.building.list_of_internal_elems())
    nx_vecs = np.full(num_elems * num_points, 0.00)
    qy_vecs = np.full(num_elems * num_points, 0.00)
    qz_vecs = np.full(num_elems * num_points, 0.00)
    tx_vecs = np.full(num_elems * num_points, 0.00)
    mz_vecs = np.full(num_elems * num_points, 0.00)
    my_vecs = np.full(num_elems * num_points, 0.00)
    x_vecs = np.full(num_elems * 3, 0.00)
    y_vecs = np.full(num_elems * 3, 0.00)
    z_vecs = np.full(num_elems * 3, 0.00)
    i_poss = np.full(num_elems * 3, 0.00)
    elm_ln = np.full(num_elems, 0.00)

    for i_elem, element in enumerate(analysis.building.list_of_internal_elems()):

        x_vec = element.x_axis
        y_vec = element.y_axis
        z_vec = element.z_axis

        i_pos = np.array(element.internal_pt_i)

        T_global2local = np.vstack((x_vec, y_vec, z_vec))

        forces_global = analysis.eleForces[
            element.uniq_id][step][0:3]
        moments_global_ends = analysis.eleForces[
            element.uniq_id][step][3:6]

        moments_global_clear = transformations.offset_transformation(
            element.offset_i, moments_global_ends, forces_global)

        ni, qyi, qzi = T_global2local @ forces_global
        ti, myi, mzi = T_global2local @ moments_global_clear

        wx, wy, wz = element.udl

        l = element.length_clear()
        t = np.linspace(0.00, l, num=num_points)

        nx_vec = - t * wx - ni
        qy_vec = t * wy + qyi
        qz_vec = t * wz + qzi
        tx_vec = np.full(num_points, -ti)
        mz_vec = t**2 * 0.50 * wy + t * qyi - mzi
        my_vec = t**2 * 0.50 * wz + t * qzi + myi

        # store results in the preallocated arrays
        nx_vecs[i_elem*num_points:i_elem*num_points+num_points] = nx_vec
        qy_vecs[i_elem*num_points:i_elem*num_points+num_points] = qy_vec
        qz_vecs[i_elem*num_points:i_elem*num_points+num_points] = qz_vec
        tx_vecs[i_elem*num_points:i_elem*num_points+num_points] = tx_vec
        my_vecs[i_elem*num_points:i_elem*num_points+num_points] = my_vec
        mz_vecs[i_elem*num_points:i_elem*num_points+num_points] = mz_vec
        x_vecs[i_elem*3: i_elem*3 + 3] = x_vec
        y_vecs[i_elem*3: i_elem*3 + 3] = y_vec
        z_vecs[i_elem*3: i_elem*3 + 3] = z_vec
        i_poss[i_elem*3: i_elem*3 + 3] = i_pos
        elm_ln[i_elem] = l

    # calculate scaling factors
    ref_len = analysis.building.reference_length()
    factor = 0.05
    nx_max = np.max(np.abs(nx_vecs))
    scaling_n = force_scaling_factor(ref_len, nx_max, factor)
    if scaling_n > 1.e8:
        scaling_t = 1.00
    qy_max = np.max(np.abs(qy_vecs))
    qz_max = np.max(np.abs(qz_vecs))
    scaling_qy = force_scaling_factor(ref_len, qy_max, factor)
    scaling_qz = force_scaling_factor(ref_len, qz_max, factor)
    if (scaling_qy > 0.00 and scaling_qz > 0.00):
        scaling_q = np.minimum(scaling_qy, scaling_qz)
    elif scaling_qy == 0.00:
        scaling_q = scaling_qz
    elif scaling_qz == 0.00:
        scaling_q = scaling_qy
    else:
        scaling_q = 0.00
    if scaling_q > 1.0e8:
        scaling_q = 1.00
    my_max = np.max(np.abs(my_vecs))
    mz_max = np.max(np.abs(mz_vecs))
    scaling_my = force_scaling_factor(ref_len, my_max, factor)
    scaling_mz = force_scaling_factor(ref_len, mz_max, factor)
    if (scaling_my > 0.00 and scaling_mz > 0.00):
        scaling_m = np.minimum(scaling_my, scaling_mz)
    elif scaling_my == 0.00:
        scaling_m = scaling_mz
    elif scaling_mz == 0.00:
        scaling_m = scaling_my
    else:
        scaling_m = 0.00
    if scaling_m > 1.0e8:
        scaling_m = 1.00

    for i_elem, element in enumerate(analysis.building.list_of_internal_elems()):

        # retrieve results from the preallocated arrays
        nx_vec = nx_vecs[i_elem*num_points:i_elem*num_points+num_points]
        qy_vec = qy_vecs[i_elem*num_points:i_elem*num_points+num_points]
        qz_vec = qz_vecs[i_elem*num_points:i_elem*num_points+num_points]
        tx_vec = tx_vecs[i_elem*num_points:i_elem*num_points+num_points]
        my_vec = my_vecs[i_elem*num_points:i_elem*num_points+num_points]
        mz_vec = mz_vecs[i_elem*num_points:i_elem*num_points+num_points]
        x_vec = x_vecs[i_elem*3: i_elem*3 + 3]
        y_vec = y_vecs[i_elem*3: i_elem*3 + 3]
        z_vec = z_vecs[i_elem*3: i_elem*3 + 3]
        i_pos = i_poss[i_elem*3: i_elem*3 + 3]
        l = elm_ln[i_elem]
        t = np.linspace(0.00, l, num=num_points)

        for i in range(num_points - 1):

            p_start = i_pos + t[i] * x_vec
            p_end = i_pos + t[i+1] * x_vec

            # axial load & torsion
            p_i = p_start + nx_vec[i] * y_vec * scaling_n
            p_j = p_end + nx_vec[i+1] * y_vec * scaling_n
            x1_a.extend((p_i[0], p_j[0], None))
            y1_a.extend((p_i[1], p_j[1], None))
            z1_a.extend((p_i[2], p_j[2], None))
            customdata_a.extend(
                (nx_vec[i], nx_vec[i+1], None))
            x2_a.extend((p_start[0], p_i[0], None))
            x2_a.extend((p_j[0], p_end[0], None))
            y2_a.extend((p_start[1], p_i[1], None))
            y2_a.extend((p_j[1], p_end[1], None))
            z2_a.extend((p_start[2], p_i[2], None))
            z2_a.extend((p_j[2], p_end[2], None))
            colors1_a.extend(["red"]*3)
            colors2_a.extend(["red"]*6)
            p_i = p_start + tx_vec[i] * z_vec * scaling_t
            p_j = p_end + tx_vec[i+1] * z_vec * scaling_t
            x1_a.extend((p_i[0], p_j[0], None))
            y1_a.extend((p_i[1], p_j[1], None))
            z1_a.extend((p_i[2], p_j[2], None))
            customdata_a.extend(
                (tx_vec[i], tx_vec[i+1], None))
            x2_a.extend((p_start[0], p_i[0], None))
            x2_a.extend((p_j[0], p_end[0], None))
            y2_a.extend((p_start[1], p_i[1], None))
            y2_a.extend((p_j[1], p_end[1], None))
            z2_a.extend((p_start[2], p_i[2], None))
            z2_a.extend((p_j[2], p_end[2], None))
            colors1_a.extend(["orange"]*3)
            colors2_a.extend(["orange"]*6)

            # shear load on y and z axes
            p_i = p_start + qy_vec[i] * y_vec * scaling_q
            p_j = p_end + qy_vec[i+1] * y_vec * scaling_q
            x1_b.extend((p_i[0], p_j[0], None))
            y1_b.extend((p_i[1], p_j[1], None))
            z1_b.extend((p_i[2], p_j[2], None))
            customdata_b.extend(
                (qy_vec[i], qy_vec[i+1], None))
            x2_b.extend((p_start[0], p_i[0], None))
            x2_b.extend((p_j[0], p_end[0], None))
            y2_b.extend((p_start[1], p_i[1], None))
            y2_b.extend((p_j[1], p_end[1], None))
            z2_b.extend((p_start[2], p_i[2], None))
            z2_b.extend((p_j[2], p_end[2], None))
            colors1_b.extend(["green"]*3)
            colors2_b.extend(["green"]*6)
            p_i = p_start + qz_vec[i] * z_vec * scaling_q
            p_j = p_end + qz_vec[i+1] * z_vec * scaling_q
            x1_b.extend((p_i[0], p_j[0], None))
            y1_b.extend((p_i[1], p_j[1], None))
            z1_b.extend((p_i[2], p_j[2], None))
            customdata_b.extend(
                (qz_vec[i], qz_vec[i+1], None))
            x2_b.extend((p_start[0], p_i[0], None))
            x2_b.extend((p_j[0], p_end[0], None))
            y2_b.extend((p_start[1], p_i[1], None))
            y2_b.extend((p_j[1], p_end[1], None))
            z2_b.extend((p_start[2], p_i[2], None))
            z2_b.extend((p_j[2], p_end[2], None))
            colors1_b.extend(["green"]*3)
            colors2_b.extend(["green"]*6)

            # moment around z and y axes
            p_i = p_start - mz_vec[i] * y_vec * scaling_m
            p_j = p_end - mz_vec[i+1] * y_vec * scaling_m
            # note: moments plotted upside down!
            x1_c.extend((p_i[0], p_j[0], None))
            y1_c.extend((p_i[1], p_j[1], None))
            z1_c.extend((p_i[2], p_j[2], None))
            customdata_c.extend(
                (mz_vec[i], mz_vec[i+1], None))
            x2_c.extend((p_start[0], p_i[0], None))
            x2_c.extend((p_j[0], p_end[0], None))
            y2_c.extend((p_start[1], p_i[1], None))
            y2_c.extend((p_j[1], p_end[1], None))
            z2_c.extend((p_start[2], p_i[2], None))
            z2_c.extend((p_j[2], p_end[2], None))
            colors1_c.extend(["blue"]*3)
            colors2_c.extend(["blue"]*6)
            # moment around y axis
            p_i = p_start - my_vec[i] * z_vec * scaling_m
            p_j = p_end - my_vec[i+1] * z_vec * scaling_m
            x1_c.extend((p_i[0], p_j[0], None))
            y1_c.extend((p_i[1], p_j[1], None))
            z1_c.extend((p_i[2], p_j[2], None))
            customdata_c.extend(
                (my_vec[i], my_vec[i+1], None))
            x2_c.extend((p_start[0], p_i[0], None))
            x2_c.extend((p_j[0], p_end[0], None))
            y2_c.extend((p_start[1], p_i[1], None))
            y2_c.extend((p_j[1], p_end[1], None))
            z2_c.extend((p_start[2], p_i[2], None))
            z2_c.extend((p_j[2], p_end[2], None))
            colors1_c.extend(["blue"]*3)
            colors2_c.extend(["blue"]*6)

            # shear load combined
            p_i = p_start + (qy_vec[i] * y_vec +
                             qz_vec[i] * z_vec) * scaling_q
            p_j = p_end + (qy_vec[i+1] * y_vec +
                           qz_vec[i+1] * z_vec) * scaling_q
            x1_d.extend((p_i[0], p_j[0], None))
            y1_d.extend((p_i[1], p_j[1], None))
            z1_d.extend((p_i[2], p_j[2], None))
            customdata_d.extend(
                (np.sqrt(qy_vec[i]**2 + qz_vec[i]**2),
                 np.sqrt(qy_vec[i+1]**2 + qz_vec[i+1]**2), None))
            x2_d.extend((p_start[0], p_i[0], None))
            x2_d.extend((p_j[0], p_end[0], None))
            y2_d.extend((p_start[1], p_i[1], None))
            y2_d.extend((p_j[1], p_end[1], None))
            z2_d.extend((p_start[2], p_i[2], None))
            z2_d.extend((p_j[2], p_end[2], None))
            colors1_d.extend(["green"]*3)
            colors2_d.extend(["green"]*6)

            # both moments combined!
            p_i = p_start - mz_vec[i] * y_vec * \
                scaling_m - my_vec[i] * z_vec * scaling_m
            p_j = p_end - mz_vec[i+1] * y_vec * \
                scaling_m - my_vec[i+1] * z_vec * scaling_m
            # note: moments plotted upside down!
            x1_e.extend((p_i[0], p_j[0], None))
            y1_e.extend((p_i[1], p_j[1], None))
            z1_e.extend((p_i[2], p_j[2], None))
            customdata_e.extend(
                (np.sqrt(mz_vec[i]**2 + my_vec[i]**2),
                 np.sqrt(mz_vec[i+1]**2 + my_vec[i+1]**2),
                 None)
            )
            x2_e.extend((p_start[0], p_i[0], None))
            x2_e.extend((p_j[0], p_end[0], None))
            y2_e.extend((p_start[1], p_i[1], None))
            y2_e.extend((p_j[1], p_end[1], None))
            z2_e.extend((p_start[2], p_i[2], None))
            z2_e.extend((p_j[2], p_end[2], None))
            colors1_e.extend(["blue"]*3)
            colors2_e.extend(["blue"]*6)

    # TODO validation: make sure they are pollted correctly
    dt_a = [
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x1_a,
            "y": y1_a,
            "z": z1_a,
            "visible": False,
            "customdata": customdata_a,
            "hovertemplate": ' %{customdata:.3g}<br>'
            '<extra></extra>',
            "line": {
                "width": 3,
                "color": colors1_a
            }
        },
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x2_a,
            "y": y2_a,
            "z": z2_a,
            "visible": False,
            "hoverinfo": "skip",
            "line": {
                "width": 1,
                "color": colors2_a
            }
        }
    ]
    dt_b = [
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x1_b,
            "y": y1_b,
            "z": z1_b,
            "visible": False,
            "customdata": customdata_b,
            "hovertemplate": ' %{customdata:.3g}<br>'
            '<extra></extra>',
            "line": {
                "width": 3,
                "color": colors1_b
            }
        },
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x2_b,
            "y": y2_b,
            "z": z2_b,
            "visible": False,
            "hoverinfo": "skip",
            "line": {
                "width": 1,
                "color": colors2_b
            }
        }
    ]
    dt_c = [
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x1_c,
            "y": y1_c,
            "z": z1_c,
            "visible": False,
            "customdata": customdata_c,
            "hovertemplate": ' %{customdata:.3g}<br>'
            '<extra></extra>',
            "line": {
                "width": 3,
                "color": colors1_c
            }
        },
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x2_c,
            "y": y2_c,
            "z": z2_c,
            "visible": False,
            "hoverinfo": "skip",
            "line": {
                "width": 1,
                "color": colors2_c
            }
        }
    ]
    dt_d = [
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x1_d,
            "y": y1_d,
            "z": z1_d,
            "visible": False,
            "customdata": customdata_d,
            "hovertemplate": ' %{customdata:.3g}<br>'
            '<extra></extra>',
            "line": {
                "width": 3,
                "color": colors1_d
            }
        },
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x2_d,
            "y": y2_d,
            "z": z2_d,
            "visible": False,
            "hoverinfo": "skip",
            "line": {
                "width": 1,
                "color": colors2_d
            }
        }
    ]
    dt_e = [
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x1_e,
            "y": y1_e,
            "z": z1_e,
            "visible": False,
            "customdata": customdata_e,
            "hovertemplate": ' %{customdata:.3g}<br>'
            '<extra></extra>',
            "line": {
                "width": 3,
                "color": colors1_e
            }
        },
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x2_e,
            "y": y2_e,
            "z": z2_e,
            "visible": False,
            "hoverinfo": "skip",
            "line": {
                "width": 1,
                "color": colors2_e
            }
        }
    ]

    fig_datastructure = dict(data=dt + dt_a +
                             dt_b + dt_c + dt_d + dt_e,
                             layout=layout)
    fig = go.Figure(fig_datastructure)
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="None",
                        method="update",
                        args=[{"visible":
                               [True]*len(dt)+[False]*2*5
                               }]
                    ),
                    dict(
                        label="Axial+Torsion",
                        method="update",
                        args=[{"visible":
                               [True]*len(dt)+[True]*2+[False]*2*4
                               }]
                    ),
                    dict(
                        label="Shear",
                        method="update",
                        args=[{"visible":
                               [True]*len(dt)+[False]*2+[True]*2+[False]*2*3
                               }]
                    ),
                    dict(
                        label="Moment",
                        method="update",
                        args=[{"visible":
                               [True]*len(dt)+[False]*2*2+[True]*2+[False]*2*2
                               }]
                    ),
                    dict(
                        label="Shear (combined)",
                        method="update",
                        args=[{"visible":
                               [True]*len(dt)+[False]*2*3+[True]*2+[False]*2
                               }]
                    ),
                    dict(
                        label="Moment (combined)",
                        method="update",
                        args=[{"visible":
                               [True]*len(dt)+[False]*2*4+[True]*2
                               }]
                    )
                ]
            )
        ]
    )

    fig.show()

    metadata = {'scaling_n': scaling_n,
                'scaling_q': scaling_q,
                'scaling_m': scaling_m,
                'scaling_t': scaling_t}
    return metadata
