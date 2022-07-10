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

import sys
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go  # type: ignore
from .. import transformations
from . import graphics_common
from . import graphics_common_3D
from .preprocessing_3D import add_data__global_axes


def force_scaling_factor(ref_len, fmax, factor):
    if fmax == 0.00:
        result = 0.00
    else:
        result = ref_len / fmax * factor
    return result


def interp3D_deformation(element, u_i, r_i, u_j, r_j, num_points):
    """
    Given the deformations of the ends of a Bernoulli beam,
    use its shape functions to obtain intermediate points.
    Args:
        element ('model.LineElement'): A line element
        u_i (npt.NDArray[np.float64]): 3 displacements at end i, global system
        r_i (npt.NDArray[np.float64]): 3 rotations at end i, global system
        u_j, r_j: similar to u_i, r_i.
        num_points: Number of interpolation points
    See note: https://notability.com/n/0wlJ17mt81uuVWAYVoFfV3
    Returns:
        d_global (npt.NDArray[np.float64]): Displacements (global system)
        r_local (npt.NDArray[np.float64]): Rotations (local system)
        (the rotations are needed for plotting the
         deformed shape with extruded frame elements)
    """

    x_vec = element.geomtransf.x_axis
    y_vec = element.geomtransf.y_axis
    z_vec = np.cross(x_vec, y_vec)

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
    p_i = np.array(element.eleNodes[0].coords) + element.geomtransf.offset_i
    p_j = np.array(element.eleNodes[1].coords) + element.geomtransf.offset_j
    len_clr = np.linalg.norm(p_i - p_j)

    # shape function matrices
    Nx_mat = np.column_stack((
        1. - t,
        t
    ))
    Nyz_mat = np.column_stack((
        1. - 3. * t**2 + 2. * t**3,
        (t - 2. * t**2 + t**3) * len_clr,
        3. * t**2 - 2. * t**3,
        (-t**2 + t**3) * len_clr
    ))
    Nyz_derivative_mat = np.column_stack((
        - 6. * t + 6. * t**2,
        (1 - 4. * t + 3. * t**2) * len_clr,
        6. * t - 6. * t**2,
        (-2. * t + 3. * t**2) * len_clr
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
    ]) / len_clr

    # bending rotation around the local y axis
    r_y_local = Nyz_derivative_mat @ np.array([
        -u_i_local[2],
        r_i_local[1],
        -u_j_local[2],
        r_j_local[1]
    ]) / len_clr

    # all deformations
    d_local = np.column_stack((d_x_local, d_y_local, d_z_local))

    # all rotations
    r_local = np.column_stack((r_x_local, r_y_local, r_z_local))

    d_global = (T_local2global @ d_local.T).T

    return d_global, r_local


def interp3D_points(element, d_global, r_local, num_points, scaling):

    p_i = np.array(element.eleNodes[0].coords) + element.geomtransf.offset_i
    p_j = np.array(element.eleNodes[1].coords) + element.geomtransf.offset_j
    element_point_samples = np.column_stack((
        np.linspace(p_i[0], p_j[0], num=num_points),
        np.linspace(p_i[1], p_j[1], num=num_points),
        np.linspace(p_i[2], p_j[2], num=num_points),
    ))

    interpolation_points = element_point_samples + d_global * scaling

    return interpolation_points


def add_data__extruded_frames_deformed_mesh(analysis,
                                            dt,
                                            list_of_frames,
                                            step,
                                            scaling):
    if not list_of_frames:
        return
    x_list = []
    y_list = []
    z_list = []
    i_list = []
    j_list = []
    k_list = []
    intensity = []
    index = 0
    for elm in list_of_frames:
        if elm.visibility.hidden_when_extruded:
            continue
        num_points = 8
        # translations and rotations at the offset ends
        u_i = analysis.results.node_displacements.registry[elm.eleNodes[0].uid][step][0:3]
        r_i = analysis.results.node_displacements.registry[elm.eleNodes[0].uid][step][3:6]
        u_j = analysis.results.node_displacements.registry[elm.eleNodes[1].uid][step][0:3]
        r_j = analysis.results.node_displacements.registry[elm.eleNodes[1].uid][step][3:6]
        # transferring them to the clear element ends
        offset_i = elm.geomtransf.offset_i
        offset_j = elm.geomtransf.offset_j
        u_i_o = transformations.offset_transformation(offset_i, u_i, r_i)
        u_j_o = transformations.offset_transformation(offset_j, u_j, r_j)
        d_global, r_local = interp3D_deformation(
            elm, u_i_o, r_i, u_j_o, r_j, num_points)
        interpolation_points = interp3D_points(
            elm, d_global, r_local, num_points, scaling)
        x_vec = elm.geomtransf.x_axis
        y_vec = elm.geomtransf.y_axis
        z_vec = elm.geomtransf.z_axis
        for i in range(num_points-1):
            loc_i_global = interpolation_points[i, :]
            loc_j_global = interpolation_points[i+1, :]
            rot_i_local = r_local[i, :]
            rot_j_local = r_local[i+1, :]

            loop = elm.section.outside_shape.halfedges
            for halfedge in loop:

                z_a = halfedge.vertex.coords[0]
                y_a = halfedge.vertex.coords[1]
                z_b = halfedge.nxt.vertex.coords[0]
                y_b = halfedge.nxt.vertex.coords[1]
                defo_ia_global = z_a * z_vec + y_a * y_vec + \
                    (- rot_i_local[2] * y_a * x_vec +
                     rot_i_local[1] * z_a * x_vec
                     + rot_i_local[0] * y_a * z_vec
                     - rot_i_local[0] * z_a * y_vec
                     )*scaling
                defo_ja_global = z_a * z_vec + y_a * y_vec + \
                    (- rot_j_local[2] * y_a * x_vec +
                     rot_j_local[1] * z_a * x_vec
                     + rot_j_local[0] * y_a * z_vec
                     - rot_j_local[0] * z_a * y_vec
                     )*scaling
                defo_ib_global = z_b * z_vec + y_b * y_vec + \
                    (- rot_i_local[2] * y_b * x_vec +
                     rot_i_local[1] * z_b * x_vec
                     + rot_i_local[0] * y_b * z_vec
                     - rot_i_local[0] * z_b * y_vec
                     )*scaling
                defo_jb_global = z_b * z_vec + y_b * y_vec + \
                    (- rot_j_local[2] * y_b * x_vec +
                     rot_i_local[1] * z_b * x_vec
                     + rot_j_local[0] * y_b * z_vec
                     - rot_j_local[0] * z_b * y_vec
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
                intensity.append(np.linalg.norm(d_global[i, :]))
                intensity.append(np.linalg.norm(d_global[i+1, :]))
                intensity.append(np.linalg.norm(d_global[i+1, :]))
                intensity.append(np.linalg.norm(d_global[i, :]))
                index += 4
    dt.append({
        "type": "mesh3d",
        "x": x_list,
        "y": y_list,
        "z": z_list,
        "colorscale": [[0, 'blue'], [1.0, 'red']],
        "i": i_list,
        "j": j_list,
        "k": k_list,
        "intensity": intensity,
        "colorbar_title": 'Displacement',
        "hoverinfo": "skip",
        "opacity": 0.65
    })


# def add_data__extruded_steel_W_PZ_deformed_mesh(
#         analysis, dt, list_of_endsegments, step, scaling):

#     # determine the mesh3d dictionary
#     # to append components to it
#     # instead of creating a new mesh3d
#     # (to have a consistent colorscale)
#     idx = 0
#     for k, item in enumerate(dt):
#         if item['type'] == 'mesh3d':
#             idx = k
#             break

#     imax = max(dt[idx]['i'])
#     jmax = max(dt[idx]['j'])
#     kmax = max(dt[idx]['k'])
#     ijkmax = max((imax, jmax, kmax))

#     index = ijkmax + 1

#     if not list_of_endsegments:
#         return
#     x_list = []
#     y_list = []
#     z_list = []
#     i_list = []
#     j_list = []
#     k_list = []
#     intensity = []

#     for elm in list_of_endsegments:

#         num_points = 2
#         # translations and rotations at the offset ends
#         u_i = analysis.node_displacements[str(elm.n_external.uid)][step][0:3]
#         r_i = analysis.node_displacements[str(elm.n_external.uid)][step][3:6]
#         u_j = analysis.node_displacements[str(elm.n_internal.uid)][step][0:3]
#         r_j = analysis.node_displacements[str(elm.n_internal.uid)][step][3:6]

#         d_global, r_local = interp3D_deformation(
#             elm, u_i, r_i, u_j, r_j, num_points)

#         interpolation_points = interp3D_points(
#             elm, d_global, r_local, num_points, scaling)
#         x_vec = elm.parent.x_axis
#         y_vec = elm.parent.y_axis
#         z_vec = np.cross(x_vec, y_vec)
#         for i in range(num_points-1):
#             loc_i_global = interpolation_points[i, :]
#             loc_j_global = interpolation_points[i+1, :]
#             rot_i_local = r_local[i, :]
#             rot_j_local = r_local[i+1, :]

#             loop = elm.parent.section.mesh.halfedges
#             for halfedge in loop:

#                 z_a = halfedge.vertex.coords[0]
#                 y_a = halfedge.vertex.coords[1]
#                 z_b = halfedge.nxt.vertex.coords[0]
#                 y_b = halfedge.nxt.vertex.coords[1]
#                 defo_ia_global = z_a * z_vec + y_a * y_vec + \
#                     (- rot_i_local[2] * y_a * x_vec +
#                      rot_i_local[1] * z_a * x_vec
#                      + rot_i_local[0] * y_a * z_vec
#                      - rot_i_local[0] * z_a * y_vec
#                      )*scaling
#                 defo_ja_global = z_a * z_vec + y_a * y_vec + \
#                     (- rot_j_local[2] * y_a * x_vec +
#                      rot_j_local[1] * z_a * x_vec
#                      + rot_j_local[0] * y_a * z_vec
#                      - rot_j_local[0] * z_a * y_vec
#                      )*scaling
#                 defo_ib_global = z_b * z_vec + y_b * y_vec + \
#                     (- rot_i_local[2] * y_b * x_vec +
#                      rot_i_local[1] * z_b * x_vec
#                      + rot_i_local[0] * y_b * z_vec
#                      - rot_i_local[0] * z_b * y_vec
#                      )*scaling
#                 defo_jb_global = z_b * z_vec + y_b * y_vec + \
#                     (- rot_j_local[2] * y_b * x_vec +
#                      rot_i_local[1] * z_b * x_vec
#                      + rot_j_local[0] * y_b * z_vec
#                      - rot_j_local[0] * z_b * y_vec
#                      )*scaling
#                 loc0 = loc_i_global + defo_ia_global
#                 loc1 = loc_j_global + defo_ja_global
#                 loc2 = loc_j_global + defo_jb_global
#                 loc3 = loc_i_global + defo_ib_global
#                 x_list.append(loc0[0])
#                 y_list.append(loc0[1])
#                 z_list.append(loc0[2])
#                 x_list.append(loc1[0])
#                 y_list.append(loc1[1])
#                 z_list.append(loc1[2])
#                 x_list.append(loc2[0])
#                 y_list.append(loc2[1])
#                 z_list.append(loc2[2])
#                 x_list.append(loc3[0])
#                 y_list.append(loc3[1])
#                 z_list.append(loc3[2])
#                 i_list.append(index + 0)
#                 j_list.append(index + 1)
#                 k_list.append(index + 2)
#                 i_list.append(index + 0)
#                 j_list.append(index + 2)
#                 k_list.append(index + 3)
#                 intensity.append(np.linalg.norm(d_global[0, :]))
#                 intensity.append(np.linalg.norm(d_global[1, :]))
#                 intensity.append(np.linalg.norm(d_global[1, :]))
#                 intensity.append(np.linalg.norm(d_global[0, :]))
#                 index += 4

#     dt[idx]['x'].extend(x_list)
#     dt[idx]['y'].extend(y_list)
#     dt[idx]['z'].extend(z_list)
#     dt[idx]['i'].extend(i_list)
#     dt[idx]['j'].extend(j_list)
#     dt[idx]['k'].extend(k_list)
#     dt[idx]['intensity'].extend(intensity)


def add_data__frames_deformed(analysis,
                              dt,
                              list_of_frames,
                              step,
                              sub,
                              scaling):
    if not list_of_frames:
        return
    x = []
    y = []
    z = []
    
    for elm in list_of_frames:
        if elm.visibility.hidden_at_line_plots:
            continue
        num_points = 8
        u_i = analysis.results.node_displacements.registry[elm.eleNodes[0].uid][step][0:3]
        r_i = analysis.results.node_displacements.registry[elm.eleNodes[0].uid][step][3:6]
        u_j = analysis.results.node_displacements.registry[elm.eleNodes[1].uid][step][0:3]
        r_j = analysis.results.node_displacements.registry[elm.eleNodes[1].uid][step][3:6]
        # transferring them to the clear element ends
        offset_i = elm.geomtransf.offset_i
        offset_j = elm.geomtransf.offset_j
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
            "color": graphics_common.BEAM_MESH_COLOR
        }
    })


def add_data__frames_offsets_deformed(analysis,
                                      dt,
                                      list_of_frames,
                                      step,
                                      scaling):
    if not list_of_frames:
        return
    x = []
    y = []
    z = []
    for elm in list_of_frames:
        if np.array_equal(elm.geomtransf.offset_i, np.zeros(3)):
            if np.array_equal(elm.geomtransf.offset_j, np.zeros(3)):
                continue
        p_i = np.array(elm.eleNodes[0].coords)
        p_io = np.array(elm.eleNodes[0].coords) + elm.geomtransf.offset_i
        offset_i = elm.geomtransf.offset_i
        u_i = np.array(
            analysis.results.node_displacements.registry[elm.eleNodes[0].uid][step][0:3])
        r_i = np.array(
            analysis.results.node_displacements.registry[elm.eleNodes[0].uid][step][3:6])
        u_io = transformations.offset_transformation(offset_i, u_i, r_i)

        p_j = np.array(elm.eleNodes[1].coords)
        p_jo = np.array(elm.eleNodes[1].coords) + elm.geomtransf.offset_j
        offset_j = elm.geomtransf.offset_j
        u_j = np.array(
            analysis.results.node_displacements.registry[elm.eleNodes[1].uid][step][0:3])
        r_j = np.array(
            analysis.results.node_displacements.registry[elm.eleNodes[1].uid][step][3:6])
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
            "color": graphics_common.OFFSET_COLOR
        }
    })


def add_data__frames_undeformed(dt, list_of_frames):
    x = []
    y = []
    z = []

    for elm in list_of_frames:

        p_i = np.array(elm.eleNodes[0].coords) + elm.geomtransf.offset_i
        p_j = np.array(elm.eleNodes[1].coords) + elm.geomtransf.offset_j

        x.extend(
            (p_i[0], p_j[0], None)
        )
        y.extend(
            (p_i[1], p_j[1], None)
        )
        z.extend(
            (p_i[2], p_j[2], None)
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
            "color": graphics_common.BEAM_MESH_COLOR
        }
    })


def add_data__nodes_deformed(analysis, dt, list_of_nodes, step,
                             scaling, function):
    ids = [int(node.uid) for node in list_of_nodes]
    location_data = np.full((len(list_of_nodes), 3), 0.00)
    displacement_data = np.full((len(list_of_nodes), 6), 0.00)
    for i, node in enumerate(list_of_nodes):
        location_data[i, :] = node.coords
        displacement_data[i, :] = \
            analysis.results.node_displacements.registry[node.uid][step]
    r = np.sqrt(displacement_data[:, 0]**2 +
                displacement_data[:, 1]**2 +
                displacement_data[:, 2]**2)
    r = np.reshape(r, (-1, 1))
    ids = np.reshape(np.array(ids), (-1, 1))
    displacement_data = np.concatenate((displacement_data, r, ids), 1)
    restraint_symbols = []
    for node in list_of_nodes:
        if True in node.restraint:
            restraint_symbols.append("fixed")
        else:
            restraint_symbols.append(function)

    marker = [graphics_common_3D.node_marker[sym][0]
              for sym in restraint_symbols]
    size = [graphics_common_3D.node_marker[sym][1]
            for sym in restraint_symbols]
    if function == 'internal':
        color = graphics_common.NODE_INTERNAL_COLOR
    else:
        color = graphics_common.NODE_PRIMARY_COLOR

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
            "symbol": marker,
            "color": color,
            "size": size,
            "line": {
                "color": color,
                "width": 4}
        }
    })


def get_auto_scaling_deformation(analysis, mdl, step):
    """
    Automatically calculate a scaling value that
    makes the maximum displacement appear approximately
    10% of the largest dimention of the building's bounding box
    """
    ref_len = mdl.reference_length()
    # maximum displacement
    max_d = 0.00
    elms = mdl.list_of_beamcolumn_elements()
    for elm in elms:
        u_i = analysis.results.node_displacements.registry[
            elm.eleNodes[0].uid][step][0:3]
        r_i = analysis.results.node_displacements.registry[
            elm.eleNodes[0].uid][step][3:6]
        u_j = analysis.results.node_displacements.registry[
            elm.eleNodes[1].uid][step][0:3]
        r_j = analysis.results.node_displacements.registry[
            elm.eleNodes[1].uid][step][3:6]
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


def deformed_shape(analysis,
                   step,
                   scaling,
                   extrude_frames,
                   camera=None,
                   subset_model=None):

    if subset_model:
        mdl = subset_model
    else:
        mdl = analysis.mdl
    # calculate a nice scaling factor if 0.00 is passed
    if scaling == 0:
        scaling = get_auto_scaling_deformation(analysis, mdl, step)

    layout = graphics_common_3D.global_layout(camera)
    dt = []

    list_of_frames = mdl.list_of_beamcolumn_elements()
    # list_of_steel_W_panel_zones = \
    #     mdl.list_of_steel_W_panel_zones()
    list_of_primary_nodes = mdl.list_of_primary_nodes()
    list_of_internal_nodes = mdl.list_of_internal_nodes()
    # list_of_parent_nodes = mdl.list_of_parent_nodes()
    # list_of_release_nodes = \
    #     [x.node_i for x in mdl.list_of_endreleases()]

    # if list_of_parent_nodes:
    #     add_data__nodes_deformed(
    #         analysis, dt, list_of_parent_nodes, step, scaling)

    if not extrude_frames:
        # draw the nodes
        add_data__nodes_deformed(
            analysis, dt, list_of_primary_nodes, step, scaling, 'free')
        add_data__nodes_deformed(
            analysis, dt, list_of_internal_nodes, step, scaling, 'internal')
    #     add_data__nodes_deformed(
    #         analysis, dt, list_of_release_nodes, step, scaling,
    #         color=graphics_common.NODE_INTERNAL_COLOR,
    #         marker=graphics_common_3D.node_marker['pinned'][0],
    #         size=graphics_common_3D.node_marker['pinned'][1])
        # draw the frames as lines
        add_data__frames_offsets_deformed(
            analysis, dt, list_of_frames, step, scaling)
        add_data__frames_deformed(
            analysis, dt, list_of_frames, step, 15, scaling)
        # we also add axes so that we can see 2D plots
        ref_len = mdl.reference_length()
        add_data__global_axes(dt, ref_len)
    else:
        # draw the extruded frames
        add_data__extruded_frames_deformed_mesh(
            analysis, dt, list_of_frames, step, scaling)
        # add_data__extruded_steel_W_PZ_deformed_mesh(
        #     analysis, dt, list_of_steel_W_panel_zones, step, scaling)

    fig_datastructure = dict(data=dt, layout=layout)
    fig = go.Figure(fig_datastructure)
    if not "pytest" in sys.modules:
        fig.show()

    metadata = {'scaling': scaling}
    return metadata


def basic_forces(analysis,
                 step,
                 scaling_global,
                 scaling_n,
                 scaling_q,
                 scaling_m,
                 scaling_t,
                 num_points,
                 force_conversion=1.00,
                 moment_conversion=1.00,
                 camera=None,
                 subset_model=None):

    layout = graphics_common_3D.global_layout(camera)
    dt = []

    if subset_model:
        mdl = subset_model
    else:
        mdl = analysis.mdl
    list_of_line_elements = mdl.list_of_beamcolumn_elements()
    list_of_primary_nodes = mdl.list_of_primary_nodes()
    list_of_internal_nodes = mdl.list_of_internal_nodes()

    # draw the frames
    add_data__frames_undeformed(
        dt, list_of_line_elements)
    # we also add axes so that we can see 2D plots
    ref_len = mdl.reference_length()
    add_data__global_axes(dt, ref_len)

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
    num_elems = len(list_of_line_elements)
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

    for i_elem, element in enumerate(
            list_of_line_elements):

        if element.visibility.skip_OpenSees_definition:
            continue

        x_vec = element.geomtransf.x_axis
        y_vec = element.geomtransf.y_axis
        z_vec = element.geomtransf.z_axis

        i_pos = np.array(element.eleNodes[0].coords) + element.geomtransf.offset_i
        
        T_global2local = np.vstack((x_vec, y_vec, z_vec))

        forces_global = analysis.results.element_forces.registry[
            element.uid][step][0:3]
        moments_global_ends = analysis.results.element_forces.registry[
            element.uid][step][3:6]

        moments_global_clear = transformations.offset_transformation(
            element.geomtransf.offset_i, moments_global_ends, forces_global)

        ni, qyi, qzi = T_global2local @ forces_global
        ti, myi, mzi = T_global2local @ moments_global_clear

        wx, wy, wz = analysis.load_case.line_element_udl.registry[element.uid].total()

        p_i = np.array(element.eleNodes[0].coords) + element.geomtransf.offset_i
        p_j = np.array(element.eleNodes[1].coords) + element.geomtransf.offset_j
        len_clr = np.linalg.norm(p_i - p_j)

        t = np.linspace(0.00, len_clr, num=num_points)

        nx_vec = - t * wx - ni
        qy_vec = t * wy + qyi
        qz_vec = t * wz + qzi
        tx_vec = np.full(num_points, -ti)
        mz_vec = t**2 * 0.50 * wy + t * qyi - mzi
        my_vec = t**2 * 0.50 * wz + t * qzi + myi

        # store results in the preallocated arrays

        nx_vecs[i_elem*num_points:i_elem*num_points +
                num_points] = nx_vec * force_conversion
        qy_vecs[i_elem*num_points:i_elem*num_points +
                num_points] = qy_vec * force_conversion
        qz_vecs[i_elem*num_points:i_elem*num_points +
                num_points] = qz_vec * force_conversion
        tx_vecs[i_elem*num_points:i_elem*num_points +
                num_points] = tx_vec * moment_conversion
        my_vecs[i_elem*num_points:i_elem*num_points +
                num_points] = my_vec * moment_conversion
        mz_vecs[i_elem*num_points:i_elem*num_points +
                num_points] = mz_vec * moment_conversion
        x_vecs[i_elem*3: i_elem*3 + 3] = x_vec
        y_vecs[i_elem*3: i_elem*3 + 3] = y_vec
        z_vecs[i_elem*3: i_elem*3 + 3] = z_vec
        i_poss[i_elem*3: i_elem*3 + 3] = i_pos
        elm_ln[i_elem] = len_clr

    # calculate scaling factors
    ref_len = mdl.reference_length()
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

    for i_elem, element in enumerate(
            list_of_line_elements):

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
        p_i = np.array(element.eleNodes[0].coords) + element.geomtransf.offset_i
        p_j = np.array(element.eleNodes[1].coords) + element.geomtransf.offset_j
        len_clr = np.linalg.norm(p_i - p_j)
        t = np.linspace(0.00, len_clr, num=num_points)

        for i in range(num_points - 1):

            p_start = i_pos + t[i] * x_vec
            p_end = i_pos + t[i+1] * x_vec

            # axial load & torsion
            p_i = p_start + \
                nx_vec[i] * y_vec * scaling_n * scaling_global
            p_j = p_end + \
                nx_vec[i+1] * y_vec * scaling_n * scaling_global
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
            p_i = p_start + \
                tx_vec[i] * z_vec * scaling_t * scaling_global
            p_j = p_end + \
                tx_vec[i+1] * z_vec * scaling_t * scaling_global
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
            p_i = p_start + \
                qy_vec[i] * y_vec * scaling_q * scaling_global
            p_j = p_end + \
                qy_vec[i+1] * y_vec * scaling_q * scaling_global
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
            p_i = p_start + \
                qz_vec[i] * z_vec * scaling_q * scaling_global
            p_j = p_end + \
                qz_vec[i+1] * z_vec * scaling_q * scaling_global
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
            p_i = p_start - \
                mz_vec[i] * y_vec * scaling_m * scaling_global
            p_j = p_end - \
                mz_vec[i+1] * y_vec * scaling_m * scaling_global
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
            p_i = p_start - \
                my_vec[i] * z_vec * scaling_m * scaling_global
            p_j = p_end - \
                my_vec[i+1] * z_vec * scaling_m * scaling_global
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
            p_i = p_start + \
                (qy_vec[i] * y_vec +
                 qz_vec[i] * z_vec) * scaling_q * scaling_global
            p_j = p_end + \
                (qy_vec[i+1] * y_vec +
                 qz_vec[i+1] * z_vec) * scaling_q * scaling_global
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
            p_i = p_start -\
                mz_vec[i] * y_vec * \
                scaling_m - my_vec[i] * z_vec * scaling_m * scaling_global
            p_j = p_end - mz_vec[i+1] * y_vec * \
                scaling_m - my_vec[i+1] * z_vec * scaling_m * scaling_global
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

    dt_a = [
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x1_a,
            "y": y1_a,
            "z": z1_a,
            "visible": False,
            "customdata": customdata_a,
            "hovertemplate": ' %{customdata:.0f}<br>'
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
            "hovertemplate": ' %{customdata:.0f}<br>'
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
            "hovertemplate": ' %{customdata:.0f}<br>'
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
            "hovertemplate": ' %{customdata:.0f}<br>'
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
            "hovertemplate": ' %{customdata:.0f}<br>'
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

    if not "pytest" in sys.modules:
        fig.show()

    metadata = {'scaling_n': scaling_n,
                'scaling_q': scaling_q,
                'scaling_m': scaling_m,
                'scaling_t': scaling_t}
    return metadata
