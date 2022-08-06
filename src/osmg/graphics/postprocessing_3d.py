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
from typing import Optional
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go  # type: ignore
from .. import transformations
from . import graphics_common
from . import graphics_common_3d
from .preprocessing_3d import add_data__global_axes
from ..postprocessing.basic_forces import basic_forces

nparr = npt.NDArray[np.float64]


def force_scaling_factor(ref_len, fmax, factor):
    """
    Applies a scaling factor to basic forces
    """
    if fmax == 0.00:
        result = 0.00
    else:
        result = ref_len / fmax * factor
    return result


def interp_3d_deformation(element, u_i, r_i, u_j, r_j, num_points):
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

    x_vec: nparr = element.geomtransf.x_axis
    y_vec: nparr = element.geomtransf.y_axis
    z_vec: nparr = np.cross(x_vec, y_vec)

    # global -> local transformation matrix
    transf_global2local = \
        transformations.transformation_matrix(x_vec, y_vec, z_vec)
    transf_local2global = transf_global2local.T

    u_i_global = u_i
    r_i_global = r_i
    u_j_global = u_j
    r_j_global = r_j

    u_i_local = transf_global2local @ u_i_global
    r_i_local = transf_global2local @ r_i_global
    u_j_local = transf_global2local @ u_j_global
    r_j_local = transf_global2local @ r_j_global

    # discrete sample location parameter
    t_vec = np.linspace(0.00, 1.00, num=num_points)
    p_i = np.array(element.nodes[0].coords) + element.geomtransf.offset_i
    p_j = np.array(element.nodes[1].coords) + element.geomtransf.offset_j
    len_clr = np.linalg.norm(p_i - p_j)

    # shape function matrices
    nx_mat = np.column_stack((
        1. - t_vec,
        t_vec
    ))
    nyz_mat = np.column_stack((
        1. - 3. * t_vec**2 + 2. * t_vec**3,
        (t_vec - 2. * t_vec**2 + t_vec**3) * len_clr,
        3. * t_vec**2 - 2. * t_vec**3,
        (-t_vec**2 + t_vec**3) * len_clr
    ))
    nyz_derivative_mat = np.column_stack((
        - 6. * t_vec + 6. * t_vec**2,
        (1 - 4. * t_vec + 3. * t_vec**2) * len_clr,
        6. * t_vec - 6. * t_vec**2,
        (-2. * t_vec + 3. * t_vec**2) * len_clr
    ))

    # axial deformation
    d_x_local = nx_mat @ np.array([u_i_local[0], u_j_local[0]])

    # bending deformation along the local xy plane
    d_y_local = nyz_mat @ np.array([
        u_i_local[1],
        r_i_local[2],
        u_j_local[1],
        r_j_local[2]
    ])

    # bending deformation along the local xz plane
    d_z_local = nyz_mat @ np.array([
        u_i_local[2],
        -r_i_local[1],
        u_j_local[2],
        -r_j_local[1]
    ])

    # torsional deformation
    r_x_local = nx_mat @ np.array([r_i_local[0], r_j_local[0]])

    # bending rotation around the local z axis
    r_z_local = nyz_derivative_mat @ np.array([
        u_i_local[1],
        r_i_local[2],
        u_j_local[1],
        r_j_local[2]
    ]) / len_clr

    # bending rotation around the local y axis
    r_y_local = nyz_derivative_mat @ np.array([
        -u_i_local[2],
        r_i_local[1],
        -u_j_local[2],
        r_j_local[1]
    ]) / len_clr

    # all deformations
    d_local = np.column_stack((d_x_local, d_y_local, d_z_local))

    # all rotations
    r_local = np.column_stack((r_x_local, r_y_local, r_z_local))

    d_global = (transf_local2global @ d_local.T).T

    return d_global, r_local


def interp_3d_points(element, d_global, num_points, scaling):
    """
    Calculates intermediate points based on end locations and
    deformations
    """
    p_i = np.array(element.nodes[0].coords) + element.geomtransf.offset_i
    p_j = np.array(element.nodes[1].coords) + element.geomtransf.offset_j
    element_point_samples: nparr = np.column_stack((
        np.linspace(p_i[0], p_j[0], num=num_points),
        np.linspace(p_i[1], p_j[1], num=num_points),
        np.linspace(p_i[2], p_j[2], num=num_points),
    ))

    interpolation_points = element_point_samples + d_global * scaling

    return interpolation_points


def add_data__extruded_frames_deformed_mesh(analysis,
                                            case_name,
                                            data_dict,
                                            list_of_frames,
                                            step,
                                            scaling):
    """
    Adds a trace containing frame element extrusion mesh
    in its deformed state
    """
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
        u_i = (analysis.results[case_name].node_displacements
               [elm.nodes[0].uid][step][0:3])
        r_i = (analysis.results[case_name].node_displacements
               [elm.nodes[0].uid][step][3:6])
        u_j = (analysis.results[case_name].node_displacements
               [elm.nodes[1].uid][step][0:3])
        r_j = (analysis.results[case_name].node_displacements
               [elm.nodes[1].uid][step][3:6])
        # transferring them to the clear element ends
        offset_i = elm.geomtransf.offset_i
        offset_j = elm.geomtransf.offset_j
        u_i_o = transformations.offset_transformation(offset_i, u_i, r_i)
        u_j_o = transformations.offset_transformation(offset_j, u_j, r_j)
        d_global, r_local = interp_3d_deformation(
            elm, u_i_o, r_i, u_j_o, r_j, num_points)
        interpolation_points = interp_3d_points(
            elm, d_global, num_points, scaling)
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
    data_dict.append({
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
                              case_name,
                              data_dict,
                              list_of_frames,
                              step,
                              scaling):
    """
    Adds a trace containing frame element centroidal axis lines
    in their deformed state
    """
    if not list_of_frames:
        return
    x_list: list[Optional[float]] = []
    y_list: list[Optional[float]] = []
    z_list: list[Optional[float]] = []

    for elm in list_of_frames:
        if elm.visibility.hidden_at_line_plots:
            continue
        num_points = 8
        u_i = (analysis.results[case_name].node_displacements
               [elm.nodes[0].uid][step][0:3])
        r_i = (analysis.results[case_name].node_displacements
               [elm.nodes[0].uid][step][3:6])
        u_j = (analysis.results[case_name].node_displacements
               [elm.nodes[1].uid][step][0:3])
        r_j = (analysis.results[case_name].node_displacements
               [elm.nodes[1].uid][step][3:6])
        # transferring them to the clear element ends
        offset_i = elm.geomtransf.offset_i
        offset_j = elm.geomtransf.offset_j
        u_i_o = transformations.offset_transformation(offset_i, u_i, r_i)
        u_j_o = transformations.offset_transformation(offset_j, u_j, r_j)
        d_global, _ = interp_3d_deformation(
            elm, u_i_o, r_i, u_j_o, r_j, num_points)
        interpolation_points = interp_3d_points(
            elm, d_global, num_points, scaling)
        for i in range(len(interpolation_points)-1):
            x_list.extend((interpolation_points[i, 0],
                           interpolation_points[i+1, 0],
                           None))
            y_list.extend((interpolation_points[i, 1],
                           interpolation_points[i+1, 1],
                           None))
            z_list.extend((interpolation_points[i, 2],
                           interpolation_points[i+1, 2],
                           None))

    data_dict.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": x_list,
        "y": y_list,
        "z": z_list,
        "hoverinfo": "skip",
        "line": {
            "width": 5,
            "color": graphics_common.BEAM_MESH_COLOR
        }
    })


def add_data__frames_offsets_deformed(analysis,
                                      case_name,
                                      data_dict,
                                      list_of_frames,
                                      step,
                                      scaling):
    """
    Adds a trace containing frame element rigid offset lines
    in their deformed state
    """
    if not list_of_frames:
        return
    x_list: list[Optional[float]] = []
    y_list: list[Optional[float]] = []
    z_list: list[Optional[float]] = []
    for elm in list_of_frames:
        if np.array_equal(elm.geomtransf.offset_i, np.zeros(3)):
            if np.array_equal(elm.geomtransf.offset_j, np.zeros(3)):
                continue
        p_i: nparr = np.array(elm.nodes[0].coords)
        p_io: nparr = (np.array(elm.nodes[0].coords)
                       + elm.geomtransf.offset_i)
        offset_i = elm.geomtransf.offset_i
        u_i: nparr = np.array(
            analysis.results[case_name].node_displacements
            [elm.nodes[0].uid][step][0:3])
        r_i: nparr = np.array(
            analysis.results[case_name].node_displacements
            [elm.nodes[0].uid][step][3:6])
        u_io: nparr = transformations.offset_transformation(offset_i, u_i, r_i)

        p_j: nparr = np.array(elm.nodes[1].coords)
        p_jo: nparr = (np.array(elm.nodes[1].coords)
                       + elm.geomtransf.offset_j)
        offset_j = elm.geomtransf.offset_j
        u_j: nparr = np.array(
            analysis.results[case_name].node_displacements
            [elm.nodes[1].uid][step][0:3])
        r_j: nparr = np.array(
            analysis.results[case_name].node_displacements
            [elm.nodes[1].uid][step][3:6])
        u_jo: nparr = transformations.offset_transformation(offset_j, u_j, r_j)

        x_i = p_i + u_i * scaling
        x_io = p_io + u_io * scaling
        x_j = p_j + u_j * scaling
        x_jo = p_jo + u_jo * scaling

        x_list.extend((x_i[0], x_io[0], None))
        y_list.extend((x_i[1], x_io[1], None))
        z_list.extend((x_i[2], x_io[2], None))
        x_list.extend((x_j[0], x_jo[0], None))
        y_list.extend((x_j[1], x_jo[1], None))
        z_list.extend((x_j[2], x_jo[2], None))

    data_dict.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": x_list,
        "y": y_list,
        "z": z_list,
        "hoverinfo": "skip",
        "line": {
            "width": 8,
            "color": graphics_common.OFFSET_COLOR
        }
    })


def add_data__frames_undeformed(data_dict, list_of_frames):
    """
    Adds a trace containing frame element centroidal axis lines
    """
    x_list: list[Optional[float]] = []
    y_list: list[Optional[float]] = []
    z_list: list[Optional[float]] = []

    for elm in list_of_frames:

        p_i = np.array(elm.nodes[0].coords) + elm.geomtransf.offset_i
        p_j = np.array(elm.nodes[1].coords) + elm.geomtransf.offset_j

        x_list.extend(
            (p_i[0], p_j[0], None)
        )
        y_list.extend(
            (p_i[1], p_j[1], None)
        )
        z_list.extend(
            (p_i[2], p_j[2], None)
        )
    data_dict.append({
        "type": "scatter3d",
        "mode": "lines",
        "x": x_list,
        "y": y_list,
        "z": z_list,
        "hoverinfo": "skip",
        "line": {
            "width": 5,
            "color": graphics_common.BEAM_MESH_COLOR
        }
    })


def add_data__nodes_deformed(analysis, case_name,
                             data_dict, list_of_nodes,
                             step, scaling, function):
    """
    Adds a trace containing nodes in their deformed locations
    """
    ids_list = [int(node.uid) for node in list_of_nodes]
    location_data = np.full((len(list_of_nodes), 3), 0.00)
    displacement_data = np.full((len(list_of_nodes), 6), 0.00)
    for i, node in enumerate(list_of_nodes):
        location_data[i, :] = node.coords
        displacement_data[i, :] = \
            (analysis.results[case_name].node_displacements
             [node.uid][step])
    dist = np.sqrt(displacement_data[:, 0]**2 +
                   displacement_data[:, 1]**2 +
                   displacement_data[:, 2]**2)
    dist = np.reshape(dist, (-1, 1))
    ids = np.reshape(np.array(ids_list), (-1, 1))
    displacement_data = np.concatenate((displacement_data, dist, ids), 1)
    restraint_symbols = []
    for node in list_of_nodes:
        if True in node.restraint:
            restraint_symbols.append("fixed")
        else:
            restraint_symbols.append(function)

    marker = [graphics_common_3d.node_marker[sym][0]
              for sym in restraint_symbols]
    size = [graphics_common_3d.node_marker[sym][1]
            for sym in restraint_symbols]
    if function == 'internal':
        color = graphics_common.NODE_INTERNAL_COLOR
    else:
        color = graphics_common.NODE_PRIMARY_COLOR

    data_dict.append({
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


def get_auto_scaling_deformation(analysis, case_name, mdl, step):
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
        u_i = analysis.results[case_name].node_displacements[
            elm.nodes[0].uid][step][0:3]
        r_i = analysis.results[case_name].node_displacements[
            elm.nodes[0].uid][step][3:6]
        u_j = analysis.results[case_name].node_displacements[
            elm.nodes[1].uid][step][0:3]
        r_j = analysis.results[case_name].node_displacements[
            elm.nodes[1].uid][step][3:6]
        d_global, _ = interp_3d_deformation(
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
    scaling = max(scaling, 1.00)
    return scaling


def show_deformed_shape(analysis,
                        case_name,
                        step,
                        scaling,
                        extrude,
                        camera=None,
                        subset_model=None):

    """
    Visualize the model in its deformed state
    Arguments:
      analysis (Analysis): an analysis object
      case_name (str): the name of the load_case to be visualized
      step (int): the analysis step to be visualized
      scaling (float): scaling factor for the deformations. If 0.00 is
        provided, the scaling factor is calculated automatically.
      extrude (bool): wether to extrude frame elements
      camera (dict): custom positioning of the camera
    """
    if subset_model:
        mdl = subset_model
    else:
        mdl = analysis.mdl
    # calculate a nice scaling factor if 0.00 is passed
    if scaling == 0:
        scaling = get_auto_scaling_deformation(analysis, case_name, mdl, step)

    layout = graphics_common_3d.global_layout(camera)
    data_dict: list[dict[str, object]] = []

    list_of_frames = [elm for elm in mdl.list_of_beamcolumn_elements()
                      if not elm.visibility.skip_opensees_definition]

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

    if not extrude:
        # draw the nodes
        add_data__nodes_deformed(
            analysis, case_name, data_dict, list_of_primary_nodes,
            step, scaling, 'free')
        add_data__nodes_deformed(
            analysis, case_name, data_dict, list_of_internal_nodes,
            step, scaling, 'internal')
    #     add_data__nodes_deformed(
    #         analysis, dt, list_of_release_nodes, step, scaling,
    #         color=graphics_common.NODE_INTERNAL_COLOR,
    #         marker=graphics_common_3d.node_marker['pinned'][0],
    #         size=graphics_common_3d.node_marker['pinned'][1])
        # draw the frames as lines
        add_data__frames_offsets_deformed(
            analysis, case_name, data_dict, list_of_frames, step, scaling)
        add_data__frames_deformed(
            analysis, case_name, data_dict, list_of_frames, step, scaling)
        # we also add axes so that we can see 2D plots
        ref_len = mdl.reference_length()
        add_data__global_axes(data_dict, ref_len)
    else:
        # draw the extruded frames
        add_data__extruded_frames_deformed_mesh(
            analysis, case_name, data_dict, list_of_frames, step, scaling)
        # add_data__extruded_steel_W_PZ_deformed_mesh(
        #     analysis, dt, list_of_steel_W_panel_zones, step, scaling)

    fig_datastructure = dict(data=data_dict, layout=layout)
    fig = go.Figure(fig_datastructure)
    if "pytest" not in sys.modules:
        fig.show()

    metadata = {'scaling': scaling}
    return metadata


def show_basic_forces(analysis,
                      case_name,
                      step,
                      scaling_global,
                      scaling_n,
                      scaling_q,
                      scaling_m,
                      scaling_t,
                      num_points,
                      force_conversion=1.00,
                      moment_conversion=1.00,
                      global_axes=False,
                      camera=None,
                      subset_model=None):
    """
    Visualize the model and plot the frame element basic forces
    Arguments:
      analysis (Analysis): an analysis object
      case_name (str): the name of the load_case to be visualized
      step (int): the analysis step to be visualized
      scaling_global (float): I don't even remember what this
        does. It's kind of a mess right now.
      scaling_n (float):
      scaling_q (float):
      scaling_m (float):
      scaling_t (float):
      num_points (int): number of points to include in the basic force
        curves
      force_conversion (float): Conversion factor to be applied at the
        hover box data for forces (for unit conversions)
      moment_conversion (float): Conversion factor to be applied at the
        hover box data for moments (for unit conversions)
      global_axes (bool): whether to show global axes
      camera (dict): custom positioning of the camera
      subset_model (Model): use this model instead of the one
        contained in the analysis object.
        It needs to be a subset of the original model. This can be
          used to only show the results for some part of a large
          model.
    """
    # TODO: what is going on with the scaling factors?...
    layout = graphics_common_3d.global_layout(camera)
    data_dict: list[dict[str, object]] = []

    if subset_model:
        mdl = subset_model
    else:
        mdl = analysis.mdl
    list_of_line_elements = [elm for elm in mdl.list_of_beamcolumn_elements()
                             if not elm.visibility.skip_opensees_definition]

    # draw the frames
    add_data__frames_undeformed(
        data_dict, list_of_line_elements)
    # we also add axes so that we can see 2D plots
    ref_len = mdl.reference_length()
    if global_axes:
        add_data__global_axes(data_dict, ref_len)

    # Plot options:
    # a: axial
    # b: shear in local Y and Z
    # c: moment in Y and Z
    # d: torsion
    # e: shear combined
    # f: moment combined
    x1_a: list[Optional[float]] = []
    y1_a: list[Optional[float]] = []
    z1_a: list[Optional[float]] = []
    colors1_a: list[Optional[str]] = []
    customdata_a: list[Optional[float]] = []

    x1_b: list[Optional[float]] = []
    y1_b: list[Optional[float]] = []
    z1_b: list[Optional[float]] = []
    colors1_b: list[Optional[str]] = []
    customdata_b: list[Optional[float]] = []

    x1_c: list[Optional[float]] = []
    y1_c: list[Optional[float]] = []
    z1_c: list[Optional[float]] = []
    colors1_c: list[Optional[str]] = []
    customdata_c: list[Optional[float]] = []

    x1_d: list[Optional[float]] = []
    y1_d: list[Optional[float]] = []
    z1_d: list[Optional[float]] = []
    colors1_d: list[Optional[str]] = []
    customdata_d: list[Optional[float]] = []

    x1_e: list[Optional[float]] = []
    y1_e: list[Optional[float]] = []
    z1_e: list[Optional[float]] = []
    colors1_e: list[Optional[str]] = []
    customdata_e: list[Optional[float]] = []

    x1_f: list[Optional[float]] = []
    y1_f: list[Optional[float]] = []
    z1_f: list[Optional[float]] = []
    colors1_f: list[Optional[str]] = []
    customdata_f: list[Optional[float]] = []

    # (we do this to determine the internal forces
    #  for all elements before we even start plotting
    #  them, to be able to compute a nice scaling factor
    #  without having to then recalculate the basic forces)
    nx_vecs = {}
    qy_vecs = {}
    qz_vecs = {}
    tx_vecs = {}
    mz_vecs = {}
    my_vecs = {}

    for element in list_of_line_elements:

        if element.visibility.skip_opensees_definition:
            continue

        forces = basic_forces(
            analysis, case_name, step, element, num_points, as_tuple=True)
        assert isinstance(forces, tuple)
        nx_vec, qy_vec, qz_vec, tx_vec, mz_vec, my_vec = forces
        assert isinstance(nx_vec, np.ndarray)
        assert isinstance(qy_vec, np.ndarray)
        assert isinstance(qz_vec, np.ndarray)
        assert isinstance(tx_vec, np.ndarray)
        assert isinstance(mz_vec, np.ndarray)
        assert isinstance(my_vec, np.ndarray)

        # store results in the preallocated arrays

        nx_vecs[element.uid] = nx_vec * force_conversion
        qy_vecs[element.uid] = qy_vec * force_conversion
        qz_vecs[element.uid] = qz_vec * force_conversion
        tx_vecs[element.uid] = tx_vec * moment_conversion
        my_vecs[element.uid] = my_vec * moment_conversion
        mz_vecs[element.uid] = mz_vec * moment_conversion

    # calculate scaling factors
    ref_len = mdl.reference_length()
    factor = 0.05
    nx_max = np.max(np.abs(np.column_stack(list(nx_vecs.values()))))
    scaling_n = force_scaling_factor(ref_len, nx_max, factor)
    if scaling_n > 1.e8:
        scaling_t = 1.00
    qy_max = np.max(np.abs(np.column_stack(list(qy_vecs.values()))))
    qz_max = np.max(np.abs(np.column_stack(list(qz_vecs.values()))))
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
    my_max = np.max(np.abs(np.column_stack(list(my_vecs.values()))))
    mz_max = np.max(np.abs(np.column_stack(list(mz_vecs.values()))))
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

    for element in list_of_line_elements:

        # retrieve results from the preallocated arrays
        nx_vec = nx_vecs[element.uid]
        qy_vec = qy_vecs[element.uid]
        qz_vec = qz_vecs[element.uid]
        tx_vec = tx_vecs[element.uid]
        my_vec = my_vecs[element.uid]
        mz_vec = mz_vecs[element.uid]
        x_vec = element.geomtransf.x_axis
        y_vec = element.geomtransf.y_axis
        z_vec = element.geomtransf.z_axis
        i_pos = (np.array(element.nodes[0].coords)
                 + element.geomtransf.offset_i)
        p_i = (np.array(element.nodes[0].coords)
               + element.geomtransf.offset_i)
        p_j = (np.array(element.nodes[1].coords)
               + element.geomtransf.offset_j)
        len_clr = np.linalg.norm(p_i - p_j)
        t_vec = np.linspace(0.00, len_clr, num=num_points)

        for i in range(num_points - 1):

            p_start = i_pos + t_vec[i] * x_vec
            p_end = i_pos + t_vec[i+1] * x_vec

            # axial load
            p_i = p_start + \
                nx_vec[i] * y_vec * scaling_n * scaling_global
            p_j = p_end + \
                nx_vec[i+1] * y_vec * scaling_n * scaling_global

            x1_a.extend((p_i[0], p_j[0], None))
            y1_a.extend((p_i[1], p_j[1], None))
            z1_a.extend((p_i[2], p_j[2], None))
            customdata_a.extend(
                (nx_vec[i], nx_vec[i+1], None))
            colors1_a.extend(["red"]*3)

            # torsion
            p_i = p_start + \
                tx_vec[i] * z_vec * scaling_t * scaling_global
            p_j = p_end + \
                tx_vec[i+1] * z_vec * scaling_t * scaling_global
            x1_d.extend((p_i[0], p_j[0], None))
            y1_d.extend((p_i[1], p_j[1], None))
            z1_d.extend((p_i[2], p_j[2], None))
            customdata_d.extend(
                (tx_vec[i], tx_vec[i+1], None))
            colors1_d.extend(["orange"]*3)

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
            colors1_b.extend(["green"]*3)
            p_i = p_start + \
                qz_vec[i] * z_vec * scaling_q * scaling_global
            p_j = p_end + \
                qz_vec[i+1] * z_vec * scaling_q * scaling_global
            x1_b.extend((p_i[0], p_j[0], None))
            y1_b.extend((p_i[1], p_j[1], None))
            z1_b.extend((p_i[2], p_j[2], None))
            customdata_b.extend(
                (qz_vec[i], qz_vec[i+1], None))
            colors1_b.extend(["green"]*3)

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
            colors1_c.extend(["blue"]*3)
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
            colors1_c.extend(["blue"]*3)

            # shear load combined
            p_i = p_start + \
                (qy_vec[i] * y_vec +
                 qz_vec[i] * z_vec) * scaling_q * scaling_global
            p_j = p_end + \
                (qy_vec[i+1] * y_vec +
                 qz_vec[i+1] * z_vec) * scaling_q * scaling_global
            x1_e.extend((p_i[0], p_j[0], None))
            y1_e.extend((p_i[1], p_j[1], None))
            z1_e.extend((p_i[2], p_j[2], None))
            customdata_e.extend(
                (np.sqrt(qy_vec[i]**2 + qz_vec[i]**2),
                 np.sqrt(qy_vec[i+1]**2 + qz_vec[i+1]**2), None))
            colors1_e.extend(["green"]*3)

            # both moments combined!
            p_i = p_start -\
                mz_vec[i] * y_vec * \
                scaling_m - my_vec[i] * z_vec * scaling_m * scaling_global
            p_j = p_end - mz_vec[i+1] * y_vec * \
                scaling_m - my_vec[i+1] * z_vec * scaling_m * scaling_global
            # note: moments plotted upside down!
            x1_f.extend((p_i[0], p_j[0], None))
            y1_f.extend((p_i[1], p_j[1], None))
            z1_f.extend((p_i[2], p_j[2], None))
            customdata_f.extend(
                (np.sqrt(mz_vec[i]**2 + my_vec[i]**2),
                 np.sqrt(mz_vec[i+1]**2 + my_vec[i+1]**2),
                 None)
            )
            colors1_f.extend(["blue"]*3)

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
        }
    ]
    dt_f = [
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x1_f,
            "y": y1_f,
            "z": z1_f,
            "visible": False,
            "customdata": customdata_f,
            "hovertemplate": ' %{customdata:.0f}<br>'
            '<extra></extra>',
            "line": {
                "width": 3,
                "color": colors1_f
            }
        }
    ]

    fig_datastructure = dict(data=data_dict + dt_a +
                             dt_b + dt_c
                             + dt_d + dt_e + dt_f,
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
                               [True]*len(data_dict)+[False]*6
                               }]
                    ),
                    dict(
                        label="Axial",
                        method="update",
                        args=[{"visible":
                               [True]*len(data_dict)+[True]+[False]*5
                               }]
                    ),
                    dict(
                        label="Shear",
                        method="update",
                        args=[{"visible":
                               [True]*len(data_dict)+[False]+[True]+[False]*4
                               }]
                    ),
                    dict(
                        label="Moment",
                        method="update",
                        args=[{"visible":
                               [True]*len(data_dict)+[False]*2+[True]+[False]*3
                               }]
                    ),
                    dict(
                        label="Torsion",
                        method="update",
                        args=[{"visible":
                               [True]*len(data_dict)+[False]*3+[True]+[False]*2
                               }]
                    ),
                    dict(
                        label="Shear (combined)",
                        method="update",
                        args=[{"visible":
                               [True]*len(data_dict)+[False]*4+[True]+[False]
                               }]
                    ),
                    dict(
                        label="Moment (combined)",
                        method="update",
                        args=[{"visible":
                               [True]*len(data_dict)+[False]*5+[True]
                               }]
                    )
                ]
            )
        ]
    )

    if "pytest" not in sys.modules:
        fig.show()

    metadata = {'scaling_n': scaling_n,
                'scaling_q': scaling_q,
                'scaling_m': scaling_m,
                'scaling_t': scaling_t}
    return metadata


def show_basic_forces_combo(
        combo,
        scaling_global,
        scaling_n,
        scaling_q,
        scaling_m,
        scaling_t,
        num_points,
        force_conversion=1.00,
        moment_conversion=1.00,
        global_axes=False,
        camera=None,
        subset_model=None):
    """
    Visualize the model and plot the enveloped frame element basic forces
    for a load combination.
    Arguments:
      combo (LoadCombination): a load combination object
      step (int): the analysis step to be visualized
      scaling_global (float): I don't even remember what this
        does. It's kind of a mess right now.
      scaling_n (float):
      scaling_q (float):
      scaling_m (float):
      scaling_t (float):
      num_points (int): number of points to include in the basic force
        curves
      force_conversion (float): Conversion factor to be applied at the
        hover box data for forces (for unit conversions)
      moment_conversion (float): Conversion factor to be applied at the
        hover box data for moments (for unit conversions)
      global_axes (bool): whether to show global axes
      camera (dict): custom positioning of the camera
      subset_model (Model): use this model instead of the one
        contained in the analysis object.
        It needs to be a subset of the original model. This can be
          used to only show the results for some part of a large
          model.
    """
    # TODO: merge code repetitions with the previous function
    layout = graphics_common_3d.global_layout(camera)
    data_dict: list[dict[str, object]] = []

    if subset_model:
        mdl = subset_model
    else:
        mdl = combo.mdl
    list_of_line_elements = [elm for elm in mdl.list_of_beamcolumn_elements()
                             if not elm.visibility.skip_opensees_definition]

    # draw the frames
    add_data__frames_undeformed(
        data_dict, list_of_line_elements)
    # we also add axes so that we can see 2D plots
    ref_len = mdl.reference_length()
    if global_axes:
        add_data__global_axes(data_dict, ref_len)

    # Plot options:
    # a: axial
    # b: shear in local Y and Z
    # c: moment in Y and Z
    # d: torsion
    # e: shear combined
    # f: moment combined
    x1_a: list[Optional[float]] = []
    y1_a: list[Optional[float]] = []
    z1_a: list[Optional[float]] = []
    colors1_a: list[Optional[str]] = []
    customdata_a: list[Optional[float]] = []

    x1_b: list[Optional[float]] = []
    y1_b: list[Optional[float]] = []
    z1_b: list[Optional[float]] = []
    colors1_b: list[Optional[str]] = []
    customdata_b: list[Optional[float]] = []

    x1_c: list[Optional[float]] = []
    y1_c: list[Optional[float]] = []
    z1_c: list[Optional[float]] = []
    colors1_c: list[Optional[str]] = []
    customdata_c: list[Optional[float]] = []

    x1_d: list[Optional[float]] = []
    y1_d: list[Optional[float]] = []
    z1_d: list[Optional[float]] = []
    colors1_d: list[Optional[str]] = []
    customdata_d: list[Optional[float]] = []

    # (we do this to determine the internal forces
    #  for all elements before we even start plotting
    #  them, to be able to compute a nice scaling factor
    #  without having to then recalculate the basic forces)
    nx_vecs_min = {}
    qy_vecs_min = {}
    qz_vecs_min = {}
    tx_vecs_min = {}
    mz_vecs_min = {}
    my_vecs_min = {}
    nx_vecs_max = {}
    qy_vecs_max = {}
    qz_vecs_max = {}
    tx_vecs_max = {}
    mz_vecs_max = {}
    my_vecs_max = {}

    for element in list_of_line_elements:

        if element.visibility.skip_opensees_definition:
            continue

        df_min, df_max = combo.envelope_basic_forces(element, num_points)

        # store results in the preallocated arrays

        nx_vecs_min[element.uid] = df_min['nx'].to_numpy() * force_conversion
        qy_vecs_min[element.uid] = df_min['qy'].to_numpy() * force_conversion
        qz_vecs_min[element.uid] = df_min['qz'].to_numpy() * force_conversion
        tx_vecs_min[element.uid] = df_min['tx'].to_numpy() * moment_conversion
        my_vecs_min[element.uid] = df_min['my'].to_numpy() * moment_conversion
        mz_vecs_min[element.uid] = df_min['mz'].to_numpy() * moment_conversion
        nx_vecs_max[element.uid] = df_max['nx'].to_numpy() * force_conversion
        qy_vecs_max[element.uid] = df_max['qy'].to_numpy() * force_conversion
        qz_vecs_max[element.uid] = df_max['qz'].to_numpy() * force_conversion
        tx_vecs_max[element.uid] = df_max['tx'].to_numpy() * moment_conversion
        my_vecs_max[element.uid] = df_max['my'].to_numpy() * moment_conversion
        mz_vecs_max[element.uid] = df_max['mz'].to_numpy() * moment_conversion

    # calculate scaling factors
    ref_len = mdl.reference_length()
    factor = 0.05
    nx_max = np.max(np.abs(
        np.column_stack(
            list(nx_vecs_min.values())+list(nx_vecs_max.values()))))
    scaling_n = force_scaling_factor(ref_len, nx_max, factor)
    if scaling_n > 1.e8:
        scaling_t = 1.00
    qy_max = np.max(np.abs(np.column_stack(
        list(qy_vecs_min.values())+list(qy_vecs_max.values()))))
    qz_max = np.max(np.abs(np.column_stack(
        list(qz_vecs_min.values())+list(qz_vecs_max.values()))))
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
    my_max = np.max(np.abs(np.column_stack(
        list(my_vecs_min.values())+list(my_vecs_max.values()))))
    mz_max = np.max(np.abs(np.column_stack(
        list(mz_vecs_min.values())+list(mz_vecs_max.values()))))
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

    for element in list_of_line_elements:

        # retrieve results from the preallocated arrays
        nx_vec_min = nx_vecs_min[element.uid]
        qy_vec_min = qy_vecs_min[element.uid]
        qz_vec_min = qz_vecs_min[element.uid]
        tx_vec_min = tx_vecs_min[element.uid]
        my_vec_min = my_vecs_min[element.uid]
        mz_vec_min = mz_vecs_min[element.uid]

        nx_vec_max = nx_vecs_max[element.uid]
        qy_vec_max = qy_vecs_max[element.uid]
        qz_vec_max = qz_vecs_max[element.uid]
        tx_vec_max = tx_vecs_max[element.uid]
        my_vec_max = my_vecs_max[element.uid]
        mz_vec_max = mz_vecs_max[element.uid]

        x_vec = element.geomtransf.x_axis
        y_vec = element.geomtransf.y_axis
        z_vec = element.geomtransf.z_axis
        i_pos = (np.array(element.nodes[0].coords)
                 + element.geomtransf.offset_i)
        p_i = (np.array(element.nodes[0].coords)
               + element.geomtransf.offset_i)
        p_j = (np.array(element.nodes[1].coords)
               + element.geomtransf.offset_j)
        len_clr = np.linalg.norm(p_i - p_j)
        t_vec = np.linspace(0.00, len_clr, num=num_points)

        for i in range(num_points - 1):

            p_start = i_pos + t_vec[i] * x_vec
            p_end = i_pos + t_vec[i+1] * x_vec

            # axial load
            p_i = p_start + \
                nx_vec_min[i] * y_vec * scaling_n * scaling_global
            p_j = p_end + \
                nx_vec_min[i+1] * y_vec * scaling_n * scaling_global
            x1_a.extend((p_i[0], p_j[0], None))
            y1_a.extend((p_i[1], p_j[1], None))
            z1_a.extend((p_i[2], p_j[2], None))
            customdata_a.extend(
                (nx_vec_min[i], nx_vec_min[i+1], None))
            colors1_a.extend(["red"]*3)
            p_i = p_start + \
                nx_vec_max[i] * y_vec * scaling_n * scaling_global
            p_j = p_end + \
                nx_vec_max[i+1] * y_vec * scaling_n * scaling_global
            x1_a.extend((p_i[0], p_j[0], None))
            y1_a.extend((p_i[1], p_j[1], None))
            z1_a.extend((p_i[2], p_j[2], None))
            customdata_a.extend(
                (nx_vec_max[i], nx_vec_max[i+1], None))
            colors1_a.extend(["green"]*3)

            # torsion
            p_i = p_start + \
                tx_vec_min[i] * z_vec * scaling_t * scaling_global
            p_j = p_end + \
                tx_vec_min[i+1] * z_vec * scaling_t * scaling_global
            x1_d.extend((p_i[0], p_j[0], None))
            y1_d.extend((p_i[1], p_j[1], None))
            z1_d.extend((p_i[2], p_j[2], None))
            customdata_d.extend(
                (tx_vec_min[i], tx_vec_min[i+1], None))
            colors1_d.extend(["red"]*3)
            p_i = p_start + \
                tx_vec_max[i] * z_vec * scaling_t * scaling_global
            p_j = p_end + \
                tx_vec_max[i+1] * z_vec * scaling_t * scaling_global
            x1_d.extend((p_i[0], p_j[0], None))
            y1_d.extend((p_i[1], p_j[1], None))
            z1_d.extend((p_i[2], p_j[2], None))
            customdata_d.extend(
                (tx_vec_max[i], tx_vec_max[i+1], None))
            colors1_d.extend(["green"]*3)

            # shear load on y and z axes
            p_i = p_start + \
                qy_vec_min[i] * y_vec * scaling_q * scaling_global
            p_j = p_end + \
                qy_vec_min[i+1] * y_vec * scaling_q * scaling_global
            x1_b.extend((p_i[0], p_j[0], None))
            y1_b.extend((p_i[1], p_j[1], None))
            z1_b.extend((p_i[2], p_j[2], None))
            customdata_b.extend(
                (qy_vec_min[i], qy_vec_min[i+1], None))
            colors1_b.extend(["red"]*3)
            p_i = p_start + \
                qy_vec_max[i] * y_vec * scaling_q * scaling_global
            p_j = p_end + \
                qy_vec_max[i+1] * y_vec * scaling_q * scaling_global
            x1_b.extend((p_i[0], p_j[0], None))
            y1_b.extend((p_i[1], p_j[1], None))
            z1_b.extend((p_i[2], p_j[2], None))
            customdata_b.extend(
                (qy_vec_max[i], qy_vec_max[i+1], None))
            colors1_b.extend(["green"]*3)
            p_i = p_start + \
                qz_vec_min[i] * z_vec * scaling_q * scaling_global
            p_j = p_end + \
                qz_vec_min[i+1] * z_vec * scaling_q * scaling_global
            x1_b.extend((p_i[0], p_j[0], None))
            y1_b.extend((p_i[1], p_j[1], None))
            z1_b.extend((p_i[2], p_j[2], None))
            customdata_b.extend(
                (qz_vec_min[i], qz_vec_min[i+1], None))
            colors1_b.extend(["red"]*3)
            p_i = p_start + \
                qz_vec_max[i] * z_vec * scaling_q * scaling_global
            p_j = p_end + \
                qz_vec_max[i+1] * z_vec * scaling_q * scaling_global
            x1_b.extend((p_i[0], p_j[0], None))
            y1_b.extend((p_i[1], p_j[1], None))
            z1_b.extend((p_i[2], p_j[2], None))
            customdata_b.extend(
                (qz_vec_max[i], qz_vec_max[i+1], None))
            colors1_b.extend(["green"]*3)

            # moment around z and y axes
            p_i = p_start - \
                mz_vec_min[i] * y_vec * scaling_m * scaling_global
            p_j = p_end - \
                mz_vec_min[i+1] * y_vec * scaling_m * scaling_global
            # note: moments plotted upside down!
            x1_c.extend((p_i[0], p_j[0], None))
            y1_c.extend((p_i[1], p_j[1], None))
            z1_c.extend((p_i[2], p_j[2], None))
            customdata_c.extend(
                (mz_vec_min[i], mz_vec_min[i+1], None))
            colors1_c.extend(["red"]*3)
            p_i = p_start - \
                mz_vec_max[i] * y_vec * scaling_m * scaling_global
            p_j = p_end - \
                mz_vec_max[i+1] * y_vec * scaling_m * scaling_global
            # note: moments plotted upside down!
            x1_c.extend((p_i[0], p_j[0], None))
            y1_c.extend((p_i[1], p_j[1], None))
            z1_c.extend((p_i[2], p_j[2], None))
            customdata_c.extend(
                (mz_vec_max[i], mz_vec_max[i+1], None))
            colors1_c.extend(["green"]*3)
            # moment around y axis
            p_i = p_start - \
                my_vec_min[i] * z_vec * scaling_m * scaling_global
            p_j = p_end - \
                my_vec_min[i+1] * z_vec * scaling_m * scaling_global
            x1_c.extend((p_i[0], p_j[0], None))
            y1_c.extend((p_i[1], p_j[1], None))
            z1_c.extend((p_i[2], p_j[2], None))
            customdata_c.extend(
                (my_vec_min[i], my_vec_min[i+1], None))
            colors1_c.extend(["red"]*3)
            p_i = p_start - \
                my_vec_max[i] * z_vec * scaling_m * scaling_global
            p_j = p_end - \
                my_vec_max[i+1] * z_vec * scaling_m * scaling_global
            x1_c.extend((p_i[0], p_j[0], None))
            y1_c.extend((p_i[1], p_j[1], None))
            z1_c.extend((p_i[2], p_j[2], None))
            customdata_c.extend(
                (my_vec_max[i], my_vec_max[i+1], None))
            colors1_c.extend(["green"]*3)

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
        }
    ]

    fig_datastructure = dict(data=data_dict + dt_a +
                             dt_b + dt_c
                             + dt_d,
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
                               [True]*len(data_dict)+[False]*4
                               }]
                    ),
                    dict(
                        label="Axial",
                        method="update",
                        args=[{"visible":
                               [True]*len(data_dict)+[True]+[False]*3
                               }]
                    ),
                    dict(
                        label="Shear",
                        method="update",
                        args=[{"visible":
                               [True]*len(data_dict)+[False]+[True]+[False]*2
                               }]
                    ),
                    dict(
                        label="Moment",
                        method="update",
                        args=[{"visible":
                               [True]*len(data_dict)+[False]*2+[True]+[False]
                               }]
                    ),
                    dict(
                        label="Torsion",
                        method="update",
                        args=[{"visible":
                               [True]*len(data_dict)+[False]*3+[True]
                               }]
                    )
                ]
            )
        ]
    )

    if "pytest" not in sys.modules:
        fig.show()

    metadata = {'scaling_n': scaling_n,
                'scaling_q': scaling_q,
                'scaling_m': scaling_m,
                'scaling_t': scaling_t}
    return metadata
