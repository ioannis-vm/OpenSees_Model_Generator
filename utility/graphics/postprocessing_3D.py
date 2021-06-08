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
from modeler import Building
from utility.graphics import common, common_3D
import openseespy.opensees as ops


def interp3D(element, num_points, scaling):
    x_vec = element.local_x_axis_vector()
    y_vec = element.local_y_axis_vector()
    z_vec = element.local_z_axis_vector()

    # global -> local transformation matrix
    T_global2local = np.vstack((x_vec, y_vec, z_vec))
    T_local2global = T_global2local.T

    u_i_global = ops.nodeDisp(element.node_i.uniq_id)[0:3]
    r_i_global = ops.nodeDisp(element.node_i.uniq_id)[3:6]
    u_j_global = ops.nodeDisp(element.node_j.uniq_id)[0:3]
    r_j_global = ops.nodeDisp(element.node_j.uniq_id)[3:6]

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

    element_point_samples = np.column_stack((
        np.linspace(element.node_i.coordinates[0],
                    element.node_j.coordinates[0], num=num_points),
        np.linspace(element.node_i.coordinates[1],
                    element.node_j.coordinates[1], num=num_points),
        np.linspace(element.node_i.coordinates[2],
                    element.node_j.coordinates[2], num=num_points),
    ))

    # maximum_d = np.max(np.abs(d_global))

    interpolation_points = element_point_samples + d_global * scaling

    return interpolation_points, r_local


def add_data__extruded_frames_deformed_mesh(dt,
                                            list_of_frames,
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
        interpolation_points, r_local = interp3D(
            elm, num_points, scaling=scaling)
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


def add_data__nodes_deformed(dt, list_of_nodes, scaling):
    dt.append({
        "type": "scatter3d",
        "mode": "markers",
        "x": [node.coordinates[0] for node in list_of_nodes],
        "y": [node.coordinates[1] for node in list_of_nodes],
        "z": [node.coordinates[2] for node in list_of_nodes],
        # "hoverinfo": "text",
        # "hovertext": ["node" + str(node.uniq_id)
        #               for node in level.nodes.node_list],
        "marker": {
            "symbol": [common_3D.node_marker[node.restraint_type][0]
                       for node in list_of_nodes],
            "color": common.NODE_PRIMARY_COLOR,
            "size": [common_3D.node_marker[node.restraint_type][1]
                     for node in list_of_nodes]
        }
    })


def deformed_shape(building: Building, scaling, extrude_frames=False):
    layout = common_3D.global_layout()
    dt = []

    list_of_frames = []
    list_of_nodes = []
    for lvl in building.levels.level_list:
        for element in lvl.beams.beam_list + lvl.columns.column_list:
            list_of_frames.append(element)
        for node in lvl.nodes.node_list:
            list_of_nodes.append(node)

    # draw the nodes
    add_data__nodes_deformed(dt, list_of_nodes, scaling=scaling)

    add_data__extruded_frames_deformed_mesh(
        dt, list_of_frames, num_points=25, scaling=scaling)

    fig_datastructure = dict(data=dt, layout=layout)
    fig = go.Figure(fig_datastructure)
    fig.show()
