"""
Defines utility functions used for data visualization.

"""

#
#   _|_|      _|_|_|  _|      _|    _|_|_|
# _|    _|  _|        _|_|  _|_|  _|
# _|    _|    _|_|    _|  _|  _|  _|  _|_|
# _|    _|        _|  _|      _|  _|    _|
#   _|_|    _|_|_|    _|      _|    _|_|_|
#
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from __future__ import annotations
import sys
from typing import Optional
from typing import Union
from typing import Any
from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go  # type: ignore
from .. import transformations
from . import graphics_common
from . import graphics_common_3d
from .preprocessing_3d import add_data__global_axes
from ..postprocessing.basic_forces import basic_forces
from ..ops import element
from ..model import Model

if TYPE_CHECKING:
    from ..solver import Analysis


nparr = npt.NDArray[np.float64]


def force_scaling_factor(ref_len, fmax, factor):
    """
    Applies a scaling factor to basic forces.

    Arguments:
      ref_len: :func:`~osmg.model.Model.reference_length` of the
        model.
      fmax: Largest value in the basic forces.
      factor: Required scaling factor.

    """
    if fmax == 0.00:
        result = 0.00
    else:
        result = ref_len / fmax * factor
    return result


def interp_3d_deformation(elm, u_i, r_i, u_j, r_j, num_points):
    """
    Given the deformations of the ends of a Bernoulli beam,
    use its shape functions to obtain intermediate points.

    Arguments:
      element: A line element
      u_i: 3 displacements at end i, global system
      r_i: 3 rotations at end i, global system
      u_j, r_j: similar to u_i, r_i.
      num_points: Number of interpolation points

    Returns:
      Displacements (global system) and rotations (local system). The
        rotations are needed for plotting the deformed shape with
        extruded frame elements.

    """

    x_vec: nparr = elm.geomtransf.x_axis
    y_vec: nparr = elm.geomtransf.y_axis
    z_vec: nparr = np.cross(x_vec, y_vec)

    # global -> local transformation matrix
    transf_global2local = transformations.transformation_matrix(
        x_vec, y_vec, z_vec
    )
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
    p_i = np.array(elm.nodes[0].coords) + elm.geomtransf.offset_i
    p_j = np.array(elm.nodes[1].coords) + elm.geomtransf.offset_j
    len_clr = np.linalg.norm(p_i - p_j)

    # shape function matrices
    nx_mat = np.column_stack((1.0 - t_vec, t_vec))
    nyz_mat = np.column_stack(
        (
            1.0 - 3.0 * t_vec**2 + 2.0 * t_vec**3,
            (t_vec - 2.0 * t_vec**2 + t_vec**3) * len_clr,
            3.0 * t_vec**2 - 2.0 * t_vec**3,
            (-(t_vec**2) + t_vec**3) * len_clr,
        )
    )
    nyz_derivative_mat = np.column_stack(
        (
            -6.0 * t_vec + 6.0 * t_vec**2,
            (1 - 4.0 * t_vec + 3.0 * t_vec**2) * len_clr,
            6.0 * t_vec - 6.0 * t_vec**2,
            (-2.0 * t_vec + 3.0 * t_vec**2) * len_clr,
        )
    )

    # axial deformation
    d_x_local = nx_mat @ np.array([u_i_local[0], u_j_local[0]])

    # bending deformation along the local xy plane
    d_y_local = nyz_mat @ np.array(
        [u_i_local[1], r_i_local[2], u_j_local[1], r_j_local[2]]
    )

    # bending deformation along the local xz plane
    d_z_local = nyz_mat @ np.array(
        [u_i_local[2], -r_i_local[1], u_j_local[2], -r_j_local[1]]
    )

    # torsional deformation
    r_x_local = nx_mat @ np.array([r_i_local[0], r_j_local[0]])

    # bending rotation around the local z axis
    r_z_local = (
        nyz_derivative_mat
        @ np.array([u_i_local[1], r_i_local[2], u_j_local[1], r_j_local[2]])
        / len_clr
    )

    # bending rotation around the local y axis
    r_y_local = (
        nyz_derivative_mat
        @ np.array([-u_i_local[2], r_i_local[1], -u_j_local[2], r_j_local[1]])
        / len_clr
    )

    # all deformations
    d_local = np.column_stack((d_x_local, d_y_local, d_z_local))

    # all rotations
    r_local = np.column_stack((r_x_local, r_y_local, r_z_local))

    d_global = (transf_local2global @ d_local.T).T

    return d_global, r_local


def interp_3d_points(elm, d_global, num_points, scaling):
    """
    Calculates intermediate points based on end locations and
    deformations.

    """

    if not isinstance(elm, element.TrussBar):
        p_i = np.array(elm.nodes[0].coords) + elm.geomtransf.offset_i
        p_j = np.array(elm.nodes[1].coords) + elm.geomtransf.offset_j
    else:
        p_i = np.array(elm.nodes[0].coords)
        p_j = np.array(elm.nodes[1].coords)
    element_point_samples: nparr = np.column_stack(
        (
            np.linspace(p_i[0], p_j[0], num=num_points),
            np.linspace(p_i[1], p_j[1], num=num_points),
            np.linspace(p_i[2], p_j[2], num=num_points),
        )
    )

    interpolation_points = element_point_samples + d_global * scaling

    return interpolation_points


def add_data__extruded_line_elms_deformed_mesh(
    analysis, case_name, data_dict, list_of_line_elems, step, scaling
):
    """
    Adds a trace containing frame element extrusion mesh
    in its deformed state.

    """

    if not list_of_line_elems:
        return
    x_list: list[float] = []
    y_list: list[float] = []
    z_list: list[float] = []
    i_list: list[int] = []
    j_list: list[int] = []
    k_list: list[int] = []
    intensity: list[float] = []
    index = 0
    for elm in list_of_line_elems:
        if elm.visibility.hidden_when_extruded:
            continue
        if isinstance(elm, element.TrussBar):
            if elm.outside_shape is None:
                continue
        else:
            if elm.section.outside_shape is None:
                continue

        num_points = 8
        # translations and rotations at the offset ends
        u_i = analysis.results[case_name].node_displacements[elm.nodes[0].uid][
            step
        ][0:3]
        r_i = analysis.results[case_name].node_displacements[elm.nodes[0].uid][
            step
        ][3:6]
        u_j = analysis.results[case_name].node_displacements[elm.nodes[1].uid][
            step
        ][0:3]
        r_j = analysis.results[case_name].node_displacements[elm.nodes[1].uid][
            step
        ][3:6]
        # transferring them to the clear element ends
        if isinstance(elm, element.TrussBar):
            offset_i = np.zeros(3)
            offset_j = np.zeros(3)
            x_vec, y_vec, z_vec = (
                transformations.local_axes_from_points_and_angle(
                    np.array(elm.nodes[0].coords),
                    np.array(elm.nodes[1].coords),
                    0.00))
            outside_shape = elm.outside_shape
        else:
            offset_i = elm.geomtransf.offset_i
            offset_j = elm.geomtransf.offset_j
            x_vec = elm.geomtransf.x_axis
            y_vec = elm.geomtransf.y_axis
            z_vec = elm.geomtransf.z_axis
            outside_shape = elm.section.outside_shape
        u_i_o = transformations.offset_transformation(offset_i, u_i, r_i)
        u_j_o = transformations.offset_transformation(offset_j, u_j, r_j)
        if isinstance(elm, element.TrussBar):
            d_global = np.column_stack((
                np.linspace(u_i[0], u_j[0], num_points),
                np.linspace(u_i[1], u_j[1], num_points),
                np.linspace(u_i[2], u_j[2], num_points),
            ))
            r_local = np.zeros((num_points, 3))
            interpolation_points = interp_3d_points(
                elm, d_global, num_points, scaling
            )
        else:
            d_global, r_local = interp_3d_deformation(
                elm, u_i_o, r_i, u_j_o, r_j, num_points)
            interpolation_points = interp_3d_points(
                elm, d_global, num_points, scaling
            )
        for i in range(num_points - 1):
            loc_i_global = interpolation_points[i, :]
            loc_j_global = interpolation_points[i + 1, :]
            rot_i_local = r_local[i, :]
            rot_j_local = r_local[i + 1, :]

            assert outside_shape is not None
            loop = outside_shape.halfedges
            for halfedge in loop:
                assert halfedge.nxt is not None
                z_a = halfedge.vertex.coords[0]
                y_a = halfedge.vertex.coords[1]
                z_b = halfedge.nxt.vertex.coords[0]
                y_b = halfedge.nxt.vertex.coords[1]
                defo_ia_global = (
                    z_a * z_vec
                    + y_a * y_vec
                    + (
                        -rot_i_local[2] * y_a * x_vec
                        + rot_i_local[1] * z_a * x_vec
                        + rot_i_local[0] * y_a * z_vec
                        - rot_i_local[0] * z_a * y_vec
                    )
                    * scaling
                )
                defo_ja_global = (
                    z_a * z_vec
                    + y_a * y_vec
                    + (
                        -rot_j_local[2] * y_a * x_vec
                        + rot_j_local[1] * z_a * x_vec
                        + rot_j_local[0] * y_a * z_vec
                        - rot_j_local[0] * z_a * y_vec
                    )
                    * scaling
                )
                defo_ib_global = (
                    z_b * z_vec
                    + y_b * y_vec
                    + (
                        -rot_i_local[2] * y_b * x_vec
                        + rot_i_local[1] * z_b * x_vec
                        + rot_i_local[0] * y_b * z_vec
                        - rot_i_local[0] * z_b * y_vec
                    )
                    * scaling
                )
                defo_jb_global = (
                    z_b * z_vec
                    + y_b * y_vec
                    + (
                        -rot_j_local[2] * y_b * x_vec
                        + rot_i_local[1] * z_b * x_vec
                        + rot_j_local[0] * y_b * z_vec
                        - rot_j_local[0] * z_b * y_vec
                    )
                    * scaling
                )
                loc0 = loc_i_global + defo_ia_global
                loc1 = loc_j_global + defo_ja_global
                loc2 = loc_j_global + defo_jb_global
                loc3 = loc_i_global + defo_ib_global
                x_list.extend((
                    loc0[0], loc1[0], loc2[0], loc3[0]))
                y_list.extend((
                    loc0[1], loc1[1], loc2[1], loc3[1]))
                z_list.extend((
                    loc0[2], loc1[2], loc2[2], loc3[2]))
                i_list.extend((
                    index + 0, index + 0))
                j_list.extend((
                    index + 1, index + 2))
                k_list.extend((
                    index + 2, index + 3))
                intensity.append(
                    float(np.sqrt(d_global[i, :] @ d_global[i, :])))
                intensity.append(
                    float(np.sqrt(d_global[i + 1, :] @ d_global[i + 1, :]))
                )
                intensity.append(
                    float(np.sqrt(d_global[i + 1, :] @ d_global[i + 1, :]))
                )
                intensity.append(
                    float(np.sqrt(d_global[i, :] @ d_global[i, :])))
                index += 4
    data_dict.append(
        {
            "type": "mesh3d",
            "x": x_list,
            "y": y_list,
            "z": z_list,
            "colorscale": [[0, "blue"], [1.0, "red"]],
            "i": i_list,
            "j": j_list,
            "k": k_list,
            "intensity": intensity,
            "colorbar": {"title": "Displacement", "ypad": 300},
            "hoverinfo": "skip",
            "opacity": 0.65,
        }
    )


def add_data__line_elms_deformed(
    analysis, case_name, data_dict, list_of_line_elems, step, scaling
):
    """
    Adds a trace containing frame element centroidal axis lines
    in their deformed state.

    """

    if not list_of_line_elems:
        return
    x_list: list[Optional[float]] = []
    y_list: list[Optional[float]] = []
    z_list: list[Optional[float]] = []

    for elm in list_of_line_elems:
        if elm.visibility.hidden_at_line_plots:
            continue
        num_points = 8
        u_i = analysis.results[case_name].node_displacements[elm.nodes[0].uid][
            step
        ][0:3]
        r_i = analysis.results[case_name].node_displacements[elm.nodes[0].uid][
            step
        ][3:6]
        u_j = analysis.results[case_name].node_displacements[elm.nodes[1].uid][
            step
        ][0:3]
        r_j = analysis.results[case_name].node_displacements[elm.nodes[1].uid][
            step
        ][3:6]
        if not isinstance(elm, element.TrussBar):
            # transferring them to the clear element ends
            offset_i = elm.geomtransf.offset_i
            offset_j = elm.geomtransf.offset_j
            u_i_o = transformations.offset_transformation(offset_i, u_i, r_i)
            u_j_o = transformations.offset_transformation(offset_j, u_j, r_j)
            d_global, _ = interp_3d_deformation(
                elm, u_i_o, r_i, u_j_o, r_j, num_points
            )
        else:
            # for a truss member, just connect the two ends
            d_global = np.column_stack((
                np.linspace(u_i[0], u_j[0], num_points),
                np.linspace(u_i[1], u_j[1], num_points),
                np.linspace(u_i[2], u_j[2], num_points),
            ))
        interpolation_points = interp_3d_points(
            elm, d_global, num_points, scaling
        )
        for i in range(len(interpolation_points) - 1):
            x_list.extend(
                (
                    interpolation_points[i, 0],
                    interpolation_points[i + 1, 0],
                    None,
                )
            )
            y_list.extend(
                (
                    interpolation_points[i, 1],
                    interpolation_points[i + 1, 1],
                    None,
                )
            )
            z_list.extend(
                (
                    interpolation_points[i, 2],
                    interpolation_points[i + 1, 2],
                    None,
                )
            )

    data_dict.append(
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x_list,
            "y": y_list,
            "z": z_list,
            "hoverinfo": "skip",
            "line": {"width": 5, "color": graphics_common.FRAME_COLOR},
        }
    )


def add_data__line_elm_offsets_deformed(
    analysis, case_name, data_dict, list_of_line_elems, step, scaling
):
    """
    Adds a trace containing frame element rigid offset lines
    in their deformed state.

    """

    if not list_of_line_elems:
        return
    x_list: list[Optional[float]] = []
    y_list: list[Optional[float]] = []
    z_list: list[Optional[float]] = []
    for elm in list_of_line_elems:
        if isinstance(elm, element.TrussBar):
            continue
        if np.array_equal(elm.geomtransf.offset_i, np.zeros(3)):
            if np.array_equal(elm.geomtransf.offset_j, np.zeros(3)):
                continue
        p_i: nparr = np.array(elm.nodes[0].coords)
        p_io: nparr = np.array(elm.nodes[0].coords) + elm.geomtransf.offset_i
        offset_i = elm.geomtransf.offset_i
        u_i: nparr = np.array(
            analysis.results[case_name].node_displacements[elm.nodes[0].uid][
                step
            ][0:3]
        )
        r_i: nparr = np.array(
            analysis.results[case_name].node_displacements[elm.nodes[0].uid][
                step
            ][3:6]
        )
        u_io: nparr = transformations.offset_transformation(offset_i, u_i, r_i)

        p_j: nparr = np.array(elm.nodes[1].coords)
        p_jo: nparr = np.array(elm.nodes[1].coords) + elm.geomtransf.offset_j
        offset_j = elm.geomtransf.offset_j
        u_j: nparr = np.array(
            analysis.results[case_name].node_displacements[elm.nodes[1].uid][
                step
            ][0:3]
        )
        r_j: nparr = np.array(
            analysis.results[case_name].node_displacements[elm.nodes[1].uid][
                step
            ][3:6]
        )
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

    data_dict.append(
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x_list,
            "y": y_list,
            "z": z_list,
            "hoverinfo": "skip",
            "line": {"width": 8, "color": graphics_common.OFFSET_COLOR},
        }
    )


def add_data__frames_undeformed(data_dict, list_of_line_elems):
    """
    Adds a trace containing frame element centroidal axis lines

    """

    x_list: list[Optional[float]] = []
    y_list: list[Optional[float]] = []
    z_list: list[Optional[float]] = []

    for elm in list_of_line_elems:

        if isinstance(elm, element.TrussBar):
            p_i = np.array(elm.nodes[0].coords)
            p_j = np.array(elm.nodes[1].coords)
        else:
            p_i = np.array(elm.nodes[0].coords) + elm.geomtransf.offset_i
            p_j = np.array(elm.nodes[1].coords) + elm.geomtransf.offset_j

        x_list.extend((p_i[0], p_j[0], None))
        y_list.extend((p_i[1], p_j[1], None))
        z_list.extend((p_i[2], p_j[2], None))
    data_dict.append(
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x_list,
            "y": y_list,
            "z": z_list,
            "hoverinfo": "skip",
            "line": {"width": 5, "color": graphics_common.BEAM_MESH_COLOR},
        }
    )


def add_data__nodes_deformed(
    analysis, case_name, data_dict, list_of_nodes, step, scaling, function
):
    """
    Adds a trace containing nodes in their deformed locations.

    """
    ids_list = [int(node.uid) for node in list_of_nodes]
    location_data = np.full((len(list_of_nodes), 3), 0.00)
    displacement_data = np.full((len(list_of_nodes), 6), 0.00)
    for i, node in enumerate(list_of_nodes):
        location_data[i, :] = node.coords
        displacement_data[i, :] = analysis.results[
            case_name
        ].node_displacements[node.uid][step]
    dist = np.sqrt(
        displacement_data[:, 0] ** 2
        + displacement_data[:, 1] ** 2
        + displacement_data[:, 2] ** 2
    )
    dist = np.reshape(dist, (-1, 1))
    ids = np.reshape(np.array(ids_list), (-1, 1))
    displacement_data = np.concatenate((displacement_data, dist, ids), 1)
    restraint_symbols = []
    for node in list_of_nodes:
        if True in node.restraint:
            restraint_symbols.append("fixed")
        else:
            restraint_symbols.append(function)

    marker = [
        graphics_common_3d.node_marker[sym][0] for sym in restraint_symbols
    ]
    size = [
        graphics_common_3d.node_marker[sym][1] for sym in restraint_symbols
    ]
    if function == "internal":
        color = graphics_common.NODE_INTERNAL_COLOR
    else:
        color = graphics_common.NODE_PRIMARY_COLOR

    data_dict.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": location_data[:, 0] + displacement_data[:, 0] * scaling,
            "y": location_data[:, 1] + displacement_data[:, 1] * scaling,
            "z": location_data[:, 2] + displacement_data[:, 2] * scaling,
            "customdata": displacement_data,
            "hovertemplate": "ux: %{customdata[0]:.6g}<br>"
            + "uy: %{customdata[1]:.6g}<br>"
            + "uz: %{customdata[2]:.6g}<br>"
            + "combined: %{customdata[6]:.6g}<br>"
            + "rx: %{customdata[3]:.6g} (rad)<br>"
            + "ry: %{customdata[4]:.6g} (rad)<br>"
            + "rz: %{customdata[5]:.6g} (rad)<br>"
            + "<extra>Node %{customdata[7]:d}</extra>",
            "marker": {
                "symbol": marker,
                "color": color,
                "size": size,
                "line": {"color": color, "width": 4},
            },
        }
    )


def get_auto_scaling_deformation(analysis, case_name, mdl, step):
    """
    Automatically calculate a scaling value that makes the maximum
    displacement appear approximately 10% of the largest dimention of
    the building's bounding box.

    """

    ref_len = mdl.reference_length()
    # maximum displacement
    max_d = 0.00
    elms: list[
        Union[element.ElasticBeamColumn,
              element.DispBeamColumn]] = []
    elms.extend(
        mdl.list_of_specific_element(element.ElasticBeamColumn))
    elms.extend(
        mdl.list_of_specific_element(element.DispBeamColumn))

    for elm in elms:
        u_i = analysis.results[case_name].node_displacements[elm.nodes[0].uid][
            step
        ][0:3]
        r_i = analysis.results[case_name].node_displacements[elm.nodes[0].uid][
            step
        ][3:6]
        u_j = analysis.results[case_name].node_displacements[elm.nodes[1].uid][
            step
        ][0:3]
        r_j = analysis.results[case_name].node_displacements[elm.nodes[1].uid][
            step
        ][3:6]
        d_global, _ = interp_3d_deformation(elm, u_i, r_i, u_j, r_j, 3)
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


def show_deformed_shape(
        analysis: Analysis,
        case_name: str,
        step: int,
        scaling: float,
        extrude: bool,
        camera: Optional[dict[str, object]] = None,
        subset_model: Model = None,
        animation: bool = False,
        init_step: int = 0,
        step_skip: int = 0,
        to_html_file: Optional[str] = None
) -> dict[str, Any]:

    """
    Visualize the model in its deformed state

    Arguments:
      analysis: an analysis object
      case_name: the name of the load_case to be visualized
      step: the analysis step to be visualized
      scaling: scaling factor for the deformations. If 0.00 is
        provided, the scaling factor is calculated automatically.
      extrude: wether to extrude frame elements
      camera: custom positioning of the camera
      subset_model: subset model used to only show certain
        components
      animation: show all frames up to the one identified with
        `step`
      init_step: starting step, in case of animation
      step_skip: how many frames to skip to reduce the number of
        frames in case an animation.
      to_html_file: If a path is specified, the figure is written in
        an html file instead of being shown.

    """

    if subset_model:
        # if a subset model is specified, only show its components
        mdl = subset_model
    else:
        # otherwise show the entire model
        mdl = analysis.mdl

    # calculate a nice scaling factor if 0.00 is passed
    if scaling == 0.00:
        scaling = get_auto_scaling_deformation(analysis, case_name, mdl, step)

    # instantiate layout and datastructures
    layout = graphics_common_3d.global_layout(mdl, camera)
    data_dict: list[dict[str, object]] = []
    frame_data_dict: list[list[dict[str, object]]] = []

    # gather lists of associated objects
    list_of_line_elems: list[element.Element] = []
    list_of_line_elems.extend(
        mdl.list_of_specific_element(element.TrussBar))
    list_of_line_elems.extend(
        mdl.list_of_specific_element(element.ElasticBeamColumn))
    list_of_line_elems.extend(
        mdl.list_of_specific_element(element.DispBeamColumn))
    list_of_primary_nodes = mdl.list_of_primary_nodes()
    list_of_internal_nodes = mdl.list_of_internal_nodes()
    # list_of_parent_nodes = mdl.list_of_parent_nodes()

    # add data for the global axes
    ref_len = mdl.reference_length()

    add_data__global_axes(data_dict, ref_len)
    if animation:
        first_step = init_step
    else:
        first_step = step

    add_data__nodes_deformed(
        analysis,
        case_name,
        data_dict,
        list_of_primary_nodes,
        first_step,
        scaling,
        "free",
    )
    add_data__nodes_deformed(
        analysis,
        case_name,
        data_dict,
        list_of_internal_nodes,
        first_step,
        scaling,
        "internal",
    )
    # add data for the frame elements as lines
    add_data__line_elm_offsets_deformed(
        analysis, case_name, data_dict, list_of_line_elems, first_step, scaling
    )
    add_data__line_elms_deformed(
        analysis, case_name, data_dict, list_of_line_elems, first_step, scaling
    )
    # add data for the extruded frame elements
    if extrude:
        add_data__extruded_line_elms_deformed_mesh(
            analysis,
            case_name,
            data_dict,
            list_of_line_elems,
            first_step,
            scaling,
        )

    # create the plot

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    if animation:
        step_of_frame = []
        for j in range(first_step, step+1, step_skip + 1):
            step_of_frame.append(j)
        for _, j in enumerate(step_of_frame):
            frame_data_dict.append([])
            add_data__global_axes(frame_data_dict[-1], ref_len)
            # add_data__global_axes(frame_data_dict[0], ref_len)
            add_data__nodes_deformed(
                analysis,
                case_name,
                frame_data_dict[-1],
                list_of_primary_nodes,
                j,
                scaling,
                "free",
            )
            add_data__nodes_deformed(
                analysis,
                case_name,
                frame_data_dict[-1],
                list_of_internal_nodes,
                j,
                scaling,
                "internal",
            )
            # add data for the frame elements as lines
            add_data__line_elm_offsets_deformed(
                analysis,
                case_name,
                frame_data_dict[-1],
                list_of_line_elems,
                j,
                scaling,
            )
            add_data__line_elms_deformed(
                analysis,
                case_name,
                frame_data_dict[-1],
                list_of_line_elems,
                j,
                scaling,
            )
            # add data for the extruded frame elements
            if extrude:
                add_data__extruded_line_elms_deformed_mesh(
                    analysis,
                    case_name,
                    frame_data_dict[-1],
                    list_of_line_elems,
                    j,
                    scaling,
                )

        fig_datastructure = dict(
            data=data_dict,
            layout=layout,
            frames=[
                go.Frame(data=frame_data_dict[j], name=str(step_of_frame[j]))
                for j in range(len(step_of_frame))
            ],
        )

        fig = go.Figure(fig_datastructure)
        sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]
        # Layout
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;",  # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;",  # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders,
        )
    else:  # (not animation)
        fig_datastructure = dict(data=data_dict, layout=layout)
        fig = go.Figure(fig_datastructure)

    # show the plot (if it's not a test)
    if "pytest" not in sys.modules:
        if to_html_file:
            fig.write_html(to_html_file)
        else:
            fig.show()

    # return plot-related metadata
    metadata = {"scaling": scaling}
    return metadata


def show_basic_forces(
    analysis,
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
    subset_model=None,
    to_html_file=None
):
    """
    Visualize the model and plot the frame element basic forces.

    Arguments:
      analysis: an analysis object
      case_name: the name of the load_case to be visualized
      step: the analysis step to be visualized
      scaling_global: I don't even remember what this
        does. It's kind of a mess right now.
      scaling_n:
      scaling_q:
      scaling_m:
      scaling_t:
      num_points: number of points to include in the basic force
        curves
      force_conversion: Conversion factor to be applied at the
        hover box data for forces (for unit conversions)
      moment_conversion: Conversion factor to be applied at the
        hover box data for moments (for unit conversions)
      global_axes: whether to show global axes
      camera: custom positioning of the camera
      subset_model: use this model instead of the one contained in the
        analysis object. It needs to be a subset of the original
        model. This can be used to only show the results for some part
        of a large model.
      to_html_file: If a path is specified, the figure is written in
        an html file instead of being shown.

    """

    if subset_model:
        mdl = subset_model
    else:
        mdl = analysis.mdl

    elms: list[
        Union[element.TrussBar,
              element.ElasticBeamColumn,
              element.DispBeamColumn]] = []
    elms.extend(
        mdl.list_of_specific_element(element.TrussBar))
    elms.extend(
        mdl.list_of_specific_element(element.ElasticBeamColumn))
    elms.extend(
        mdl.list_of_specific_element(element.DispBeamColumn))

    list_of_line_elements = [
        elm
        for elm in elms
        if not elm.visibility.skip_opensees_definition
    ]

    layout = graphics_common_3d.global_layout(mdl, camera)
    data_dict: list[dict[str, object]] = []

    # draw the frames
    add_data__frames_undeformed(data_dict, list_of_line_elements)
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

    for elm in list_of_line_elements:

        if elm.visibility.hidden_basic_forces:
            continue

        forces = basic_forces(
            analysis, case_name, step, elm, num_points, as_tuple=True
        )
        assert isinstance(forces, tuple)
        nx_vec, qy_vec, qz_vec, tx_vec, mz_vec, my_vec = forces
        assert isinstance(nx_vec, np.ndarray)
        assert isinstance(qy_vec, np.ndarray)
        assert isinstance(qz_vec, np.ndarray)
        assert isinstance(tx_vec, np.ndarray)
        assert isinstance(mz_vec, np.ndarray)
        assert isinstance(my_vec, np.ndarray)

        # store results in the preallocated arrays

        nx_vecs[elm.uid] = nx_vec * force_conversion
        qy_vecs[elm.uid] = qy_vec * force_conversion
        qz_vecs[elm.uid] = qz_vec * force_conversion
        tx_vecs[elm.uid] = tx_vec * moment_conversion
        my_vecs[elm.uid] = my_vec * moment_conversion
        mz_vecs[elm.uid] = mz_vec * moment_conversion

    # calculate scaling factors
    ref_len = mdl.reference_length()
    factor = 0.05
    nx_max = np.max(np.abs(np.column_stack(list(nx_vecs.values()))))
    scaling_n = force_scaling_factor(ref_len, nx_max, factor)
    if scaling_n > 1.0e8:
        scaling_t = 1.00
    qy_max = np.max(np.abs(np.column_stack(list(qy_vecs.values()))))
    qz_max = np.max(np.abs(np.column_stack(list(qz_vecs.values()))))
    scaling_qy = force_scaling_factor(ref_len, qy_max, factor)
    scaling_qz = force_scaling_factor(ref_len, qz_max, factor)
    if scaling_qy > 0.00 and scaling_qz > 0.00:
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
    if scaling_my > 0.00 and scaling_mz > 0.00:
        scaling_m = np.minimum(scaling_my, scaling_mz)
    elif scaling_my == 0.00:
        scaling_m = scaling_mz
    elif scaling_mz == 0.00:
        scaling_m = scaling_my
    else:
        scaling_m = 0.00
    if scaling_m > 1.0e8:
        scaling_m = 1.00

    for elm in list_of_line_elements:

        if elm.visibility.hidden_basic_forces:
            continue

        # retrieve results from the preallocated arrays
        nx_vec = nx_vecs[elm.uid]
        qy_vec = qy_vecs[elm.uid]
        qz_vec = qz_vecs[elm.uid]
        tx_vec = tx_vecs[elm.uid]
        my_vec = my_vecs[elm.uid]
        mz_vec = mz_vecs[elm.uid]
        if isinstance(
                elm, (element.ElasticBeamColumn, element.DispBeamColumn)):
            x_vec = elm.geomtransf.x_axis
            y_vec = elm.geomtransf.y_axis
            z_vec = elm.geomtransf.z_axis
            i_pos = np.array(elm.nodes[0].coords) + elm.geomtransf.offset_i
            p_i = np.array(elm.nodes[0].coords) + elm.geomtransf.offset_i
            p_j = np.array(elm.nodes[1].coords) + elm.geomtransf.offset_j
        else:
            p_i = np.array(elm.nodes[0].coords)
            p_j = np.array(elm.nodes[1].coords)
            i_pos = np.array(elm.nodes[0].coords)
            x_vec, y_vec, z_vec = (
                transformations.local_axes_from_points_and_angle(
                    p_i, p_j, 0.00))

        len_clr = np.linalg.norm(p_i - p_j)
        t_vec = np.linspace(0.00, len_clr, num=num_points)

        for i in range(num_points - 1):

            p_start = i_pos + t_vec[i] * x_vec
            p_end = i_pos + t_vec[i + 1] * x_vec

            # axial load
            p_i = p_start + nx_vec[i] * y_vec * scaling_n * scaling_global
            p_j = p_end + nx_vec[i + 1] * y_vec * scaling_n * scaling_global

            x1_a.extend((p_i[0], p_j[0], None))
            y1_a.extend((p_i[1], p_j[1], None))
            z1_a.extend((p_i[2], p_j[2], None))
            customdata_a.extend((nx_vec[i], nx_vec[i + 1], None))
            colors1_a.extend(["red"] * 3)

            # torsion
            p_i = p_start + tx_vec[i] * z_vec * scaling_t * scaling_global
            p_j = p_end + tx_vec[i + 1] * z_vec * scaling_t * scaling_global
            x1_d.extend((p_i[0], p_j[0], None))
            y1_d.extend((p_i[1], p_j[1], None))
            z1_d.extend((p_i[2], p_j[2], None))
            customdata_d.extend((tx_vec[i], tx_vec[i + 1], None))
            colors1_d.extend(["orange"] * 3)

            # shear load on y and z axes
            p_i = p_start + qy_vec[i] * y_vec * scaling_q * scaling_global
            p_j = p_end + qy_vec[i + 1] * y_vec * scaling_q * scaling_global
            x1_b.extend((p_i[0], p_j[0], None))
            y1_b.extend((p_i[1], p_j[1], None))
            z1_b.extend((p_i[2], p_j[2], None))
            customdata_b.extend((qy_vec[i], qy_vec[i + 1], None))
            colors1_b.extend(["green"] * 3)
            p_i = p_start + qz_vec[i] * z_vec * scaling_q * scaling_global
            p_j = p_end + qz_vec[i + 1] * z_vec * scaling_q * scaling_global
            x1_b.extend((p_i[0], p_j[0], None))
            y1_b.extend((p_i[1], p_j[1], None))
            z1_b.extend((p_i[2], p_j[2], None))
            customdata_b.extend((qz_vec[i], qz_vec[i + 1], None))
            colors1_b.extend(["green"] * 3)

            # moment around z and y axes
            p_i = p_start - mz_vec[i] * y_vec * scaling_m * scaling_global
            p_j = p_end - mz_vec[i + 1] * y_vec * scaling_m * scaling_global
            # note: moments plotted upside down!
            x1_c.extend((p_i[0], p_j[0], None))
            y1_c.extend((p_i[1], p_j[1], None))
            z1_c.extend((p_i[2], p_j[2], None))
            customdata_c.extend((mz_vec[i], mz_vec[i + 1], None))
            colors1_c.extend(["blue"] * 3)
            # moment around y axis
            p_i = p_start - my_vec[i] * z_vec * scaling_m * scaling_global
            p_j = p_end - my_vec[i + 1] * z_vec * scaling_m * scaling_global
            x1_c.extend((p_i[0], p_j[0], None))
            y1_c.extend((p_i[1], p_j[1], None))
            z1_c.extend((p_i[2], p_j[2], None))
            customdata_c.extend((my_vec[i], my_vec[i + 1], None))
            colors1_c.extend(["blue"] * 3)

            # shear load combined
            p_i = (
                p_start
                + (qy_vec[i] * y_vec + qz_vec[i] * z_vec)
                * scaling_q
                * scaling_global
            )
            p_j = (
                p_end
                + (qy_vec[i + 1] * y_vec + qz_vec[i + 1] * z_vec)
                * scaling_q
                * scaling_global
            )
            x1_e.extend((p_i[0], p_j[0], None))
            y1_e.extend((p_i[1], p_j[1], None))
            z1_e.extend((p_i[2], p_j[2], None))
            customdata_e.extend(
                (
                    np.sqrt(qy_vec[i] ** 2 + qz_vec[i] ** 2),
                    np.sqrt(qy_vec[i + 1] ** 2 + qz_vec[i + 1] ** 2),
                    None,
                )
            )
            colors1_e.extend(["green"] * 3)

            # both moments combined!
            p_i = (
                p_start
                - mz_vec[i] * y_vec * scaling_m
                - my_vec[i] * z_vec * scaling_m * scaling_global
            )
            p_j = (
                p_end
                - mz_vec[i + 1] * y_vec * scaling_m
                - my_vec[i + 1] * z_vec * scaling_m * scaling_global
            )
            # note: moments plotted upside down!
            x1_f.extend((p_i[0], p_j[0], None))
            y1_f.extend((p_i[1], p_j[1], None))
            z1_f.extend((p_i[2], p_j[2], None))
            customdata_f.extend(
                (
                    np.sqrt(mz_vec[i] ** 2 + my_vec[i] ** 2),
                    np.sqrt(mz_vec[i + 1] ** 2 + my_vec[i + 1] ** 2),
                    None,
                )
            )
            colors1_f.extend(["blue"] * 3)

    dt_a = [
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x1_a,
            "y": y1_a,
            "z": z1_a,
            "visible": False,
            "customdata": customdata_a,
            "hovertemplate": " %{customdata:.0f}<br>" "<extra></extra>",
            "line": {"width": 3, "color": colors1_a},
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
            "hovertemplate": " %{customdata:.0f}<br>" "<extra></extra>",
            "line": {"width": 3, "color": colors1_b},
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
            "hovertemplate": " %{customdata:.0f}<br>" "<extra></extra>",
            "line": {"width": 3, "color": colors1_c},
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
            "hovertemplate": " %{customdata:.0f}<br>" "<extra></extra>",
            "line": {"width": 3, "color": colors1_d},
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
            "hovertemplate": " %{customdata:.0f}<br>" "<extra></extra>",
            "line": {"width": 3, "color": colors1_e},
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
            "hovertemplate": " %{customdata:.0f}<br>" "<extra></extra>",
            "line": {"width": 3, "color": colors1_f},
        }
    ]

    fig_datastructure = dict(
        data=data_dict + dt_a + dt_b + dt_c + dt_d + dt_e + dt_f, layout=layout
    )
    fig = go.Figure(fig_datastructure)
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="None",
                        method="update",
                        args=[
                            {"visible": [True] * len(data_dict) + [False] * 6}
                        ],
                    ),
                    dict(
                        label="Axial",
                        method="update",
                        args=[
                            {
                                "visible": [True] * len(data_dict)
                                + [True]
                                + [False] * 5
                            }
                        ],
                    ),
                    dict(
                        label="Shear",
                        method="update",
                        args=[
                            {
                                "visible": [True] * len(data_dict)
                                + [False]
                                + [True]
                                + [False] * 4
                            }
                        ],
                    ),
                    dict(
                        label="Moment",
                        method="update",
                        args=[
                            {
                                "visible": [True] * len(data_dict)
                                + [False] * 2
                                + [True]
                                + [False] * 3
                            }
                        ],
                    ),
                    dict(
                        label="Torsion",
                        method="update",
                        args=[
                            {
                                "visible": [True] * len(data_dict)
                                + [False] * 3
                                + [True]
                                + [False] * 2
                            }
                        ],
                    ),
                    dict(
                        label="Shear (combined)",
                        method="update",
                        args=[
                            {
                                "visible": [True] * len(data_dict)
                                + [False] * 4
                                + [True]
                                + [False]
                            }
                        ],
                    ),
                    dict(
                        label="Moment (combined)",
                        method="update",
                        args=[
                            {
                                "visible": [True] * len(data_dict)
                                + [False] * 5
                                + [True]
                            }
                        ],
                    ),
                ],
            )
        ]
    )

    if "pytest" not in sys.modules:
        if to_html_file:
            fig.write_html(to_html_file)
        else:
            fig.show()

    metadata = {
        "scaling_n": scaling_n,
        "scaling_q": scaling_q,
        "scaling_m": scaling_m,
        "scaling_t": scaling_t,
    }
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
    subset_model=None,
    to_html_file=None
):
    """
    Visualize the model and plot the enveloped frame element basic forces
    for a load combination.

    Arguments:
      combo: a load combination object
      step: the analysis step to be visualized
      scaling_global: I don't even remember what this
        does. It's kind of a mess right now.
      scaling_n:
      scaling_q:
      scaling_m:
      scaling_t:
      num_points: number of points to include in the basic force
        curves
      force_conversion: Conversion factor to be applied at the
        hover box data for forces (for unit conversions)
      moment_conversion: Conversion factor to be applied at the
        hover box data for moments (for unit conversions)
      global_axes: whether to show global axes
      camera: custom positioning of the camera
      subset_model: use this model instead of the one contained in the
        analysis object. It needs to be a subset of the original
        model. This can be used to only show the results for some part
        of a large model.
      to_html_file: If a path is specified, the figure is written in
        an html file instead of being shown.

    """

    # TODO: merge code repetitions with the previous function

    if subset_model:
        mdl = subset_model
    else:
        mdl = combo.mdl
    elms: list[
        Union[element.TrussBar,
              element.ElasticBeamColumn,
              element.DispBeamColumn]] = []
    elms.extend(
        mdl.list_of_specific_element(element.TrussBar))
    elms.extend(
        mdl.list_of_specific_element(element.ElasticBeamColumn))
    elms.extend(
        mdl.list_of_specific_element(element.DispBeamColumn))
    list_of_line_elements = [
        elm
        for elm in elms
        if not elm.visibility.skip_opensees_definition
    ]

    layout = graphics_common_3d.global_layout(mdl, camera)
    data_dict: list[dict[str, object]] = []

    # draw the frames
    add_data__frames_undeformed(data_dict, list_of_line_elements)
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

    for elm in list_of_line_elements:

        if elm.visibility.hidden_basic_forces:
            continue

        df_min, df_max = combo.envelope_basic_forces(elm, num_points)

        # store results in the preallocated arrays

        nx_vecs_min[elm.uid] = df_min["nx"].to_numpy() * force_conversion
        qy_vecs_min[elm.uid] = df_min["qy"].to_numpy() * force_conversion
        qz_vecs_min[elm.uid] = df_min["qz"].to_numpy() * force_conversion
        tx_vecs_min[elm.uid] = df_min["tx"].to_numpy() * moment_conversion
        my_vecs_min[elm.uid] = df_min["my"].to_numpy() * moment_conversion
        mz_vecs_min[elm.uid] = df_min["mz"].to_numpy() * moment_conversion
        nx_vecs_max[elm.uid] = df_max["nx"].to_numpy() * force_conversion
        qy_vecs_max[elm.uid] = df_max["qy"].to_numpy() * force_conversion
        qz_vecs_max[elm.uid] = df_max["qz"].to_numpy() * force_conversion
        tx_vecs_max[elm.uid] = df_max["tx"].to_numpy() * moment_conversion
        my_vecs_max[elm.uid] = df_max["my"].to_numpy() * moment_conversion
        mz_vecs_max[elm.uid] = df_max["mz"].to_numpy() * moment_conversion

    # calculate scaling factors
    ref_len = mdl.reference_length()
    factor = 0.05
    nx_max = np.max(
        np.abs(
            np.column_stack(
                list(nx_vecs_min.values()) + list(nx_vecs_max.values())
            )
        )
    )
    scaling_n = force_scaling_factor(ref_len, nx_max, factor)
    if scaling_n > 1.0e8:
        scaling_t = 1.00
    qy_max = np.max(
        np.abs(
            np.column_stack(
                list(qy_vecs_min.values()) + list(qy_vecs_max.values())
            )
        )
    )
    qz_max = np.max(
        np.abs(
            np.column_stack(
                list(qz_vecs_min.values()) + list(qz_vecs_max.values())
            )
        )
    )
    scaling_qy = force_scaling_factor(ref_len, qy_max, factor)
    scaling_qz = force_scaling_factor(ref_len, qz_max, factor)
    if scaling_qy > 0.00 and scaling_qz > 0.00:
        scaling_q = np.minimum(scaling_qy, scaling_qz)
    elif scaling_qy == 0.00:
        scaling_q = scaling_qz
    elif scaling_qz == 0.00:
        scaling_q = scaling_qy
    else:
        scaling_q = 0.00
    if scaling_q > 1.0e8:
        scaling_q = 1.00
    my_max = np.max(
        np.abs(
            np.column_stack(
                list(my_vecs_min.values()) + list(my_vecs_max.values())
            )
        )
    )
    mz_max = np.max(
        np.abs(
            np.column_stack(
                list(mz_vecs_min.values()) + list(mz_vecs_max.values())
            )
        )
    )
    scaling_my = force_scaling_factor(ref_len, my_max, factor)
    scaling_mz = force_scaling_factor(ref_len, mz_max, factor)
    if scaling_my > 0.00 and scaling_mz > 0.00:
        scaling_m = np.minimum(scaling_my, scaling_mz)
    elif scaling_my == 0.00:
        scaling_m = scaling_mz
    elif scaling_mz == 0.00:
        scaling_m = scaling_my
    else:
        scaling_m = 0.00
    if scaling_m > 1.0e8:
        scaling_m = 1.00

    for elm in list_of_line_elements:

        if elm.visibility.hidden_basic_forces:
            continue

        # retrieve results from the preallocated arrays
        nx_vec_min = nx_vecs_min[elm.uid]
        qy_vec_min = qy_vecs_min[elm.uid]
        qz_vec_min = qz_vecs_min[elm.uid]
        tx_vec_min = tx_vecs_min[elm.uid]
        my_vec_min = my_vecs_min[elm.uid]
        mz_vec_min = mz_vecs_min[elm.uid]

        nx_vec_max = nx_vecs_max[elm.uid]
        qy_vec_max = qy_vecs_max[elm.uid]
        qz_vec_max = qz_vecs_max[elm.uid]
        tx_vec_max = tx_vecs_max[elm.uid]
        my_vec_max = my_vecs_max[elm.uid]
        mz_vec_max = mz_vecs_max[elm.uid]

        if isinstance(
                elm, (element.ElasticBeamColumn, element.DispBeamColumn)):
            x_vec = elm.geomtransf.x_axis
            y_vec = elm.geomtransf.y_axis
            z_vec = elm.geomtransf.z_axis
            i_pos = np.array(elm.nodes[0].coords) + elm.geomtransf.offset_i
            p_i = np.array(elm.nodes[0].coords) + elm.geomtransf.offset_i
            p_j = np.array(elm.nodes[1].coords) + elm.geomtransf.offset_j
        else:
            p_i = np.array(elm.nodes[0].coords)
            p_j = np.array(elm.nodes[1].coords)
            i_pos = np.array(elm.nodes[0].coords)
            x_vec, y_vec, z_vec = (
                transformations.local_axes_from_points_and_angle(
                    p_i, p_j, 0.00))

        len_clr = np.linalg.norm(p_i - p_j)
        t_vec = np.linspace(0.00, len_clr, num=num_points)

        for i in range(num_points - 1):

            p_start = i_pos + t_vec[i] * x_vec
            p_end = i_pos + t_vec[i + 1] * x_vec

            # axial load
            p_i = p_start + nx_vec_min[i] * y_vec * scaling_n * scaling_global
            p_j = (
                p_end + nx_vec_min[i + 1] * y_vec * scaling_n * scaling_global
            )
            x1_a.extend((p_i[0], p_j[0], None))
            y1_a.extend((p_i[1], p_j[1], None))
            z1_a.extend((p_i[2], p_j[2], None))
            customdata_a.extend((nx_vec_min[i], nx_vec_min[i + 1], None))
            colors1_a.extend(["red"] * 3)
            p_i = p_start + nx_vec_max[i] * y_vec * scaling_n * scaling_global
            p_j = (
                p_end + nx_vec_max[i + 1] * y_vec * scaling_n * scaling_global
            )
            x1_a.extend((p_i[0], p_j[0], None))
            y1_a.extend((p_i[1], p_j[1], None))
            z1_a.extend((p_i[2], p_j[2], None))
            customdata_a.extend((nx_vec_max[i], nx_vec_max[i + 1], None))
            colors1_a.extend(["green"] * 3)

            # torsion
            p_i = p_start + tx_vec_min[i] * z_vec * scaling_t * scaling_global
            p_j = (
                p_end + tx_vec_min[i + 1] * z_vec * scaling_t * scaling_global
            )
            x1_d.extend((p_i[0], p_j[0], None))
            y1_d.extend((p_i[1], p_j[1], None))
            z1_d.extend((p_i[2], p_j[2], None))
            customdata_d.extend((tx_vec_min[i], tx_vec_min[i + 1], None))
            colors1_d.extend(["red"] * 3)
            p_i = p_start + tx_vec_max[i] * z_vec * scaling_t * scaling_global
            p_j = (
                p_end + tx_vec_max[i + 1] * z_vec * scaling_t * scaling_global
            )
            x1_d.extend((p_i[0], p_j[0], None))
            y1_d.extend((p_i[1], p_j[1], None))
            z1_d.extend((p_i[2], p_j[2], None))
            customdata_d.extend((tx_vec_max[i], tx_vec_max[i + 1], None))
            colors1_d.extend(["green"] * 3)

            # shear load on y and z axes
            p_i = p_start + qy_vec_min[i] * y_vec * scaling_q * scaling_global
            p_j = (
                p_end + qy_vec_min[i + 1] * y_vec * scaling_q * scaling_global
            )
            x1_b.extend((p_i[0], p_j[0], None))
            y1_b.extend((p_i[1], p_j[1], None))
            z1_b.extend((p_i[2], p_j[2], None))
            customdata_b.extend((qy_vec_min[i], qy_vec_min[i + 1], None))
            colors1_b.extend(["red"] * 3)
            p_i = p_start + qy_vec_max[i] * y_vec * scaling_q * scaling_global
            p_j = (
                p_end + qy_vec_max[i + 1] * y_vec * scaling_q * scaling_global
            )
            x1_b.extend((p_i[0], p_j[0], None))
            y1_b.extend((p_i[1], p_j[1], None))
            z1_b.extend((p_i[2], p_j[2], None))
            customdata_b.extend((qy_vec_max[i], qy_vec_max[i + 1], None))
            colors1_b.extend(["green"] * 3)
            p_i = p_start + qz_vec_min[i] * z_vec * scaling_q * scaling_global
            p_j = (
                p_end + qz_vec_min[i + 1] * z_vec * scaling_q * scaling_global
            )
            x1_b.extend((p_i[0], p_j[0], None))
            y1_b.extend((p_i[1], p_j[1], None))
            z1_b.extend((p_i[2], p_j[2], None))
            customdata_b.extend((qz_vec_min[i], qz_vec_min[i + 1], None))
            colors1_b.extend(["red"] * 3)
            p_i = p_start + qz_vec_max[i] * z_vec * scaling_q * scaling_global
            p_j = (
                p_end + qz_vec_max[i + 1] * z_vec * scaling_q * scaling_global
            )
            x1_b.extend((p_i[0], p_j[0], None))
            y1_b.extend((p_i[1], p_j[1], None))
            z1_b.extend((p_i[2], p_j[2], None))
            customdata_b.extend((qz_vec_max[i], qz_vec_max[i + 1], None))
            colors1_b.extend(["green"] * 3)

            # moment around z and y axes
            p_i = p_start - mz_vec_min[i] * y_vec * scaling_m * scaling_global
            p_j = (
                p_end - mz_vec_min[i + 1] * y_vec * scaling_m * scaling_global
            )
            # note: moments plotted upside down!
            x1_c.extend((p_i[0], p_j[0], None))
            y1_c.extend((p_i[1], p_j[1], None))
            z1_c.extend((p_i[2], p_j[2], None))
            customdata_c.extend((mz_vec_min[i], mz_vec_min[i + 1], None))
            colors1_c.extend(["red"] * 3)
            p_i = p_start - mz_vec_max[i] * y_vec * scaling_m * scaling_global
            p_j = (
                p_end - mz_vec_max[i + 1] * y_vec * scaling_m * scaling_global
            )
            # note: moments plotted upside down!
            x1_c.extend((p_i[0], p_j[0], None))
            y1_c.extend((p_i[1], p_j[1], None))
            z1_c.extend((p_i[2], p_j[2], None))
            customdata_c.extend((mz_vec_max[i], mz_vec_max[i + 1], None))
            colors1_c.extend(["green"] * 3)
            # moment around y axis
            p_i = p_start - my_vec_min[i] * z_vec * scaling_m * scaling_global
            p_j = (
                p_end - my_vec_min[i + 1] * z_vec * scaling_m * scaling_global
            )
            x1_c.extend((p_i[0], p_j[0], None))
            y1_c.extend((p_i[1], p_j[1], None))
            z1_c.extend((p_i[2], p_j[2], None))
            customdata_c.extend((my_vec_min[i], my_vec_min[i + 1], None))
            colors1_c.extend(["red"] * 3)
            p_i = p_start - my_vec_max[i] * z_vec * scaling_m * scaling_global
            p_j = (
                p_end - my_vec_max[i + 1] * z_vec * scaling_m * scaling_global
            )
            x1_c.extend((p_i[0], p_j[0], None))
            y1_c.extend((p_i[1], p_j[1], None))
            z1_c.extend((p_i[2], p_j[2], None))
            customdata_c.extend((my_vec_max[i], my_vec_max[i + 1], None))
            colors1_c.extend(["green"] * 3)

    dt_a = [
        {
            "type": "scatter3d",
            "mode": "lines",
            "x": x1_a,
            "y": y1_a,
            "z": z1_a,
            "visible": False,
            "customdata": customdata_a,
            "hovertemplate": " %{customdata:.0f}<br>" "<extra></extra>",
            "line": {"width": 3, "color": colors1_a},
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
            "hovertemplate": " %{customdata:.0f}<br>" "<extra></extra>",
            "line": {"width": 3, "color": colors1_b},
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
            "hovertemplate": " %{customdata:.0f}<br>" "<extra></extra>",
            "line": {"width": 3, "color": colors1_c},
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
            "hovertemplate": " %{customdata:.0f}<br>" "<extra></extra>",
            "line": {"width": 3, "color": colors1_d},
        }
    ]

    fig_datastructure = dict(
        data=data_dict + dt_a + dt_b + dt_c + dt_d, layout=layout
    )
    fig = go.Figure(fig_datastructure)
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="None",
                        method="update",
                        args=[
                            {"visible": [True] * len(data_dict) + [False] * 4}
                        ],
                    ),
                    dict(
                        label="Axial",
                        method="update",
                        args=[
                            {
                                "visible": [True] * len(data_dict)
                                + [True]
                                + [False] * 3
                            }
                        ],
                    ),
                    dict(
                        label="Shear",
                        method="update",
                        args=[
                            {
                                "visible": [True] * len(data_dict)
                                + [False]
                                + [True]
                                + [False] * 2
                            }
                        ],
                    ),
                    dict(
                        label="Moment",
                        method="update",
                        args=[
                            {
                                "visible": [True] * len(data_dict)
                                + [False] * 2
                                + [True]
                                + [False]
                            }
                        ],
                    ),
                    dict(
                        label="Torsion",
                        method="update",
                        args=[
                            {
                                "visible": [True] * len(data_dict)
                                + [False] * 3
                                + [True]
                            }
                        ],
                    ),
                ],
            )
        ]
    )

    if "pytest" not in sys.modules:
        if to_html_file:
            fig.write_html(to_html_file)
        else:
            fig.show()

    metadata = {
        "scaling_n": scaling_n,
        "scaling_q": scaling_q,
        "scaling_m": scaling_m,
        "scaling_t": scaling_t,
    }
    return metadata
