"""
Objects that generate component assemblies for a model.
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


# pylint: disable=dangerous-default-value

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Type
from typing import Union
from typing import Optional
from typing import Callable
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from .query import ElmQuery
from .node_gen import NodeGenerator
from ..ops.uniaxial_material import UniaxialMaterial
from ..physical_material import PhysicalMaterial
from ..mesh import Mesh
from ..ops.node import Node
from ..component_assembly import ComponentAssembly
from ..ops.element import TrussBar
from ..ops.element import ElasticBeamColumn
from ..ops.element import DispBeamColumn
from ..ops.element import ZeroLength
from ..ops.element import TwoNodeLink
from ..ops.element import GeomTransf
from ..ops.element import Lobatto
from ..ops.section import ElasticSection
from ..ops.section import FiberSection
from ..preprocessing.split_component import split_component
from ..transformations import local_axes_from_points_and_angle
from ..transformations import transformation_matrix
from ..defaults import load_util_rigid_elastic
from ..gen import zerolength_gen
from .. import common


if TYPE_CHECKING:
    from ..model import Model
    from ..level import Level


nparr = npt.NDArray[np.float64]


def retrieve_snap_pt_global_offset(
        placement: str, section: ElasticSection | FiberSection,
        p_i: nparr, p_j: nparr, angle: float) -> nparr:
    """
    Returns the necessary offset to connect an element at a specified
    snap point of the section.

    Arguments:
      placement: Placement tag. Can be any of "centroid",
        "top_center", "top_left", "top_right", "center_left",
        "center_right", "bottom_center", "bottom_left", "bottom_right"
      section: Section object.
      p_i: Internal point at the i-end.
      p_j: Internal point at the j-end.

    Returns:
      The offset.

    """

    if section.snap_points and (placement != "centroid"):
        # obtain offset from section (local system)
        d_z, d_y = section.snap_points[placement]
        sec_offset_local: nparr = np.array([0.00, d_y, d_z])
        # retrieve local coordinate system
        x_axis, y_axis, z_axis = local_axes_from_points_and_angle(
            p_i, p_j, angle
        )
        t_glob_to_loc = transformation_matrix(x_axis, y_axis, z_axis)
        t_loc_to_glob = t_glob_to_loc.T
        sec_offset_global = t_loc_to_glob @ sec_offset_local
    else:
        sec_offset_global = np.zeros(3)
    return sec_offset_global


def beam_placement_lookup(
    x_coord,
    y_coord,
    query,
    ndg,
    lvls,
    key,
    user_offset,
    section_offset,
    split_existing,
    snap,
):
    """
    Performs lookup operations before placing a beam-functioning
    component assembly to determine how to connect it with
    respect to the other existing objects in the model.

    """

    lvl = lvls[key]
    node = query.search_node_lvl(x_coord, y_coord, lvl.uid)
    pinit = np.array((x_coord, y_coord, lvl.elevation)) + user_offset
    e_o = user_offset.copy() + section_offset
    if not node:
        if split_existing:
            node, offset = split_component(split_existing, pinit)
            e_o += offset
        else:
            node = ndg.add_node_lvl(x_coord, y_coord, key)
    else:
        # first check if a panel zone or other type of joint-like
        # component assembly exists at that node
        result_node = None
        components = query.retrieve_components_from_nodes([node], lvl.uid)
        for component in components.values():
            if component.component_purpose == "steel_W_panel_zone":
                if snap in [
                    "middle_front",
                    "middle_back",
                    "top_node",
                    "bottom_node",
                ]:
                    result_node = component.external_nodes.named_contents[snap]
                    e_o += np.array(
                        (0.00, 0.00, node.coords[2] - result_node.coords[2])
                    )
                    node = result_node
                    return node, e_o
                if snap in [
                    "centroid",
                    "top_center",
                    "top_left",
                    "top_right",
                    "center_left",
                    "center_right",
                    "bottom_center",
                    "bottom_left",
                    "bottom_right",
                ]:
                    elm = component.elements.named_contents[
                        "elm_interior"
                    ]
                    d_z, d_y = elm.section.snap_points[snap]
                    sec_offset_local: nparr = -np.array([0.00, d_y, d_z])
                    # retrieve local coordinate system
                    x_axis = elm.geomtransf.x_axis
                    y_axis = elm.geomtransf.y_axis
                    z_axis = elm.geomtransf.z_axis
                    t_glob_to_loc = transformation_matrix(
                        x_axis, y_axis, z_axis
                    )
                    t_loc_to_glob = t_glob_to_loc.T
                    sec_offset_global = t_loc_to_glob @ sec_offset_local
                    result_node = node
                    e_o += sec_offset_global
                    return node, e_o
                raise ValueError(f"Unsupported snap keyword: {snap}")

        # else check if a column-like component assembly exists
        if key - 1 in lvls:
            node_below = query.search_node_lvl(x_coord, y_coord, key - 1)
            if node_below:
                column = query.search_connectivity([node, node_below])
                if column:
                    elms = []
                    for dctkey in column.element_connectivity().keys():
                        if node.uid in dctkey:
                            elms.append(column.element_connectivity()[dctkey])
                    assert elms, "There should be an element here."
                    assert (
                        len(elms) == 1
                    ), "There should only be one element here."
                    elm = elms[0]
                    # obtain offset from section (local system)
                    if hasattr(elm, 'section'):
                        if elm.section.snap_points:
                            d_z, d_y = elm.section.snap_points[snap]
                            sec_offset_local = -np.array([0.00, d_y, d_z])
                            # retrieve local coordinate system
                            x_axis = elm.geomtransf.x_axis
                            y_axis = elm.geomtransf.y_axis
                            z_axis = elm.geomtransf.z_axis
                            t_glob_to_loc = transformation_matrix(
                                x_axis, y_axis, z_axis
                            )
                            t_loc_to_glob = t_glob_to_loc.T
                            sec_offset_global = (
                                t_loc_to_glob @ sec_offset_local)
                            e_o += sec_offset_global
    return node, e_o


def look_for_panel_zone(node: Node, lvl: Level, query: ElmQuery) -> Node:
    """
    Determines if a panel zone joint component assembly is present
    at the specified node.

    """

    components = query.retrieve_components_from_nodes([node], lvl.uid)
    result_node = node
    for component in components.values():
        if component.component_purpose == "steel_W_panel_zone":
            result_node = component.external_nodes.named_contents[
                "bottom_node"
            ]
            break
    return result_node


@dataclass(repr=False)
class TrussBarGenerator:
    """
    Introduces bar elements to a model.
    Bar elements are linear elements that can only carry axial load.

    """

    model: Model = field(repr=False)

    def add(
            self,
            xi_coord: float,
            yi_coord: float,
            lvl_key_i: int,
            offset_i: nparr,
            snap_i: str,
            xj_coord: float,
            yj_coord: float,
            lvl_key_j: int,
            offset_j: nparr,
            snap_j: str,
            transf_type: str,
            area: float,
            mat: UniaxialMaterial,
            outside_shape: Mesh,
            weight_per_length: float = 0.00,
            split_existing_i: bool = None,
            split_existing_j: bool = None,
            component_purpose: str = 'Truss Element'
    ) -> ComponentAssembly:
        """
        Adds a truss bar element.
        If offsets are required, they are implemented through the
        addition of RigidLink elements.

        """

        query = ElmQuery(self.model)
        ndg = NodeGenerator(self.model)
        lvls = self.model.levels

        lvl_i = lvls[lvl_key_i]

        sec_offset_global = np.zeros(3)

        node_i, eo_i = beam_placement_lookup(
            xi_coord,
            yi_coord,
            query,
            ndg,
            lvls,
            lvl_key_i,
            offset_i,
            sec_offset_global,
            split_existing_i,
            snap_i,
        )
        node_j, eo_j = beam_placement_lookup(
            xj_coord,
            yj_coord,
            query,
            ndg,
            lvls,
            lvl_key_j,
            offset_j,
            sec_offset_global,
            split_existing_j,
            snap_j,
        )

        # for braces, even if we specify a snap value that results in
        # the brace being connected to a different node (this is
        # typically done at panel zones), we still want the ends of
        # the brace to be located at the coordinates we specify, which
        # is different to what is done with the connectivity of beams.
        # Therefore, if the coordinates of the returned nodes differ
        # with those that we specified, we add the difference in the
        # offset, to move the ends of the brace back where we want
        # them to be. The effect of this is that the rigid offsets
        # (twonodelinks) will connect to that other node.
        i_diff = (np.array((xi_coord, yi_coord))
                  - np.array(node_i.coords[0:2]))
        if np.linalg.norm(i_diff) > common.EPSILON:
            eo_i[0:2] += i_diff
        j_diff = (np.array((xj_coord, yj_coord))
                  - np.array(node_j.coords[0:2]))
        if np.linalg.norm(j_diff) > common.EPSILON:
            eo_j[0:2] += j_diff

        # instantiate a component assembly
        component = ComponentAssembly(
            uid=self.model.uid_generator.new("component"),
            parent_collection=lvl_i.components,
            component_purpose=component_purpose,
        )
        # add it to the level
        lvl_i.components.add(component)
        # fill component assembly
        component.external_nodes.add(node_i)
        component.external_nodes.add(node_j)

        def prepare_connection(
                node_x: Node, eo_x: nparr) \
                -> Node:
            """
            For each end of the bar element, creates a rigid link if
            an offset exists, and returns the node to which the bar
            element should connect to. This function is called twice,
            once for the i-end and once for the j-end.  For purposes
            of clarity, the index x will be used here, assuming that
            it will be substituted with i and j.
            """

            # if there is an offset at the x-end, create an internal node
            # and add a rigidlink element to the component assembly
            if np.linalg.norm(eo_x) > common.EPSILON:
                int_node_x = Node(
                    self.model.uid_generator.new('node'),
                    [*(np.array(node_x.coords) + eo_x)]
                )
                component.internal_nodes.add(int_node_x)
                n_x = int_node_x
                dirs, mats = zerolength_gen.fix_all(self.model)
                # flip the nodes if the element is about to be defined
                # upside down
                if np.allclose(
                        np.array(node_x.coords[0:2]),
                        np.array(int_node_x.coords[0:2]),
                ) and int_node_x.coords[2] > node_x.coords[2]:
                    x_axis, y_axis, _ = local_axes_from_points_and_angle(
                        np.array(int_node_x.coords),
                        np.array(node_x.coords),
                        0.00)
                    elm_link = TwoNodeLink(
                        component,
                        self.model.uid_generator.new("element"),
                        [node_x, int_node_x],
                        mats,
                        dirs,
                        x_axis,
                        y_axis,
                    )
                else:
                    x_axis, y_axis, _ = local_axes_from_points_and_angle(
                        np.array(node_x.coords),
                        np.array(int_node_x.coords),
                        0.00)
                    elm_link = TwoNodeLink(
                        component,
                        self.model.uid_generator.new("element"),
                        [node_x, int_node_x],
                        mats,
                        dirs,
                        x_axis,
                        y_axis,
                    )
                component.elements.add(elm_link)
            else:
                n_x = node_x
            return n_x

        # call the function here for the i and the j ends.
        n_i = prepare_connection(node_i, eo_i)
        n_j = prepare_connection(node_j, eo_j)

        # create the element
        elm_truss = TrussBar(
            parent_component=component,
            uid=self.model.uid_generator.new("element"),
            nodes=[n_i, n_j],
            transf_type=transf_type,
            area=area,
            mat=mat,
            outside_shape=outside_shape,
            weight_per_length=weight_per_length
        )

        # add it to the component assembly
        component.elements.add(elm_truss)

        return component


@dataclass(repr=False)
class BeamColumnGenerator:
    """
    Introduces beamcolumn elements to a model.

    """

    model: Model = field(repr=False)

    def define_beamcolumn(
            self,
            assembly: ComponentAssembly,
            node_i: Node,
            node_j: Node,
            offset_i: nparr,
            offset_j: nparr,
            transf_type: str,
            section: ElasticSection | FiberSection,
            element_type: Type[Union[ElasticBeamColumn, DispBeamColumn]],
            angle: float = 0.00,
    ) -> ElasticBeamColumn | DispBeamColumn:
        """
        Adds a beamcolumn element to the model, connecting the
        specified nodes.

        """

        p_i = np.array(node_i.coords) + offset_i
        p_j = np.array(node_j.coords) + offset_j
        axes = local_axes_from_points_and_angle(
            p_i, p_j, angle
        )  # type: ignore
        if element_type.__name__ == "ElasticBeamColumn":
            assert isinstance(section, ElasticSection)
            transf = GeomTransf(
                transf_type,
                self.model.uid_generator.new("transformation"),
                offset_i,
                offset_j,
                *axes,
            )
            elm_el = ElasticBeamColumn(
                parent_component=assembly,
                uid=self.model.uid_generator.new("element"),
                nodes=[node_i, node_j],
                section=section,
                geomtransf=transf,
            )
            res: Union[ElasticBeamColumn, DispBeamColumn] = elm_el
        elif element_type.__name__ == "DispBeamColumn":
            assert isinstance(section, FiberSection)
            # TODO: add elastic section support
            transf = GeomTransf(
                transf_type,
                self.model.uid_generator.new("transformation"),
                offset_i,
                offset_j,
                *axes,
            )
            beam_integration = Lobatto(
                uid=self.model.uid_generator.new("beam integration"),
                parent_section=section,
                n_p=2,
            )
            elm_disp = DispBeamColumn(
                parent_component=assembly,
                uid=self.model.uid_generator.new("element"),
                nodes=[node_i, node_j],
                section=section,
                geomtransf=transf,
                integration=beam_integration,
            )
            res = elm_disp
        else:
            raise ValueError(
                'Invalid element type: {element_type.__name__}')
        return res

    def define_zerolength(
        self,
        assembly: ComponentAssembly,
        node_i: Node,
        node_j: Node,
        x_axis: nparr,
        y_axis: nparr,
        zerolength_gen: Callable,  # type: ignore
        zerolength_gen_args: dict[str, object],
    ) -> ZeroLength:
        """
        Defines a zerolength element.

        """

        dirs, mats = zerolength_gen(model=self.model, **zerolength_gen_args)
        elm = ZeroLength(
            assembly,
            self.model.uid_generator.new("element"),
            [node_i, node_j],
            mats,
            dirs,
            x_axis,
            y_axis,
        )
        return elm

    def define_two_node_link(
        self,
        assembly: ComponentAssembly,
        node_i: Node,
        node_j: Node,
        x_axis: nparr,
        y_axis: nparr,
        zerolength_gen: Callable,  # type: ignore
        zerolength_gen_args: dict[str, object],
    ) -> TwoNodeLink:
        """
        Defines a TwoNodeLink element.

        """

        dirs, mats = zerolength_gen(model=self.model, **zerolength_gen_args)
        elm = TwoNodeLink(
            assembly,
            self.model.uid_generator.new("element"),
            [node_i, node_j],
            mats,
            dirs,
            x_axis,
            y_axis,
        )
        return elm

    def add_beamcolumn_elements_in_series(
        self,
        component,
        node_i,
        node_j,
        eo_i,
        eo_j,
        n_sub,
        transf_type,
        section,
        element_type,
        angle,
        camber_2,
        camber_3,
    ):
        """
        Adds beamcolumn elemens in series.

        """

        if n_sub > 1:
            p_i = np.array(node_i.coords) + eo_i
            p_j = np.array(node_j.coords) + eo_j
            clear_len = np.linalg.norm(p_j - p_i)
            internal_pt_coords = np.linspace(
                tuple(p_i), tuple(p_j), num=n_sub + 1
            )

            # initial deformation
            t_vals = np.linspace(0.00, 1.00, num=n_sub + 1)
            # quadratic initial imperfection
            # offset_vals = 4.00 * (-t_vals**2 + t_vals)
            # sinusoidal initial imperfection
            offset_vals = np.sin(np.pi * t_vals)
            offset_2 = offset_vals * camber_2 * clear_len
            offset_3 = offset_vals * camber_3 * clear_len
            camber_offset: nparr = np.column_stack(
                (np.zeros(n_sub + 1), offset_2, offset_3)
            )
            x_axis, y_axis, z_axis = local_axes_from_points_and_angle(
                p_i, p_j, angle
            )
            t_glob_to_loc = transformation_matrix(x_axis, y_axis, z_axis)
            t_loc_to_glob = t_glob_to_loc.T
            camber_offset_global = (t_loc_to_glob @ camber_offset.T).T
            internal_pt_coords += camber_offset_global

            intnodes = []
            for i in range(1, len(internal_pt_coords) - 1):
                intnode = Node(
                    self.model.uid_generator.new("node"),
                    [*internal_pt_coords[i]],
                )
                component.internal_nodes.add(intnode)
                intnodes.append(intnode)
        for i in range(n_sub):
            if i == 0:
                n_i = node_i
                o_i = eo_i
            else:
                n_i = intnodes[i - 1]
                o_i = np.zeros(3)
            if i == n_sub - 1:
                n_j = node_j
                o_j = eo_j
            else:
                n_j = intnodes[i]
                o_j = np.zeros(3)
            element = self.define_beamcolumn(
                assembly=component,
                node_i=n_i,
                node_j=n_j,
                offset_i=o_i,
                offset_j=o_j,
                transf_type=transf_type,
                section=section,
                element_type=element_type,
                angle=angle,
            )
            if element_type.__name__ not in {
                    'ElasticBeamColumn', 'DispBeamColumn'}:
                raise TypeError(
                    "Unsupported element type:" f" {element_type.__name__}"
                )
            component.elements.add(element)

    def generate_plain_component_assembly(
        self,
        component_purpose,
        lvl,
        node_i,
        node_j,
        n_sub,
        eo_i,
        eo_j,
        section,
        element_type,
        transf_type,
        angle,
        camber_2,
        camber_3,
    ):
        """
        Generates a plain component assembly, with line elements in
        series.

        """

        assert isinstance(node_i, Node)
        assert isinstance(node_j, Node)

        uids = [node.uid for node in (node_i, node_j)]
        uids.sort()
        uids_tuple = (*uids,)
        assert uids_tuple not in self.model.component_connectivity()

        # instantiate a component assembly
        component = ComponentAssembly(
            uid=self.model.uid_generator.new("component"),
            parent_collection=lvl.components,
            component_purpose=component_purpose,
        )
        # add it to the level
        lvl.components.add(component)
        # fill component assembly
        component.external_nodes.add(node_i)
        component.external_nodes.add(node_j)

        self.add_beamcolumn_elements_in_series(
            component,
            node_i,
            node_j,
            eo_i,
            eo_j,
            n_sub,
            transf_type,
            section,
            element_type,
            angle,
            camber_2,
            camber_3,
        )

        return component

    def generate_hinged_component_assembly(
        self,
        component_purpose,
        lvl,
        node_i,
        node_j,
        n_sub,
        eo_i,
        eo_j,
        section,
        element_type,
        transf_type,
        angle,
        camber_2,
        camber_3,
        zerolength_gen_i,
        zerolength_gen_args_i,
        zerolength_gen_j,
        zerolength_gen_args_j,
    ):
        """
        Defines a component assembly that is comprised of
        beamcolumn elements connected in series with nonlinear springs
        attached at the ends, followed by another sequence of
        beamcolumn elements (in order to be able to specify rigid offsets).

        """

        uids = [node.uid for node in (node_i, node_j)]
        uids.sort()
        uids_tuple = (*uids,)
        assert uids_tuple not in self.model.component_connectivity()

        # instantiate a component assembly
        component = ComponentAssembly(
            uid=self.model.uid_generator.new("component"),
            parent_collection=lvl.components,
            component_purpose=component_purpose,
        )
        # fill component assembly
        component.external_nodes.add(node_i)
        component.external_nodes.add(node_j)
        # add it to the level
        lvl.components.add(component)

        p_i = np.array(node_i.coords) + eo_i
        p_j = np.array(node_j.coords) + eo_j
        axes = local_axes_from_points_and_angle(p_i, p_j, angle)
        x_axis, y_axis, _ = axes
        clear_length = np.linalg.norm(p_j - p_i)
        zerolength_gen_args_i.update({"element_length": clear_length})
        zerolength_gen_args_j.update({"element_length": clear_length})

        # we can have hinges at both ends, or just one of the two ends.
        # ...or even no hinges!
        if zerolength_gen_i:

            hinge_location_i = p_i + x_axis * zerolength_gen_args_i["distance"]
            nh_i_out = Node(
                self.model.uid_generator.new("node"), [*hinge_location_i]
            )
            nh_i_in = Node(
                self.model.uid_generator.new("node"), [*hinge_location_i]
            )
            nh_i_in.visibility.connected_to_zerolength = True
            component.internal_nodes.add(nh_i_out)
            component.internal_nodes.add(nh_i_in)
            if "element_type" in zerolength_gen_args_i:
                element_type_i = zerolength_gen_args_i["element_type"]
            else:
                element_type_i = element_type
            if element_type_i.__name__ in (
                "ElasticBeamColumn",
                "DispBeamColumn",
            ):
                if "section" in zerolength_gen_args_i:
                    section_i = zerolength_gen_args_i["section"]
                else:
                    section_i = section
                if "transf_type" in zerolength_gen_args_i:
                    transf_type_i = zerolength_gen_args_i["transf_type"]
                else:
                    transf_type_i = transf_type
                self.add_beamcolumn_elements_in_series(
                    component,
                    node_i,
                    nh_i_out,
                    eo_i,
                    np.zeros(3),
                    zerolength_gen_args_i["n_sub"],
                    transf_type_i,
                    section_i,
                    element_type_i,
                    angle,
                    0.00,
                    0.00,
                )
            elif element_type_i.__name__ == "TwoNodeLink":
                elm = self.define_two_node_link(
                    component, node_i, nh_i_out,
                    x_axis, y_axis,
                    zerolength_gen.fix_all, {}
                )
                component.elements.add(elm)
            else:
                raise ValueError(f"Invalid element_type_i: {element_type_i}")
            zerolen_elm = self.define_zerolength(
                component,
                nh_i_out,
                nh_i_in,
                x_axis,
                y_axis,
                zerolength_gen_i,
                zerolength_gen_args_i,
            )
            component.elements.add(zerolen_elm)
            conn_node_i = nh_i_in
            conn_eo_i = np.zeros(3)
        else:
            conn_node_i = node_i
            conn_eo_i = eo_i
        if zerolength_gen_j:

            hinge_location_j = p_i + x_axis * (
                clear_length - zerolength_gen_args_j["distance"]
            )
            nh_j_out = Node(
                self.model.uid_generator.new("node"), [*hinge_location_j]
            )
            nh_j_in = Node(
                self.model.uid_generator.new("node"), [*hinge_location_j]
            )
            nh_j_in.visibility.connected_to_zerolength = True
            component.internal_nodes.add(nh_j_out)
            component.internal_nodes.add(nh_j_in)
            if "element_type" in zerolength_gen_args_j:
                element_type_j = zerolength_gen_args_j["element_type"]
            else:
                element_type_j = element_type
            if element_type_j.__name__ in (
                "ElasticBeamColumn",
                "DispBeamColumn",
            ):
                if "section" in zerolength_gen_args_j:
                    section_j = zerolength_gen_args_j["section"]
                else:
                    section_j = section
                if "transf_type" in zerolength_gen_args_j:
                    transf_type_j = zerolength_gen_args_j["transf_type"]
                else:
                    transf_type_j = transf_type
                self.add_beamcolumn_elements_in_series(
                    component,
                    nh_j_out,
                    node_j,
                    np.zeros(3),
                    eo_j,
                    zerolength_gen_args_j["n_sub"],
                    transf_type_j,
                    section_j,
                    element_type_j,
                    angle,
                    0.00,
                    0.00,
                )
            elif element_type_j.__name__ == "TwoNodeLink":
                elm = self.define_two_node_link(
                    component, nh_j_out, node_j,
                    x_axis, y_axis,
                    zerolength_gen.fix_all, {}
                )
                component.elements.add(elm)
            else:
                raise ValueError(f"Invalid element_type_j: {element_type_j}")
            zerolen_elm = self.define_zerolength(
                component,
                nh_j_out,
                nh_j_in,
                -x_axis,
                y_axis,
                zerolength_gen_j,
                zerolength_gen_args_j,
            )
            component.elements.add(zerolen_elm)
            conn_node_j = nh_j_in
            conn_eo_j = np.zeros(3)
        else:
            conn_node_j = node_j
            conn_eo_j = eo_j

        self.add_beamcolumn_elements_in_series(
            component,
            conn_node_i,
            conn_node_j,
            conn_eo_i,
            conn_eo_j,
            n_sub,
            transf_type,
            section,
            element_type,
            angle,
            camber_2,
            camber_3,
        )
        return component

    def add_vertical_active(
        self,
        x_coord: float,
        y_coord: float,
        offset_i: nparr,
        offset_j: nparr,
        transf_type: str,
        n_sub: int,
        section: ElasticSection | FiberSection,
        element_type: Type[Union[ElasticBeamColumn, DispBeamColumn]],
        placement: str = "centroid",
        angle: float = 0.00,
        camber_2: float = 0.00,
        camber_3: float = 0.00,
        method: str = "generate_plain_component_assembly",
        additional_args: dict[str, object] = {},
    ) -> dict[int, ComponentAssembly]:
        """
        Adds a vertical beamcolumn element to all active levels.  This
        method assumes that the levels are defined in order, from
        lowest to highest elevation, with consecutive ascending
        integer keys.

        """

        ndg = NodeGenerator(self.model)
        query = ElmQuery(self.model)
        lvls = self.model.levels
        assert lvls.active, "No active levels."
        defined_component_assemblies: dict[int, ComponentAssembly] = {}
        for key in lvls.active:
            lvl = lvls[key]
            if key - 1 not in lvls:
                continue

            top_node = query.search_node_lvl(x_coord, y_coord, key)
            if not top_node:
                top_node = ndg.add_node_lvl(x_coord, y_coord, key)

            bottom_node = query.search_node_lvl(x_coord, y_coord, key - 1)
            if not bottom_node:
                bottom_node = ndg.add_node_lvl(x_coord, y_coord, key - 1)

            # check for a panel zone
            top_node = look_for_panel_zone(top_node, lvl, query)

            p_i = np.array(top_node.coords) + offset_i
            p_j = np.array(bottom_node.coords) + offset_j
            sec_offset_global = retrieve_snap_pt_global_offset(
                placement, section, p_i, p_j, angle
            )
            p_i += sec_offset_global
            p_j += sec_offset_global
            eo_i = offset_i + sec_offset_global
            eo_j = offset_j + sec_offset_global

            args = {
                "component_purpose": "vertical_component",
                "lvl": lvl,
                "node_i": top_node,
                "node_j": bottom_node,
                "n_sub": n_sub,
                "eo_i": eo_i,
                "eo_j": eo_j,
                "section": section,
                "element_type": element_type,
                "transf_type": transf_type,
                "angle": angle,
                "camber_2": camber_2,
                "camber_3": camber_3,
            }

            args.update(additional_args)
            assert hasattr(self, method), f"Method not available: {method}"
            mthd = getattr(self, method)
            defined_component_assemblies[key] = mthd(**args)
        return defined_component_assemblies

    def add_horizontal_active(
        self,
        xi_coord: float,
        yi_coord: float,
        xj_coord: float,
        yj_coord: float,
        offset_i: nparr,
        offset_j: nparr,
        snap_i: str,
        snap_j: str,
        transf_type: str,
        n_sub: int,
        section: ElasticSection,
        element_type: Type[Union[ElasticBeamColumn, DispBeamColumn]],
        placement: str = "centroid",
        angle: float = 0.00,
        camber_2: float = 0.00,
        camber_3: float = 0.00,
        split_existing_i: Optional[ComponentAssembly] = None,
        split_existing_j: Optional[ComponentAssembly] = None,
        h_offset_i: float = 0.00,
        h_offset_j: float = 0.00,
        method: str = "generate_plain_component_assembly",
        additional_args: dict[str, object] = {},
    ) -> dict[int, ComponentAssembly]:
        """
        Adds a diagonal beamcolumn element to all active levels.

        """

        query = ElmQuery(self.model)
        ndg = NodeGenerator(self.model)
        lvls = self.model.levels
        assert lvls.active, "No active levels."
        defined_component_assemblies: dict[int, ComponentAssembly] = {}
        for key in lvls.active:
            lvl = lvls[key]
            lvl_prev = lvls.get(key - 1)

            if not lvl_prev:
                continue

            p_i_init = np.array((xi_coord, yi_coord, lvl.elevation)) + offset_i
            p_j_init = np.array((xj_coord, yj_coord, lvl.elevation)) + offset_j

            # retrieve local coordinate system
            x_axis, y_axis, z_axis = local_axes_from_points_and_angle(
                p_i_init, p_j_init, angle
            )  # type: ignore

            p_i_init += h_offset_i * x_axis
            p_j_init += - h_offset_j * x_axis
            offset_i += h_offset_i * x_axis
            offset_j += - h_offset_j * x_axis

            if section.snap_points and (placement != "centroid"):
                # obtain offset from section (local system)
                d_z, d_y = section.snap_points[placement]
                sec_offset_local: nparr = np.array([0.00, d_y, d_z])
                t_glob_to_loc = transformation_matrix(x_axis, y_axis, z_axis)
                t_loc_to_glob = t_glob_to_loc.T
                sec_offset_global = t_loc_to_glob @ sec_offset_local
            else:
                sec_offset_global = np.zeros(3)

            node_i, eo_i = beam_placement_lookup(
                xi_coord,
                yi_coord,
                query,
                ndg,
                lvls,
                key,
                offset_i,
                sec_offset_global,
                split_existing_i,
                snap_i,
            )
            node_j, eo_j = beam_placement_lookup(
                xj_coord,
                yj_coord,
                query,
                ndg,
                lvls,
                key,
                offset_j,
                sec_offset_global,
                split_existing_j,
                snap_j,
            )

            args = {
                "component_purpose": "horizontal_component",
                "lvl": lvl,
                "node_i": node_i,
                "node_j": node_j,
                "n_sub": n_sub,
                "eo_i": eo_i,
                "eo_j": eo_j,
                "section": section,
                "element_type": element_type,
                "transf_type": transf_type,
                "angle": angle,
                "camber_2": camber_2,
                "camber_3": camber_3,
            }

            args.update(additional_args)
            assert hasattr(self, method), f"Method not available: {method}"
            mthd = getattr(self, method)
            defined_component_assemblies[key] = mthd(**args)
        return defined_component_assemblies

    def add_diagonal_active(
            self,
            xi_coord: float,
            yi_coord: float,
            xj_coord: float,
            yj_coord: float,
            offset_i: nparr,
            offset_j: nparr,
            snap_i: str,
            snap_j: str,
            transf_type: str,
            n_sub: int,
            section: ElasticSection,
            element_type: Type[Union[ElasticBeamColumn, DispBeamColumn]],
            placement: str = 'centroid',
            angle: float = 0.00,
            camber_2: float = 0.00, camber_3: float = 0.00,
            split_existing_i: Optional[ComponentAssembly] = None,
            split_existing_j: Optional[ComponentAssembly] = None,
            method: str = 'generate_plain_component_assembly',
            additional_args: dict[str, object] = {}
    ) -> dict[int, ComponentAssembly]:
        """
        Adds a diagonal beamcolumn element to all active levels.

        """

        query = ElmQuery(self.model)
        ndg = NodeGenerator(self.model)
        lvls = self.model.levels
        assert lvls.active, 'No active levels.'
        defined_component_assemblies: dict[int, ComponentAssembly] = {}
        for key in lvls.active:
            lvl = lvls[key]
            lvl_prev = lvls.get(key-1)

            if not lvl_prev:
                continue

            p_i_init = np.array((xi_coord, yi_coord, lvl.elevation)) + offset_i
            p_j_init = np.array((xj_coord, yj_coord, lvl.elevation)) + offset_j

            if section.snap_points and (placement != 'centroid'):
                # obtain offset from section (local system)
                d_z, d_y = section.snap_points[placement]
                sec_offset_local: nparr = np.array([0.00, d_y, d_z])
                # retrieve local coordinate system
                x_axis, y_axis, z_axis = \
                    local_axes_from_points_and_angle(
                        p_i_init, p_j_init, angle)  # type: ignore
                t_glob_to_loc = transformation_matrix(
                    x_axis, y_axis, z_axis)
                t_loc_to_glob = t_glob_to_loc.T
                sec_offset_global = t_loc_to_glob @ sec_offset_local
            else:
                sec_offset_global = np.zeros(3)

            node_i, eo_i = beam_placement_lookup(
                xi_coord, yi_coord, query, ndg,
                lvls, key, offset_i,
                sec_offset_global,
                split_existing_i,
                snap_i)
            node_j, eo_j = beam_placement_lookup(
                xj_coord, yj_coord, query, ndg,
                lvls, key-1, offset_j,
                sec_offset_global,
                split_existing_j,
                snap_j)

            args = {
                'component_purpose': 'diagonal_component',
                'lvl': lvl,
                'node_i': node_i,
                'node_j': node_j,
                'n_sub': n_sub,
                'eo_i': eo_i,
                'eo_j': eo_j,
                'section': section,
                'element_type': element_type,
                'transf_type': transf_type,
                'angle': angle,
                'camber_2': camber_2,
                'camber_3': camber_3
            }

            args.update(additional_args)
            assert hasattr(self, method), \
                f'Method not available: {method}'
            mthd = getattr(self, method)
            defined_component_assemblies[key] = mthd(**args)
        return defined_component_assemblies

    def add_pz_active(
            self,
            x_coord: float,
            y_coord: float,
            section: ElasticSection,
            physical_material: PhysicalMaterial,
            angle: float,
            column_depth: float,
            beam_depth: float,
            zerolength_method: str,
            zerolength_args: dict[str, object],
    ) -> dict[int, ComponentAssembly]:
        """
        Adds a component assembly representing a steel W-section
        panel zone joint.

        """

        ndg = NodeGenerator(self.model)
        query = ElmQuery(self.model)
        lvls = self.model.levels
        assert lvls.active, "No active levels."
        defined_components: dict[int, ComponentAssembly] = {}
        for key in lvls.active:

            lvl = lvls[key]
            if key - 1 not in lvls:
                continue

            top_node = query.search_node_lvl(x_coord, y_coord, key)
            if not top_node:
                top_node = ndg.add_node_lvl(x_coord, y_coord, key)

            # instantiate a component assembly
            component = ComponentAssembly(
                uid=self.model.uid_generator.new("component"),
                parent_collection=lvl.components,
                component_purpose="steel_W_panel_zone",
            )
            # add it to the level
            lvl.components.add(component)

            p_i: nparr = np.array(top_node.coords)
            p_j = np.array(top_node.coords) + np.array(
                (0.00, 0.00, -beam_depth)
            )
            x_axis, y_axis, z_axis = local_axes_from_points_and_angle(
                p_i, p_j, angle
            )  # type: ignore

            # determine node locations
            top_h_f_loc = p_i + y_axis * column_depth / 2.00
            top_h_b_loc = p_i - y_axis * column_depth / 2.00
            top_v_f_loc = p_i + y_axis * column_depth / 2.00
            top_v_b_loc = p_i - y_axis * column_depth / 2.00
            mid_v_f_loc = (
                p_i + y_axis * column_depth / 2.00 + x_axis * beam_depth / 2.00
            )
            mid_v_b_loc = (
                p_i - y_axis * column_depth / 2.00 + x_axis * beam_depth / 2.00
            )
            bottom_h_f_loc = (
                p_i + y_axis * column_depth / 2.00 + x_axis * beam_depth
            )
            bottom_h_b_loc = (
                p_i - y_axis * column_depth / 2.00 + x_axis * beam_depth
            )
            bottom_v_f_loc = (
                p_i + y_axis * column_depth / 2.00 + x_axis * beam_depth
            )
            bottom_v_b_loc = (
                p_i - y_axis * column_depth / 2.00 + x_axis * beam_depth
            )

            # define nodes
            top_h_f = Node(
                self.model.uid_generator.new("node"), [*top_h_f_loc]
            )
            top_h_b = Node(
                self.model.uid_generator.new("node"), [*top_h_b_loc]
            )
            top_v_f = Node(
                self.model.uid_generator.new("node"), [*top_v_f_loc]
            )
            top_v_f.visibility.connected_to_zerolength = True
            top_v_b = Node(
                self.model.uid_generator.new("node"), [*top_v_b_loc]
            )
            top_v_b.visibility.connected_to_zerolength = True

            mid_v_f = ndg.add_node_lvl_xyz(
                mid_v_f_loc[0], mid_v_f_loc[1], mid_v_f_loc[2], lvl.uid
            )
            mid_v_b = ndg.add_node_lvl_xyz(
                mid_v_b_loc[0], mid_v_b_loc[1], mid_v_b_loc[2], lvl.uid
            )

            bottom_h_f = Node(
                self.model.uid_generator.new("node"), [*bottom_h_f_loc]
            )
            bottom_h_b = Node(
                self.model.uid_generator.new("node"), [*bottom_h_b_loc]
            )
            bottom_v_f = Node(
                self.model.uid_generator.new("node"), [*bottom_v_f_loc]
            )
            bottom_v_f.visibility.connected_to_zerolength = True
            bottom_v_b = Node(
                self.model.uid_generator.new("node"), [*bottom_v_b_loc]
            )
            bottom_v_b.visibility.connected_to_zerolength = True

            bottom_mid = ndg.add_node_lvl_xyz(p_j[0], p_j[1], p_j[2], lvl.uid)

            # define rigid beamcolumn elements
            if not self.model.elastic_sections.retrieve_by_attr(
                "name", "rigid_link_section"
            ):
                load_util_rigid_elastic(self.model)
            rigid_sec = self.model.elastic_sections.retrieve_by_attr(
                "name", "rigid_link_section"
            )
            assert rigid_sec

            elm_top_h_f = ElasticBeamColumn(
                component,
                self.model.uid_generator.new("element"),
                [top_node, top_h_f],
                rigid_sec,
                GeomTransf(
                    "Corotational",
                    self.model.uid_generator.new("transformation"),
                    np.zeros(3),
                    np.zeros(3),
                    y_axis,
                    -x_axis,
                    z_axis,
                ),
            )
            elm_top_h_f.visibility.hidden_when_extruded = True
            elm_top_h_f.visibility.hidden_basic_forces = True

            elm_top_h_b = ElasticBeamColumn(
                component,
                self.model.uid_generator.new("element"),
                [top_h_b, top_node],
                rigid_sec,
                GeomTransf(
                    "Corotational",
                    self.model.uid_generator.new("transformation"),
                    np.zeros(3),
                    np.zeros(3),
                    y_axis,
                    -x_axis,
                    z_axis,
                ),
            )
            elm_top_h_b.visibility.hidden_when_extruded = True
            elm_top_h_b.visibility.hidden_basic_forces = True

            elm_bottom_h_f = ElasticBeamColumn(
                component,
                self.model.uid_generator.new("element"),
                [bottom_mid, bottom_h_f],
                rigid_sec,
                GeomTransf(
                    "Corotational",
                    self.model.uid_generator.new("transformation"),
                    np.zeros(3),
                    np.zeros(3),
                    y_axis,
                    -x_axis,
                    z_axis,
                ),
            )
            elm_bottom_h_f.visibility.hidden_when_extruded = True
            elm_bottom_h_f.visibility.hidden_basic_forces = True

            elm_bottom_h_b = ElasticBeamColumn(
                component,
                self.model.uid_generator.new("element"),
                [bottom_h_b, bottom_mid],
                rigid_sec,
                GeomTransf(
                    "Corotational",
                    self.model.uid_generator.new("transformation"),
                    np.zeros(3),
                    np.zeros(3),
                    y_axis,
                    -x_axis,
                    z_axis,
                ),
            )
            elm_bottom_h_b.visibility.hidden_when_extruded = True
            elm_bottom_h_b.visibility.hidden_basic_forces = True

            elm_top_v_f = ElasticBeamColumn(
                component,
                self.model.uid_generator.new("element"),
                [top_v_f, mid_v_f],
                rigid_sec,
                GeomTransf(
                    "Corotational",
                    self.model.uid_generator.new("transformation"),
                    np.zeros(3),
                    np.zeros(3),
                    x_axis,
                    y_axis,
                    z_axis,
                ),
            )
            elm_top_v_f.visibility.hidden_when_extruded = True
            elm_top_v_f.visibility.hidden_basic_forces = True

            elm_top_v_b = ElasticBeamColumn(
                component,
                self.model.uid_generator.new("element"),
                [top_v_b, mid_v_b],
                rigid_sec,
                GeomTransf(
                    "Corotational",
                    self.model.uid_generator.new("transformation"),
                    np.zeros(3),
                    np.zeros(3),
                    x_axis,
                    y_axis,
                    z_axis,
                ),
            )
            elm_top_v_b.visibility.hidden_when_extruded = True
            elm_top_v_b.visibility.hidden_basic_forces = True

            elm_bottom_v_f = ElasticBeamColumn(
                component,
                self.model.uid_generator.new("element"),
                [mid_v_f, bottom_v_f],
                rigid_sec,
                GeomTransf(
                    "Corotational",
                    self.model.uid_generator.new("transformation"),
                    np.zeros(3),
                    np.zeros(3),
                    x_axis,
                    y_axis,
                    z_axis,
                ),
            )
            elm_bottom_v_f.visibility.hidden_when_extruded = True
            elm_bottom_v_f.visibility.hidden_basic_forces = True

            elm_bottom_v_b = ElasticBeamColumn(
                component,
                self.model.uid_generator.new("element"),
                [mid_v_b, bottom_v_b],
                rigid_sec,
                GeomTransf(
                    "Corotational",
                    self.model.uid_generator.new("transformation"),
                    np.zeros(3),
                    np.zeros(3),
                    x_axis,
                    y_axis,
                    z_axis,
                ),
            )
            elm_bottom_v_b.visibility.hidden_when_extruded = True
            elm_bottom_v_b.visibility.hidden_basic_forces = True

            elm_interior = ElasticBeamColumn(
                component,
                self.model.uid_generator.new("element"),
                [top_node, bottom_mid],
                section,
                GeomTransf(
                    "Corotational",
                    self.model.uid_generator.new("transformation"),
                    np.zeros(3),
                    np.zeros(3),
                    x_axis,
                    y_axis,
                    z_axis,
                ),
            )
            elm_interior.visibility.skip_opensees_definition = True
            elm_interior.visibility.hidden_at_line_plots = True

            assert hasattr(
                zerolength_gen, zerolength_method), \
                f"Method not available: {zerolength_method}"
            mthd = getattr(zerolength_gen, zerolength_method)

            # define zerolength elements
            zerolength_gen_args = {
                    "section": section,
                    "physical_material": physical_material,
                    "pz_length": beam_depth
                 }
            zerolength_gen_args.update(zerolength_args)
            zerolen_top_f = self.define_zerolength(
                component,
                top_h_f,
                top_v_f,
                x_axis,
                y_axis,
                mthd,
                zerolength_gen_args
            )
            zerolen_top_b = self.define_zerolength(
                component,
                top_h_b,
                top_v_b,
                x_axis,
                y_axis,
                zerolength_gen.release_6,
                {}
            )
            zerolen_bottom_f = self.define_zerolength(
                component,
                bottom_h_f,
                bottom_v_f,
                x_axis,
                y_axis,
                zerolength_gen.release_6,
                {}
            )
            zerolen_bottom_b = self.define_zerolength(
                component,
                bottom_h_b,
                bottom_v_b,
                x_axis,
                y_axis,
                zerolength_gen.release_6,
                {}
            )

            # fill component assembly
            component.external_nodes.add(top_node)
            component.external_nodes.named_contents["top_node"] = top_node
            component.external_nodes.add(bottom_mid)
            component.external_nodes.named_contents["bottom_node"] = bottom_mid
            component.external_nodes.add(mid_v_f)
            component.external_nodes.named_contents["middle_front"] = mid_v_f
            component.external_nodes.add(mid_v_b)
            component.external_nodes.named_contents["middle_back"] = mid_v_b

            component.internal_nodes.add(top_h_f)
            component.internal_nodes.add(top_h_b)
            component.internal_nodes.add(top_v_f)
            component.internal_nodes.add(top_v_b)
            component.internal_nodes.add(bottom_h_f)
            component.internal_nodes.add(bottom_h_b)
            component.internal_nodes.add(bottom_v_f)
            component.internal_nodes.add(bottom_v_b)

            component.elements.add(elm_top_h_f)
            (
                component.elements.named_contents[
                    "elm_top_h_f"
                ]
            ) = elm_top_h_f
            component.elements.add(elm_top_h_b)
            (
                component.elements.named_contents[
                    "elm_top_h_b"
                ]
            ) = elm_top_h_b
            component.elements.add(elm_bottom_h_f)
            component.elements.add(elm_bottom_h_b)
            component.elements.add(elm_top_v_f)
            component.elements.add(elm_top_v_b)
            component.elements.add(elm_bottom_v_f)
            component.elements.add(elm_bottom_v_b)
            component.elements.add(elm_interior)
            (
                component.elements.named_contents[
                    "elm_interior"
                ]
            ) = elm_interior

            component.elements.add(zerolen_top_f)
            (
                component.elements.named_contents[
                    "nonlinear_spring"
                ]
            ) = zerolen_top_f  # type: ignore
            component.elements.add(zerolen_top_b)
            component.elements.add(zerolen_bottom_f)
            component.elements.add(zerolen_bottom_b)
            defined_components[key] = component

        return defined_components
