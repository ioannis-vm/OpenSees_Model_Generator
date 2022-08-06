"""
Model Generator for OpenSees ~ plain beamcolumn element generator
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator


# pylint: disable=dangerous-default-value

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Type
from typing import Union
from typing import Callable
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from ..ops.node import Node
from ..component_assembly import ComponentAssembly
from .querry import ElmQuerry
from .node_gen import NodeGenerator
from ..ops.element import ElasticBeamColumn
from ..ops.element import DispBeamColumn
from ..ops.element import ZeroLength
from ..ops.element import GeomTransf
from ..ops.element import Lobatto
from ..ops.section import ElasticSection
from ..ops.section import FiberSection
from ..preprocessing.split_component import split_component
from ..transformations import local_axes_from_points_and_angle
from ..transformations import transformation_matrix
from ..defaults import load_util_rigid_elastic
from ..gen.zerolength_gen import steel_w_col_pz


if TYPE_CHECKING:
    from ..model import Model
    from ..level import Level


nparr = npt.NDArray[np.float64]


def retrieve_snap_pt_global_offset(placement, section, p_i, p_j, angle):
    """
    Returns the necessary offset to connect an element at a specified
    snap point of the section
    """
    if section.snap_points and (placement != 'centroid'):
        # obtain offset from section (local system)
        d_z, d_y = section.snap_points[placement]
        sec_offset_local: nparr = np.array([0.00, d_y, d_z])
        # retrieve local coordinate system
        x_axis, y_axis, z_axis = \
            local_axes_from_points_and_angle(
                p_i, p_j, angle)
        t_glob_to_loc = transformation_matrix(
            x_axis, y_axis, z_axis)
        t_loc_to_glob = t_glob_to_loc.T
        sec_offset_global = t_loc_to_glob @ sec_offset_local
    else:
        sec_offset_global = np.zeros(3)
    return sec_offset_global


def beam_placement_lookup(
        x_coord, y_coord, querry, ndg, lvls, key,
        user_offset, section_offset, split_existing, snap):
    """
    Performs lookup operations before placing a beam-functioning
    component assembly to determine how to connect it with
    respect to the other existing objects in the model.
    """
    lvl = lvls[key]
    node = querry.search_node_lvl(x_coord, y_coord, lvl.uid)
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
        components = querry.retrieve_components_from_nodes([node], lvl.uid)
        for component in components.values():
            if component.component_purpose == 'steel_W_panel_zone':
                if snap in ['middle_front', 'middle_back']:
                    result_node = component.external_nodes.named_contents[snap]
                    e_o += np.array(
                        (0.00, 0.00, node.coords[2] - result_node.coords[2]))
                    node = result_node
                    return node, e_o
                if snap in [
                        'centroid', 'top_center', 'top_left',
                        'top_right', 'center_left', 'center_right',
                        'bottom_center', 'bottom_left', 'bottom_right']:
                    elm = component.elastic_beamcolumn_elements.named_contents[
                        'elm_interior']
                    d_z, d_y = elm.section.snap_points[snap]
                    sec_offset_local: nparr = -np.array([0.00, d_y, d_z])
                    # retrieve local coordinate system
                    x_axis = elm.geomtransf.x_axis
                    y_axis = elm.geomtransf.y_axis
                    z_axis = elm.geomtransf.z_axis
                    t_glob_to_loc = transformation_matrix(
                        x_axis, y_axis, z_axis)
                    t_loc_to_glob = t_glob_to_loc.T
                    sec_offset_global = (
                        t_loc_to_glob @ sec_offset_local)
                    result_node = node
                    e_o += sec_offset_global
                    return node, e_o
                raise ValueError(f'Unsupported snap keyword: {snap}')

        # else check if a column-like component assembly exists
        if key-1 in lvls:
            node_below = querry.search_node_lvl(
                x_coord, y_coord, key-1)
            if node_below:
                column = querry.search_connectivity(
                    [node, node_below])
                if column:
                    elms = []
                    for dctkey in column.element_connectivity().keys():
                        if node.uid in dctkey:
                            elms.append(
                                column
                                .element_connectivity()[dctkey])
                    assert elms, 'There should be an element here.'
                    assert len(elms) == 1, \
                        'There should only be one element here.'
                    elm = elms[0]
                    # obtain offset from section (local system)
                    if elm.section.snap_points:
                        d_z, d_y = elm.section.snap_points[snap]
                        sec_offset_local = -np.array([0.00, d_y, d_z])
                        # retrieve local coordinate system
                        x_axis = elm.geomtransf.x_axis
                        y_axis = elm.geomtransf.y_axis
                        z_axis = elm.geomtransf.z_axis
                        t_glob_to_loc = transformation_matrix(
                            x_axis, y_axis, z_axis)
                        t_loc_to_glob = t_glob_to_loc.T
                        sec_offset_global = (
                            t_loc_to_glob @ sec_offset_local)
                        e_o += sec_offset_global
        else:
            raise ValueError(
                'Error: existing node without any elements to connect to.')
    return node, e_o


def look_for_panel_zone(
        node: Node,
        lvl: Level,
        querry: ElmQuerry
) -> Node:
    """
    Determines if a panel zone joint component assembly is present
    at the specified node.
    """
    components = querry.retrieve_components_from_nodes([node], lvl.uid)
    result_node = node
    for component in components.values():
        if component.component_purpose == 'steel_W_panel_zone':
            result_node = (component.external_nodes
                           .named_contents['bottom_node'])
            break
    return result_node


@dataclass(repr=False)
class BeamColumnGenerator:
    """
    This object introduces beamcolumn elements to a model.
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
            angle=0.00) -> ElasticBeamColumn | DispBeamColumn:
        """
        Adds a beamcolumn element to the model, connecting the specified nodes
        """

        p_i = np.array(node_i.coords) + offset_i
        p_j = np.array(node_j.coords) + offset_j
        axes = local_axes_from_points_and_angle(
            p_i, p_j, angle)  # type: ignore
        if element_type.__name__ == 'ElasticBeamColumn':
            assert isinstance(section, ElasticSection)
            transf = GeomTransf(
                transf_type,
                self.model.uid_generator.new('transformation'),
                offset_i,
                offset_j,
                *axes)
            elm_el = ElasticBeamColumn(
                parent_component=assembly,
                uid=self.model.uid_generator.new('element'),
                nodes=[node_i, node_j],
                section=section,
                geomtransf=transf)
            res: Union[ElasticBeamColumn, DispBeamColumn] = elm_el
        elif element_type.__name__ == 'DispBeamColumn':
            assert isinstance(section, FiberSection)
            # TODO: add elastic section support
            transf = GeomTransf(
                transf_type,
                self.model.uid_generator.new('transformation'),
                offset_i,
                offset_j,
                *axes)
            beam_integration = Lobatto(
                uid=self.model.uid_generator.new('beam integration'),
                parent_section=section,
                n_p=5
            )
            elm_disp = DispBeamColumn(
                parent_component=assembly,
                uid=self.model.uid_generator.new('element'),
                nodes=[node_i, node_j],
                section=section,
                geomtransf=transf,
                integration=beam_integration)
            res = elm_disp
        return res

    def define_zerolength(
            self,
            assembly: ComponentAssembly,
            node_i: Node,
            node_j: Node,
            x_axis: nparr,
            y_axis: nparr,
            zerolength_gen: Callable,  # type: ignore
            zerolength_gen_args: dict[str, object]
    ):
        """
        Defines a zerolength element
        """
        dirs, mats = zerolength_gen(model=self.model, **zerolength_gen_args)
        elm = ZeroLength(
            assembly,
            self.model.uid_generator.new('element'),
            [node_i, node_j],
            mats,
            dirs,
            x_axis,
            y_axis
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
            angle
    ):
        """
        Adds beamcolumn elemens in series
        """

        if n_sub > 1:
            p_i = np.array(node_i.coords) + eo_i
            p_j = np.array(node_j.coords) + eo_j
            internal_pt_coords = np.linspace(
                tuple(p_i),
                tuple(p_j),
                num=n_sub+1)
            intnodes = []
            for i in range(1, len(internal_pt_coords)-1):
                intnode = Node(self.model.uid_generator
                               .new('node'),
                               [*internal_pt_coords[i]])
                component.internal_nodes.add(intnode)
                intnodes.append(intnode)
        for i in range(n_sub):
            if i == 0:
                n_i = node_i
                o_i = eo_i
            else:
                n_i = intnodes[i-1]
                o_i = np.zeros(3)
            if i == n_sub - 1:
                n_j = node_j
                o_j = eo_j
            else:
                n_j = intnodes[i]
                o_j = np.zeros(3)
            element = self.define_beamcolumn(
                assembly=component,
                node_i=n_i, node_j=n_j,
                offset_i=o_i, offset_j=o_j, transf_type=transf_type,
                section=section,
                element_type=element_type,
                angle=angle)
            if element_type.__name__ == 'ElasticBeamColumn':
                component.elastic_beamcolumn_elements.add(element)
            elif element_type.__name__ == 'DispBeamColumn':
                component.disp_beamcolumn_elements.add(element)
            else:
                raise TypeError(
                    'Unsupported element type:'
                    f' {element_type.__name__}')

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
            angle):
        """
        Generates a plain component assembly, with line elements in
        series
        """

        assert isinstance(node_i, Node)
        assert isinstance(node_j, Node)

        # instantiate a component assembly
        component = ComponentAssembly(
            uid=self.model.uid_generator.new('component'),
            parent_collection=lvl.components,
            component_purpose=component_purpose)
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
            angle
        )

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
            zerolength_gen_i,
            zerolength_gen_args_i,
            zerolength_gen_j,
            zerolength_gen_args_j):
        """
        Defines a component assembly that is comprised of
        beamcolumn elements connected in series with nonlinear springs
        attached at the ends, followed by another sequence of
        beamcolumn elements (in order to be able to specify rigid offsets)
        """
        # instantiate a component assembly
        component = ComponentAssembly(
            uid=self.model.uid_generator.new('component'),
            parent_collection=lvl.components,
            component_purpose=component_purpose)
        # fill component assembly
        component.external_nodes.add(node_i)
        component.external_nodes.add(node_j)
        # add it to the level
        lvl.components.add(component)

        p_i = np.array(node_i.coords) + eo_i
        p_j = np.array(node_j.coords) + eo_j
        axes = local_axes_from_points_and_angle(
            p_i, p_j, angle)
        x_axis, y_axis, _ = axes
        clear_length = np.linalg.norm(p_j - p_i)
        zerolength_gen_args_i.update({'element_length': clear_length})
        zerolength_gen_args_j.update({'element_length': clear_length})

        # we can have hinges at both ends, or just one of the two ends.
        # ...or even no hinges!
        if zerolength_gen_i:
            hinge_location_i = p_i + x_axis * zerolength_gen_args_i['distance']
            nh_i_out = Node(self.model.uid_generator.new('node'),
                            [*hinge_location_i])
            nh_i_in = Node(self.model.uid_generator.new('node'),
                           [*hinge_location_i])
            nh_i_in.visibility.connected_to_zerolength = True
            component.internal_nodes.add(nh_i_out)
            component.internal_nodes.add(nh_i_in)
            self.add_beamcolumn_elements_in_series(
                component,
                node_i,
                nh_i_out,
                eo_i,
                np.zeros(3),
                zerolength_gen_args_i['n_sub'],
                transf_type,
                section,
                element_type,
                angle
            )
            zerolen_elm = self.define_zerolength(
                component,
                nh_i_out,
                nh_i_in,
                x_axis,
                y_axis,
                zerolength_gen_i,
                zerolength_gen_args_i
            )
            component.zerolength_elements.add(zerolen_elm)
            conn_node_i = nh_i_in
            conn_eo_i = np.zeros(3)
        else:
            conn_node_i = node_i
            conn_eo_i = eo_i
        if zerolength_gen_j:
            hinge_location_j = (p_i + x_axis *
                                (clear_length
                                 - zerolength_gen_args_j['distance']))
            nh_j_out = Node(self.model.uid_generator.new('node'),
                            [*hinge_location_j])
            nh_j_in = Node(self.model.uid_generator.new('node'),
                           [*hinge_location_j])
            nh_j_in.visibility.connected_to_zerolength = True
            component.internal_nodes.add(nh_j_out)
            component.internal_nodes.add(nh_j_in)
            self.add_beamcolumn_elements_in_series(
                component,
                nh_j_out,
                node_j,
                np.zeros(3),
                eo_j,
                zerolength_gen_args_j['n_sub'],
                transf_type,
                section,
                element_type,
                angle
            )
            zerolen_elm = self.define_zerolength(
                component,
                nh_j_out,
                nh_j_in,
                -x_axis,
                y_axis,
                zerolength_gen_j,
                zerolength_gen_args_j
            )
            component.zerolength_elements.add(zerolen_elm)
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
            angle
        )

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
            placement='centroid',
            angle=0.00,
            method='generate_plain_component_assembly',
            additional_args={}
    ):
        """
        Adds a vertical beamcolumn element to all active levels.  This
        method assumes that the levels are defined in order, from
        lowest to highest elevation, with consecutive ascending
        integer keys.
        """
        ndg = NodeGenerator(self.model)
        querry = ElmQuerry(self.model)
        lvls = self.model.levels
        assert lvls.active, 'No active levels.'
        for key in lvls.active:
            lvl = lvls[key]
            if key-1 not in lvls:
                continue

            top_node = querry.search_node_lvl(x_coord, y_coord, key)
            if not top_node:
                top_node = ndg.add_node_lvl(x_coord, y_coord, key)

            bottom_node = querry.search_node_lvl(x_coord, y_coord, key-1)
            if not bottom_node:
                bottom_node = ndg.add_node_lvl(x_coord, y_coord, key-1)

            # check for a panel zone
            top_node = look_for_panel_zone(top_node, lvl, querry)

            p_i = np.array(top_node.coords) + offset_i
            p_j = np.array(bottom_node.coords) + offset_j
            sec_offset_global = retrieve_snap_pt_global_offset(
                placement, section, p_i, p_j, angle)
            p_i += sec_offset_global
            p_j += sec_offset_global
            eo_i = offset_i + sec_offset_global
            eo_j = offset_j + sec_offset_global

            args = {
                'component_purpose': 'vertical_component',
                'lvl': lvl,
                'node_i': top_node,
                'node_j': bottom_node,
                'n_sub': n_sub,
                'eo_i': eo_i,
                'eo_j': eo_j,
                'section': section,
                'element_type': element_type,
                'transf_type': transf_type,
                'angle': angle
            }

            args.update(additional_args)
            assert hasattr(self, method), \
                f'Method not available: {method}'
            mthd = getattr(self, method)
            mthd(**args)

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
            placement='centroid',
            angle=0.00,
            split_existing_i=None,
            split_existing_j=None,
            method='generate_plain_component_assembly',
            additional_args={}
    ):
        """
        Adds a horizontal beamcolumn element to all active levels.
        """
        querry = ElmQuerry(self.model)
        ndg = NodeGenerator(self.model)
        lvls = self.model.levels
        assert lvls.active, 'No active levels.'
        for key in lvls.active:
            lvl = lvls[key]

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
                xi_coord, yi_coord, querry, ndg,
                lvls, key, offset_i,
                sec_offset_global,
                split_existing_i,
                snap_i)
            node_j, eo_j = beam_placement_lookup(
                xj_coord, yj_coord, querry, ndg,
                lvls, key, offset_j,
                sec_offset_global,
                split_existing_j,
                snap_j)

            args = {
                'component_purpose': 'horizontal_component',
                'lvl': lvl,
                'node_i': node_i,
                'node_j': node_j,
                'n_sub': n_sub,
                'eo_i': eo_i,
                'eo_j': eo_j,
                'section': section,
                'element_type': element_type,
                'transf_type': transf_type,
                'angle': angle
            }

            args.update(additional_args)
            assert hasattr(self, method), \
                f'Method not available: {method}'
            mthd = getattr(self, method)
            mthd(**args)

    def add_pz_active(
            self,
            x_coord: float,
            y_coord: float,
            section: ElasticSection,
            physical_material,
            angle,
            column_depth,
            beam_depth,
            pz_doubler_plate_thickness: float,
            pz_hardening: float
    ):
        """
        Adds a component assembly representing a steel W-section
        panel zone joint.
        """
        ndg = NodeGenerator(self.model)
        querry = ElmQuerry(self.model)
        lvls = self.model.levels
        assert lvls.active, 'No active levels.'
        for key in lvls.active:

            lvl = lvls[key]
            if key-1 not in lvls:
                continue

            top_node = querry.search_node_lvl(x_coord, y_coord, key)
            if not top_node:
                top_node = ndg.add_node_lvl(x_coord, y_coord, key)

            # instantiate a component assembly
            component = ComponentAssembly(
                uid=self.model.uid_generator.new('component'),
                parent_collection=lvl.components,
                component_purpose='steel_W_panel_zone')
            # add it to the level
            lvl.components.add(component)

            p_i: nparr = np.array(top_node.coords)
            p_j = (np.array(top_node.coords)
                   + np.array((0.00, 0.00, -beam_depth)))
            x_axis, y_axis, z_axis = \
                local_axes_from_points_and_angle(
                    p_i, p_j, angle)  # type: ignore

            # determine node locations
            top_h_f_loc = p_i + y_axis * column_depth/2.00
            top_h_b_loc = p_i - y_axis * column_depth/2.00
            top_v_f_loc = p_i + y_axis * column_depth/2.00
            top_v_b_loc = p_i - y_axis * column_depth/2.00
            mid_v_f_loc = (p_i + y_axis * column_depth/2.00
                           + x_axis * beam_depth/2.00)
            mid_v_b_loc = (p_i - y_axis * column_depth/2.00
                           + x_axis * beam_depth/2.00)
            bottom_h_f_loc = (p_i + y_axis * column_depth/2.00
                              + x_axis * beam_depth)
            bottom_h_b_loc = (p_i - y_axis * column_depth/2.00
                              + x_axis * beam_depth)
            bottom_v_f_loc = (p_i + y_axis * column_depth/2.00
                              + x_axis * beam_depth)
            bottom_v_b_loc = (p_i - y_axis * column_depth/2.00
                              + x_axis * beam_depth)

            # define nodes
            top_h_f = Node(
                self.model.uid_generator.new('node'),
                [*top_h_f_loc]
            )
            top_h_b = Node(
                self.model.uid_generator.new('node'),
                [*top_h_b_loc]
            )
            top_v_f = Node(
                self.model.uid_generator.new('node'),
                [*top_v_f_loc]
            )
            top_v_f.visibility.connected_to_zerolength = True
            top_v_b = Node(
                self.model.uid_generator.new('node'),
                [*top_v_b_loc]
            )
            top_v_b.visibility.connected_to_zerolength = True

            mid_v_f = ndg.add_node_lvl_xyz(
                mid_v_f_loc[0],
                mid_v_f_loc[1],
                mid_v_f_loc[2],
                lvl.uid)
            mid_v_b = ndg.add_node_lvl_xyz(
                mid_v_b_loc[0],
                mid_v_b_loc[1],
                mid_v_b_loc[2],
                lvl.uid)

            bottom_h_f = Node(
                self.model.uid_generator.new('node'),
                [*bottom_h_f_loc]
            )
            bottom_h_b = Node(
                self.model.uid_generator.new('node'),
                [*bottom_h_b_loc]
            )
            bottom_v_f = Node(
                self.model.uid_generator.new('node'),
                [*bottom_v_f_loc]
            )
            bottom_v_f.visibility.connected_to_zerolength = True
            bottom_v_b = Node(
                self.model.uid_generator.new('node'),
                [*bottom_v_b_loc]
            )
            bottom_v_b.visibility.connected_to_zerolength = True

            bottom_mid = ndg.add_node_lvl_xyz(
                p_j[0], p_j[1], p_j[2], lvl.uid)

            # define rigid beamcolumn elements
            if not self.model.elastic_sections.retrieve_by_attr(
                    'name', 'rigid_link_section'):
                load_util_rigid_elastic(self.model)
            rigid_sec = self.model.elastic_sections.retrieve_by_attr(
                'name', 'rigid_link_section')
            assert rigid_sec

            elm_top_h_f = ElasticBeamColumn(
                component,
                self.model.uid_generator.new('element'),
                [top_node, top_h_f],
                rigid_sec,
                GeomTransf(
                    'Corotational',
                    self.model.uid_generator.new('transformation'),
                    np.zeros(3), np.zeros(3),
                    y_axis, -x_axis, z_axis
                )
            )
            elm_top_h_f.visibility.hidden_when_extruded = True

            elm_top_h_b = ElasticBeamColumn(
                component,
                self.model.uid_generator.new('element'),
                [top_h_b, top_node],
                rigid_sec,
                GeomTransf(
                    'Corotational',
                    self.model.uid_generator.new('transformation'),
                    np.zeros(3), np.zeros(3),
                    y_axis, -x_axis, z_axis
                )
            )
            elm_top_h_b.visibility.hidden_when_extruded = True

            elm_bottom_h_f = ElasticBeamColumn(
                component,
                self.model.uid_generator.new('element'),
                [bottom_mid, bottom_h_f],
                rigid_sec,
                GeomTransf(
                    'Corotational',
                    self.model.uid_generator.new('transformation'),
                    np.zeros(3), np.zeros(3),
                    y_axis, -x_axis, z_axis
                )
            )
            elm_bottom_h_f.visibility.hidden_when_extruded = True

            elm_bottom_h_b = ElasticBeamColumn(
                component,
                self.model.uid_generator.new('element'),
                [bottom_h_b, bottom_mid],
                rigid_sec,
                GeomTransf(
                    'Corotational',
                    self.model.uid_generator.new('transformation'),
                    np.zeros(3), np.zeros(3),
                    y_axis, -x_axis, z_axis
                )
            )
            elm_bottom_h_b.visibility.hidden_when_extruded = True

            elm_top_v_f = ElasticBeamColumn(
                component,
                self.model.uid_generator.new('element'),
                [top_v_f, mid_v_f],
                rigid_sec,
                GeomTransf(
                    'Corotational',
                    self.model.uid_generator.new('transformation'),
                    np.zeros(3), np.zeros(3),
                    x_axis, y_axis, z_axis
                )
            )
            elm_top_v_f.visibility.hidden_when_extruded = True

            elm_top_v_b = ElasticBeamColumn(
                component,
                self.model.uid_generator.new('element'),
                [top_v_b, mid_v_b],
                rigid_sec,
                GeomTransf(
                    'Corotational',
                    self.model.uid_generator.new('transformation'),
                    np.zeros(3), np.zeros(3),
                    x_axis, y_axis, z_axis
                )
            )
            elm_top_v_b.visibility.hidden_when_extruded = True

            elm_bottom_v_f = ElasticBeamColumn(
                component,
                self.model.uid_generator.new('element'),
                [mid_v_f, bottom_v_f],
                rigid_sec,
                GeomTransf(
                    'Corotational',
                    self.model.uid_generator.new('transformation'),
                    np.zeros(3), np.zeros(3),
                    x_axis, y_axis, z_axis
                )
            )
            elm_bottom_v_f.visibility.hidden_when_extruded = True

            elm_bottom_v_b = ElasticBeamColumn(
                component,
                self.model.uid_generator.new('element'),
                [mid_v_b, bottom_v_b],
                rigid_sec,
                GeomTransf(
                    'Corotational',
                    self.model.uid_generator.new('transformation'),
                    np.zeros(3), np.zeros(3),
                    x_axis, y_axis, z_axis
                )
            )
            elm_bottom_v_b.visibility.hidden_when_extruded = True

            elm_interior = ElasticBeamColumn(
                component,
                self.model.uid_generator.new('element'),
                [top_node, bottom_mid],
                section,
                GeomTransf(
                    'Corotational',
                    self.model.uid_generator.new('transformation'),
                    np.zeros(3), np.zeros(3),
                    x_axis, y_axis, z_axis
                )
            )
            elm_interior.visibility.skip_opensees_definition = True
            elm_interior.visibility.hidden_at_line_plots = True

            # define zerolength elements
            zerolen_top_f = self.define_zerolength(
                component,
                top_h_f,
                top_v_f,
                x_axis,
                y_axis,
                steel_w_col_pz,
                {
                    'section': section,
                    'physical_material': physical_material,
                    'pz_length': beam_depth,
                    'pz_doubler_plate_thickness': pz_doubler_plate_thickness,
                    'pz_hardening': pz_hardening
                 }
            )
            zerolen_top_b = self.define_zerolength(
                component,
                top_h_b,
                top_v_b,
                x_axis,
                y_axis,
                steel_w_col_pz,
                {
                    'section': section,
                    'physical_material': physical_material,
                    'pz_length': beam_depth,
                    'pz_doubler_plate_thickness': pz_doubler_plate_thickness,
                    'pz_hardening': pz_hardening
                 }
            )
            zerolen_bottom_f = self.define_zerolength(
                component,
                bottom_h_f,
                bottom_v_f,
                x_axis,
                y_axis,
                steel_w_col_pz,
                {
                    'section': section,
                    'physical_material': physical_material,
                    'pz_length': beam_depth,
                    'pz_doubler_plate_thickness': pz_doubler_plate_thickness,
                    'pz_hardening': pz_hardening
                 }
            )
            zerolen_bottom_b = self.define_zerolength(
                component,
                bottom_h_b,
                bottom_v_b,
                x_axis,
                y_axis,
                steel_w_col_pz,
                {
                    'section': section,
                    'physical_material': physical_material,
                    'pz_length': beam_depth,
                    'pz_doubler_plate_thickness': pz_doubler_plate_thickness,
                    'pz_hardening': pz_hardening
                 }
            )

            # fill component assembly
            component.external_nodes.add(top_node)
            component.external_nodes.named_contents['top_node'] = top_node
            component.external_nodes.add(bottom_mid)
            component.external_nodes.named_contents['bottom_node'] = bottom_mid
            component.external_nodes.add(mid_v_f)
            component.external_nodes.named_contents['middle_front'] = mid_v_f
            component.external_nodes.add(mid_v_b)
            component.external_nodes.named_contents['middle_back'] = mid_v_b

            component.internal_nodes.add(top_h_f)
            component.internal_nodes.add(top_h_b)
            component.internal_nodes.add(top_v_f)
            component.internal_nodes.add(top_v_b)
            component.internal_nodes.add(bottom_h_f)
            component.internal_nodes.add(bottom_h_b)
            component.internal_nodes.add(bottom_v_f)
            component.internal_nodes.add(bottom_v_b)

            component.elastic_beamcolumn_elements.add(elm_top_h_f)
            (component.elastic_beamcolumn_elements
             .named_contents['elm_top_h_f']) = elm_top_h_f
            component.elastic_beamcolumn_elements.add(elm_top_h_b)
            (component.elastic_beamcolumn_elements
             .named_contents['elm_top_h_b']) = elm_top_h_b
            component.elastic_beamcolumn_elements.add(elm_bottom_h_f)
            component.elastic_beamcolumn_elements.add(elm_bottom_h_b)
            component.elastic_beamcolumn_elements.add(elm_top_v_f)
            component.elastic_beamcolumn_elements.add(elm_top_v_b)
            component.elastic_beamcolumn_elements.add(elm_bottom_v_f)
            component.elastic_beamcolumn_elements.add(elm_bottom_v_b)
            component.elastic_beamcolumn_elements.add(elm_interior)
            (component.elastic_beamcolumn_elements
             .named_contents['elm_interior']) = elm_interior

            component.zerolength_elements.add(zerolen_top_f)
            (component.zerolength_elements
             .named_contents['nonlinear_spring']) = zerolen_top_f
            component.zerolength_elements.add(zerolen_top_b)
            component.zerolength_elements.add(zerolen_bottom_f)
            component.zerolength_elements.add(zerolen_bottom_b)
