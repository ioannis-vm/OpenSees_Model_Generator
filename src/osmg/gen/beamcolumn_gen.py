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


from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Type
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from ..ops.node import Node
from ..collections import Collection
from ..component_assembly import ComponentAssembly
from .querry import ElmQuerry
from .node_gen import NodeGenerator
from ..ops.element import elasticBeamColumn
from ..ops.element import dispBeamColumn
from ..ops.element import geomTransf
from ..ops.element import Lobatto
from ..ops.section import ElasticSection
from ..ops.section import FiberSection
from ..preprocessing.split_component import split_component
from ..transformations import local_axes_from_points_and_angle
from ..transformations import transformation_matrix
if TYPE_CHECKING:
    from ..model import Model


nparr = npt.NDArray[np.float64]

def retrieve_snap_pt_global_offset(placement, section, p_i, p_j, angle):
    if section.snap_points and (placement != 'centroid'):
        # obtain offset from section (local system)
        dz, dy = section.snap_points[placement]
        sec_offset_local = np.array([0.00, dy, dz])
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


@dataclass(repr=False)
class BeamColumnGenerator:
    model: Model = field(repr=False)


    def define_beamcolumn(
            self,
            assembly: ComponentAssembly,
            node_i: Node,
            node_j: Node,
            offset_i: nparr,
            offset_j: nparr,
            transfType: str,
            section: ElasticSection | FiberSection,
            element_type: Type,
            angle=0.00) -> elasticBeamColumn | dispBeamColumn:

        p_i = np.array(node_i.coords) + offset_i
        p_j = np.array(node_j.coords) + offset_j
        axes = local_axes_from_points_and_angle(
            p_i, p_j, angle)
        if element_type.__name__ == 'elasticBeamColumn':
            assert isinstance(section, ElasticSection)
            transf = geomTransf(
                transfType,
                self.model.uid_generator.new('transformation'),
                offset_i,
                offset_j,
                *axes)
            elm = elasticBeamColumn(
                parent=assembly,
                uid=self.model.uid_generator.new('element'),
                eleNodes=[node_i, node_j],
                section=section,
                geomtransf=transf)
            return elm
        elif element_type.__name__ == 'dispBeamColumn':
            assert isinstance(section, FiberSection)
            # todo: add elastic section support
            transf = geomTransf(
                transfType,
                self.model.uid_generator.new('transformation'),
                offset_i,
                offset_j,
                *axes)
            beam_integration = Lobatto(
                uid=self.model.uid_generator.new('beam integration'),
                parent_section=section,
                n_p=5
            )
            elm = dispBeamColumn(
                parent=assembly,
                uid=self.model.uid_generator.new('element'),
                eleNodes=[node_i, node_j],
                section=section,
                geomtransf=transf,
                integration=beam_integration)
            return elm

    def add_vertical_active(
            self,
            x_coord: float,
            y_coord: float,
            offset_i: nparr,
            offset_j: nparr,
            transfType: str,
            n_sub: int,
            section: ElasticSection | FiberSection,
            element_type: Type,
            placement='centroid',
            angle=0.00):
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
            lvl = lvls.registry[key]
            if key-1 not in lvls.registry:
                continue

            top_node = querry.search_node_lvl(x_coord, y_coord, key)
            if not top_node:
                top_node = ndg.add_node_lvl(x_coord, y_coord, key)

            bottom_node = querry.search_node_lvl(x_coord, y_coord, key-1)
            if not bottom_node:
                bottom_node = ndg.add_node_lvl(x_coord, y_coord, key-1)

            p_i = np.array(top_node.coords) + offset_i
            p_j = np.array(bottom_node.coords) + offset_j

            sec_offset_global = retrieve_snap_pt_global_offset(
                placement, section, p_i, p_j, angle)
            
            p_i += sec_offset_global
            p_j += sec_offset_global
            eo_i = offset_i + sec_offset_global
            eo_j = offset_j + sec_offset_global

        # instantiate a component assembly
        component = ComponentAssembly(
            self.model.uid_generator.new('component'),
            lvl.components)
        # add it to the level
        lvl.components.add(component)
        # fill component assembly
        component.external_nodes.add(top_node)
        component.external_nodes.add(bottom_node)
        
        if n_sub > 1:
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
                n_i = top_node
                o_i = eo_i
            else:
                n_i = intnodes[i-1]
                o_i = np.zeros(3)
            if i == n_sub - 1:
                n_j = bottom_node
                o_j = eo_j
            else:
                n_j = intnodes[i]
                o_j = np.zeros(3)
            element = self.define_beamcolumn(
                assembly=component,
                node_i=n_i, node_j=n_j,
                offset_i=o_i, offset_j=o_j, transfType=transfType,
                section=section,
                element_type=element_type,
                angle=angle)
            if element_type.__name__ == 'elasticBeamColumn':
                component.elastic_beamcolumn_elements.add(element)
            elif element_type.__name__ == 'dispBeamColumn':
                component.disp_beamcolumn_elements.add(element)
            else:
                raise TypeError(
                    'Unsupported element type:'
                    f' {element_type.__name__}')

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
            transfType: str,
            n_sub: int,
            section: ElasticSection,
            element_type: Type,
            placement='centroid',
            angle=0.00,
            split_existing_i=None,
            split_existing_j=None,
            debug=False):
        """
        Adds a horizontal beamcolumn element to all active levels.
        """
        querry = ElmQuerry(self.model)
        ndg = NodeGenerator(self.model)
        lvls = self.model.levels
        assert lvls.active, 'No active levels.'
        for key in lvls.active:
            lvl = lvls.registry[key]
            node_i = querry.search_node_lvl(xi_coord, yi_coord, key)
            node_j = querry.search_node_lvl(xj_coord, yj_coord, key)
            p_i_init = np.array((xi_coord, yi_coord, lvl.elevation)) + offset_i
            p_j_init = np.array((xj_coord, yj_coord, lvl.elevation)) + offset_j
            p_i = np.array((xi_coord, yi_coord, lvl.elevation)) + offset_i
            p_j = np.array((xj_coord, yj_coord, lvl.elevation)) + offset_j
            if section.snap_points and (placement != 'centroid'):
                # obtain offset from section (local system)
                dz, dy = section.snap_points[placement]
                sec_offset_local = np.array([0.00, dy, dz])
                # retrieve local coordinate system
                x_axis, y_axis, z_axis = \
                    local_axes_from_points_and_angle(
                        p_i, p_j, angle)
                t_glob_to_loc = transformation_matrix(
                    x_axis, y_axis, z_axis)
                t_loc_to_glob = t_glob_to_loc.T
                sec_offset_global = t_loc_to_glob @ sec_offset_local
                p_i += sec_offset_global
                p_j += sec_offset_global
            else:
                sec_offset_global = np.zeros(3)
            eo_i = offset_i.copy() + sec_offset_global
            eo_j = offset_j.copy() + sec_offset_global
            if not node_i:
                if split_existing_i:
                    node_i, offset = split_component(split_existing_i, p_i_init)
                    eo_i += offset
                    p_i = np.array((node_i.coords)) + eo_i
            else:
                node_i = querry.search_node_lvl(xi_coord, yi_coord, key)
                if key-1 in lvls.registry:
                    node_i_below = querry.search_node_lvl(
                        xi_coord, yi_coord, key-1)
                    if node_i_below:
                        column = querry.search_connectivity(
                            [node_i, node_i_below])
                        if column:
                            elms = []
                            for dctkey in column.element_connectivity().keys():
                                if node_i.uid in dctkey:
                                    elms.append(
                                        column
                                        .element_connectivity()[dctkey])
                            assert elms, 'There should be an element here.'
                            assert len(elms) == 1, \
                                'There should only be one element here.'
                            elm = elms[0]
                            # obtain offset from section (local system)
                            if elm.section.snap_points:
                                dz, dy = elm.section.snap_points[snap_i]
                                sec_offset_local = np.array([0.00, dy, dz])
                                # retrieve local coordinate system
                                x_axis = elm.geomtransf.x_axis
                                y_axis = elm.geomtransf.y_axis
                                z_axis = elm.geomtransf.z_axis
                                t_glob_to_loc = transformation_matrix(
                                    x_axis, y_axis, z_axis)
                                t_loc_to_glob = t_glob_to_loc.T
                                sec_offset_global = (
                                    t_loc_to_glob @ sec_offset_local)
                                eo_i += sec_offset_global
            if not node_j:
                if split_existing_j:
                    node_j, offset = split_component(split_existing_j, p_j_init)
                    eo_j += offset
                    p_j = np.array((node_j.coords)) + eo_j
            else:
                node_j = querry.search_node_lvl(xj_coord, yj_coord, key)
                if key-1 in lvls.registry:
                    node_j_below = querry.search_node_lvl(
                        xj_coord, yj_coord, key-1)
                    if node_j_below:
                        node_j_below = querry.search_node_lvl(
                            xj_coord, yj_coord, key-1)
                        column = querry.search_connectivity(
                            [node_j, node_j_below])
                        if column:
                            elms = []
                            for dctkey in column.element_connectivity().keys():
                                if node_j.uid in dctkey:
                                    elms.append(
                                        column
                                        .element_connectivity()[dctkey])
                            assert elms, 'There should be an element here.'
                            assert len(elms) == 1, 'There should only be one element here.'
                            elm = elms[0]
                            # obtain offset from section (local system)
                            if elm.section.snap_points:
                                dz, dy = elm.section.snap_points[snap_j]
                                sec_offset_local = np.array([0.00, dy, dz])
                                # retrieve local coordinate system
                                x_axis = elm.geomtransf.x_axis
                                y_axis = elm.geomtransf.y_axis
                                z_axis = elm.geomtransf.z_axis
                                t_glob_to_loc = transformation_matrix(
                                    x_axis, y_axis, z_axis)
                                t_loc_to_glob = t_glob_to_loc.T
                                sec_offset_global = (
                                    t_loc_to_glob @ sec_offset_local)
                                eo_j += sec_offset_global

        # instantiate a component assembly
        component = ComponentAssembly(
            self.model.uid_generator.new('component'),
            lvl.components)
        # add it to the level
        lvl.components.add(component)
        # fill component assembly
        component.external_nodes.add(node_i)
        component.external_nodes.add(node_j)
        
        if n_sub > 1:
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
                offset_i=o_i, offset_j=o_j, transfType=transfType,
                section=section,
                element_type=element_type,
                angle=angle)
            if element_type.__name__ == 'elasticBeamColumn':
                component.elastic_beamcolumn_elements.add(element)
            elif element_type.__name__ == 'dispBeamColumn':
                component.disp_beamcolumn_elements.add(element)
            else:
                raise TypeError(
                    'Unsupported element type:'
                    f' {element_type.__name__}')


































@dataclass(repr=False)
class BeamColumnGenerator:
    model: Model = field(repr=False)

    def define_beamcolumn(
            self,
            assembly: ComponentAssembly,
            node_i: Node,
            node_j: Node,
            offset_i: nparr,
            offset_j: nparr,
            transfType: str,
            section: ElasticSection | FiberSection,
            element_type: Type,
            angle=0.00) -> elasticBeamColumn | dispBeamColumn:

        p_i = np.array(node_i.coords) + offset_i
        p_j = np.array(node_j.coords) + offset_j
        axes = local_axes_from_points_and_angle(
            p_i, p_j, angle)
        if element_type.__name__ == 'elasticBeamColumn':
            assert isinstance(section, ElasticSection)
            transf = geomTransf(
                transfType,
                self.model.uid_generator.new('transformation'),
                offset_i,
                offset_j,
                *axes)
            elm = elasticBeamColumn(
                parent_component=assembly,
                uid=self.model.uid_generator.new('element'),
                eleNodes=[node_i, node_j],
                section=section,
                geomtransf=transf)
            return elm
        elif element_type.__name__ == 'dispBeamColumn':
            assert isinstance(section, FiberSection)
            # todo: add elastic section support
            transf = geomTransf(
                transfType,
                self.model.uid_generator.new('transformation'),
                offset_i,
                offset_j,
                *axes)
            beam_integration = Lobatto(
                uid=self.model.uid_generator.new('beam integration'),
                parent_section=section,
                n_p=5
            )
            elm = dispBeamColumn(
                parent_component=assembly,
                uid=self.model.uid_generator.new('element'),
                eleNodes=[node_i, node_j],
                section=section,
                geomtransf=transf,
                integration=beam_integration)
            return elm

    def generate_plain_component_assembly(
            self,
            lvl,
            node_i,
            node_j,
            p_i,
            p_j,
            n_sub,
            eo_i,
            eo_j,
            section,
            element_type,
            transf_type,
            angle):

        # instantiate a component assembly
        component = ComponentAssembly(
            self.model.uid_generator.new('component'),
            lvl.components)
        # fill component assembly
        component.external_nodes.add(node_i)
        component.external_nodes.add(node_j)
        # add it to the level
        lvl.components.add(component)
        
        if n_sub > 1:
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
                offset_i=o_i, offset_j=o_j, transfType=transf_type,
                section=section,
                element_type=element_type,
                angle=angle)
            if element_type.__name__ == 'elasticBeamColumn':
                component.elastic_beamcolumn_elements.add(element)
            elif element_type.__name__ == 'dispBeamColumn':
                component.disp_beamcolumn_elements.add(element)
            else:
                raise TypeError(
                    'Unsupported element type:'
                    f' {element_type.__name__}')

    def add_vertical_active(
            self,
            x_coord: float,
            y_coord: float,
            offset_i: nparr,
            offset_j: nparr,
            transf_type: str,
            n_sub: int,
            section: ElasticSection | FiberSection,
            element_type: Type,
            placement='centroid',
            angle=0.00):
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
            lvl = lvls.registry[key]
            if key-1 not in lvls.registry:
                continue

            top_node = querry.search_node_lvl(x_coord, y_coord, key)
            if not top_node:
                top_node = ndg.add_node_lvl(x_coord, y_coord, key)

            bottom_node = querry.search_node_lvl(x_coord, y_coord, key-1)
            if not bottom_node:
                bottom_node = ndg.add_node_lvl(x_coord, y_coord, key-1)

            p_i = np.array(top_node.coords) + offset_i
            p_j = np.array(bottom_node.coords) + offset_j
            sec_offset_global = retrieve_snap_pt_global_offset(
                placement, section, p_i, p_j, angle)
            p_i += sec_offset_global
            p_j += sec_offset_global
            eo_i = offset_i + sec_offset_global
            eo_j = offset_j + sec_offset_global

            self.generate_plain_component_assembly(
                lvl, top_node, bottom_node,
                p_i, p_j, n_sub, eo_i, eo_j,
                section, element_type,
                transf_type, angle)

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
            element_type: Type,
            placement='centroid',
            angle=0.00,
            split_existing_i=None,
            split_existing_j=None):
        """
        Adds a horizontal beamcolumn element to all active levels.
        """
        querry = ElmQuerry(self.model)
        ndg = NodeGenerator(self.model)
        lvls = self.model.levels
        assert lvls.active, 'No active levels.'
        for key in lvls.active:
            lvl = lvls.registry[key]
            node_i = querry.search_node_lvl(xi_coord, yi_coord, key)
            node_j = querry.search_node_lvl(xj_coord, yj_coord, key)
            p_i_init = np.array((xi_coord, yi_coord, lvl.elevation)) + offset_i
            p_j_init = np.array((xj_coord, yj_coord, lvl.elevation)) + offset_j
            p_i = np.array((xi_coord, yi_coord, lvl.elevation)) + offset_i
            p_j = np.array((xj_coord, yj_coord, lvl.elevation)) + offset_j
            if section.snap_points and (placement != 'centroid'):
                # obtain offset from section (local system)
                dz, dy = section.snap_points[placement]
                sec_offset_local = np.array([0.00, dy, dz])
                # retrieve local coordinate system
                x_axis, y_axis, z_axis = \
                    local_axes_from_points_and_angle(
                        p_i, p_j, angle)
                t_glob_to_loc = transformation_matrix(
                    x_axis, y_axis, z_axis)
                t_loc_to_glob = t_glob_to_loc.T
                sec_offset_global = t_loc_to_glob @ sec_offset_local
                p_i += sec_offset_global
                p_j += sec_offset_global
            else:
                sec_offset_global = np.zeros(3)
            eo_i = offset_i.copy() + sec_offset_global
            eo_j = offset_j.copy() + sec_offset_global
            if not node_i:
                if split_existing_i:
                    node_i, offset = split_component(split_existing_i, p_i_init)
                    eo_i += offset
                    p_i = np.array((node_i.coords)) + eo_i
            else:
                if key-1 in lvls.registry:
                    node_i_below = querry.search_node_lvl(
                        xi_coord, yi_coord, key-1)
                    if node_i_below:
                        column = querry.search_connectivity(
                            [node_i, node_i_below])
                        if column:
                            elms = []
                            for dctkey in column.element_connectivity().keys():
                                if node_i.uid in dctkey:
                                    elms.append(
                                        column
                                        .element_connectivity()[dctkey])
                            assert elms, 'There should be an element here.'
                            assert len(elms) == 1, \
                                'There should only be one element here.'
                            elm = elms[0]
                            # obtain offset from section (local system)
                            if elm.section.snap_points:
                                dz, dy = elm.section.snap_points[snap_i]
                                sec_offset_local = np.array([0.00, dy, dz])
                                # retrieve local coordinate system
                                x_axis = elm.geomtransf.x_axis
                                y_axis = elm.geomtransf.y_axis
                                z_axis = elm.geomtransf.z_axis
                                t_glob_to_loc = transformation_matrix(
                                    x_axis, y_axis, z_axis)
                                t_loc_to_glob = t_glob_to_loc.T
                                sec_offset_global = (
                                    t_loc_to_glob @ sec_offset_local)
                                eo_i += sec_offset_global
            if not node_j:
                if split_existing_j:
                    node_j, offset = split_component(split_existing_j, p_j_init)
                    eo_j += offset
                    p_j = np.array((node_j.coords)) + eo_j
            else:
                if key-1 in lvls.registry:
                    node_j_below = querry.search_node_lvl(
                        xj_coord, yj_coord, key-1)
                    if node_j_below:
                        node_j_below = querry.search_node_lvl(
                            xj_coord, yj_coord, key-1)
                        column = querry.search_connectivity(
                            [node_j, node_j_below])
                        if column:
                            elms = []
                            for dctkey in column.element_connectivity().keys():
                                if node_j.uid in dctkey:
                                    elms.append(
                                        column
                                        .element_connectivity()[dctkey])
                            assert elms, 'There should be an element here.'
                            assert len(elms) == 1, 'There should only be one element here.'
                            elm = elms[0]
                            # obtain offset from section (local system)
                            if elm.section.snap_points:
                                dz, dy = elm.section.snap_points[snap_j]
                                sec_offset_local = np.array([0.00, dy, dz])
                                # retrieve local coordinate system
                                x_axis = elm.geomtransf.x_axis
                                y_axis = elm.geomtransf.y_axis
                                z_axis = elm.geomtransf.z_axis
                                t_glob_to_loc = transformation_matrix(
                                    x_axis, y_axis, z_axis)
                                t_loc_to_glob = t_glob_to_loc.T
                                sec_offset_global = (
                                    t_loc_to_glob @ sec_offset_local)
                                eo_j += sec_offset_global

            self.generate_plain_component_assembly(
                lvl, node_i, node_j,
                p_i, p_j, n_sub, eo_i, eo_j,
                section, element_type,
                transf_type, angle)
            
