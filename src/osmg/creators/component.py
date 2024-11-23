"""objects that create component assemblies for a model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from osmg.core.component_assemblies import ComponentAssembly
from osmg.elements.element import (
    DispBeamColumn,
    ElasticBeamColumn,
    GeomTransf,
    ModifiedStiffnessParameterConfig,
)
from osmg.elements.node import Node
from osmg.elements.section import ElasticSection, FiberSection
from osmg.geometry.transformations import (
    local_axes_from_points_and_angle,
    transformation_matrix,
)

if TYPE_CHECKING:
    from osmg.core.model import Model
    from osmg.creators.zerolength import ZeroLengthCreator


nparr = npt.NDArray[np.float64]


@dataclass(repr=False)
class HingeConfig:
    """Configuration for `generate_hinged_component_assembly`."""

    zerolength_creator: ZeroLengthCreator
    distance: float
    n_sub: int
    element_type: Literal['elastic', 'disp']
    transf_type: Literal['Linear', 'Corotational', 'PDelta']


@dataclass(repr=False)
class PanelZoneConfig:
    """Configuration object for panel zones."""

    doubler_plate_thickness: float
    axial_load_ratio: float
    slab_depth: float
    location: Literal['interior', 'exterior_first', 'exterior_last']
    moment_modifier: float
    consider_composite: bool


@dataclass(repr=False)
class InitialDeformationConfig:
    """
    Initial deformation configuration.

    Configuration object for initial deformations of beamcolumn
    elements defined in series.
    """

    camber_2: float
    camber_3: float
    method: Literal['sine', 'parabola']


@dataclass(repr=False)
class BeamColumnCreator:
    """Introduces beamcolumn elements to a model."""

    model: Model = field(repr=False)
    element_type: Literal['elastic', 'disp']

    def __post_init__(self) -> None:
        """
        Code executed after initializing an object.

        Raises:
          ValueError: If an invalid element type is provided.
        """
        if self.element_type not in {'elastic', 'disp'}:
            msg = 'Invalid element type: {element_type.__name__}'
            raise ValueError(msg)

    def define_beamcolumn(
        self,
        node_i: Node,
        node_j: Node,
        offset_i: nparr,
        offset_j: nparr,
        transf_type: Literal['Linear', 'Corotational', 'PDelta'],
        section: ElasticSection | FiberSection,
        angle: float = 0.00,
        modified_stiffness_config: ModifiedStiffnessParameterConfig | None = None,
    ) -> ElasticBeamColumn | DispBeamColumn:
        """
        Define a beamcolumn element.

        Adds a beamcolumn element to the model, connecting the
        specified nodes.

        Returns:
          The added element.

        Raises:
          ValueError: If an invalid element type is provided.
        """
        p_i = np.array(node_i.coordinates) + offset_i
        p_j = np.array(node_j.coordinates) + offset_j
        axes = local_axes_from_points_and_angle(p_i, p_j, angle)  # type: ignore
        if self.element_type == 'elastic':
            assert isinstance(section, ElasticSection)
            transf = GeomTransf(
                uid_generator=self.model.uid_generator,
                transf_type=transf_type,
                offset_i=offset_i,
                offset_j=offset_j,
                x_axis=axes[0],
                y_axis=axes[1],
                z_axis=axes[2],
            )
            elm_el = ElasticBeamColumn(
                uid_generator=self.model.uid_generator,
                nodes=[node_i, node_j],
                section=section,
                geomtransf=transf,
                modified_stiffness_config=modified_stiffness_config,
            )
            res: ElasticBeamColumn | DispBeamColumn = elm_el
        elif self.element_type == 'disp':
            raise NotImplementedError
            # assert isinstance(section, FiberSection)
            # assert modified_stiffness_config is None
            # # TODO(JVM): add elastic section support
            # transf = GeomTransf(
            #     self.model.uid_generator,
            #     transf_type,
            #     offset_i,
            #     offset_j,
            #     *axes,
            # )
            # beam_integration = Lobatto(
            #     uid_generator=self.model.uid_generator,
            #     parent_section=section,
            #     n_p=2,
            # )
            # elm_disp = DispBeamColumn(
            #     uid_generator=self.model.uid_generator,
            #     parent_component=assembly,
            #     nodes=[node_i, node_j],
            #     section=section,
            #     geomtransf=transf,
            #     integration=beam_integration,
            # )
            # res = elm_disp
        else:
            msg = 'Invalid element type: {element_type.__name__}'
            raise ValueError(msg)
        return res

    def add_beamcolumn_elements_in_series(
        self,
        component: ComponentAssembly,
        node_i: Node,
        node_j: Node,
        eo_i: nparr,
        eo_j: nparr,
        n_sub: int,
        transf_type: Literal['Linear', 'Corotational', 'PDelta'],
        section: ElasticSection | FiberSection,
        angle: float,
        initial_deformation_config: InitialDeformationConfig | None = None,
        modified_stiffness_config: ModifiedStiffnessParameterConfig | None = None,
    ) -> None:
        """Add beamcolumn elements in series."""
        if modified_stiffness_config is not None:
            assert n_sub == 1

        num_dimensions = len(node_i.coordinates)

        if n_sub > 1:
            p_i = np.array(node_i.coordinates) + eo_i
            p_j = np.array(node_j.coordinates) + eo_j
            clear_len = np.linalg.norm(p_j - p_i)
            internal_pt_coordinates = np.linspace(
                tuple(p_i), tuple(p_j), num=n_sub + 1
            )

            if initial_deformation_config:
                t_vals = np.linspace(0.00, 1.00, num=n_sub + 1)
                if initial_deformation_config.method == 'parabola':
                    offset_vals = 4.00 * (-(t_vals**2) + t_vals)
                elif initial_deformation_config.method == 'sine':
                    offset_vals = np.sin(np.pi * t_vals)
                offset_2 = (
                    offset_vals * initial_deformation_config.camber_2 * clear_len
                )
                offset_3 = (
                    offset_vals * initial_deformation_config.camber_3 * clear_len
                )
                camber_offset: nparr = np.column_stack(
                    (np.zeros(n_sub + 1), offset_2, offset_3)
                )
                x_axis, y_axis, z_axis = local_axes_from_points_and_angle(
                    p_i, p_j, angle
                )
                assert y_axis is not None
                t_glob_to_loc = transformation_matrix(x_axis, y_axis, z_axis)
                t_loc_to_glob = t_glob_to_loc.T
                camber_offset_global = (t_loc_to_glob @ camber_offset.T).T
                internal_pt_coordinates += camber_offset_global

            intnodes = []
            for i in range(1, len(internal_pt_coordinates) - 1):
                intnode = Node(
                    self.model.uid_generator,
                    (*internal_pt_coordinates[i],),
                )
                component.internal_nodes.add(intnode)
                intnodes.append(intnode)
        for i in range(n_sub):
            if i == 0:
                n_i = node_i
                o_i = eo_i
            else:
                n_i = intnodes[i - 1]
                o_i = np.zeros(num_dimensions)
            if i == n_sub - 1:
                n_j = node_j
                o_j = eo_j
            else:
                n_j = intnodes[i]
                o_j = np.zeros(num_dimensions)
            element = self.define_beamcolumn(
                node_i=n_i,
                node_j=n_j,
                offset_i=o_i,
                offset_j=o_j,
                transf_type=transf_type,
                section=section,
                angle=angle,
                modified_stiffness_config=modified_stiffness_config,
            )
            component.elements.add(element)

    def generate_plain_component_assembly(
        self,
        tags: set[str],
        node_i: Node,
        node_j: Node,
        n_sub: int,
        eo_i: nparr,
        eo_j: nparr,
        section: ElasticSection | FiberSection,
        transf_type: Literal['Linear', 'Corotational', 'PDelta'],
        angle: float = 0.00,
        initial_deformation_config: InitialDeformationConfig | None = None,
    ) -> ComponentAssembly:
        """
        Plain component assembly.

        Generates a plain component assembly with line elements in
        series.

        Returns:
          The created component assembly.
        """
        # TODO(JVM): restore this check
        # uids = [node.uid for node in (node_i, node_j)]
        # uids.sort()
        # uids_tuple = (*uids,)
        # assert uids_tuple not in self.model.component_connectivity()

        # instantiate a component assembly and add it to the model.
        component = ComponentAssembly(self.model.uid_generator, tags=tags)
        self.model.components.add(component)

        # populate the component assembly
        component.external_nodes.add(node_i)
        component.external_nodes.add(node_j)

        self.add_beamcolumn_elements_in_series(
            component=component,
            node_i=node_i,
            node_j=node_j,
            eo_i=eo_i,
            eo_j=eo_j,
            n_sub=n_sub,
            transf_type=transf_type,
            section=section,
            angle=angle,
            initial_deformation_config=initial_deformation_config,
        )

        return component
