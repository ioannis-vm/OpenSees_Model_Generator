"""objects that create component assemblies for a model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from osmg.core.common import EPSILON, STIFF, STIFF_ROT
from osmg.core.osmg_collections import BarAssembly, BeamColumnAssembly
from osmg.creators.material import ElasticMaterialCreator
from osmg.creators.zerolength import ZeroLengthCreator
from osmg.geometry.transformations import (
    local_axes_from_points_and_angle,
    transformation_matrix,
)
from osmg.model_objects.element import (
    Bar,
    DispBeamColumn,
    ElasticBeamColumn,
    GeomTransf,
    ModifiedStiffnessParameterConfig,
    TwoNodeLink,
)
from osmg.model_objects.node import Node
from osmg.model_objects.section import ElasticSection, FiberSection

if TYPE_CHECKING:
    from osmg.core.common import numpy_array
    from osmg.core.model import Model
    from osmg.creators.material import MaterialCreator
    from osmg.geometry.mesh import Mesh
    from osmg.model_objects.uniaxial_material import UniaxialMaterial


@dataclass(repr=False)
class BarGenerator:
    """
    Bar generator object.

    Introduces bar elements to a model.
    Bar elements are linear elements that can only carry axial load.
    """

    model: Model = field(repr=False)

    def add(
        self,
        tags: set[str],
        node_i: Node,
        node_j: Node,
        eo_i: numpy_array,
        eo_j: numpy_array,
        transf_type: Literal['Linear', 'Corotational'],
        area: float,
        material: UniaxialMaterial,
        outside_shape: Mesh | None = None,
        weight_per_length: float = 0.00,
    ) -> BarAssembly:
        """
        Add a bar element.

        If offsets are required, they are implemented through the
        addition of RigidLink elements.

        Returns:
          The added component.
        """
        assert node_i.uid != node_j.uid, 'Nodes need to be different.'
        # TODO(JVM): check connectivity to avoid placing another
        # element on the same pair of nodes (or warn).

        # instantiate a component assembly
        component = BarAssembly(
            uid_generator=self.model.uid_generator,
            tags=tags,
        )
        component.external_nodes.add(node_i)
        component.external_nodes.add(node_j)

        n_i = self._prepare_connection(node_i, eo_i, component)
        n_j = self._prepare_connection(node_j, eo_j, component)

        component.elements.add(
            Bar(
                uid_generator=self.model.uid_generator,
                nodes=[n_i, n_j],
                transf_type=transf_type,
                area=area,
                material=material,
                outside_shape=outside_shape,
                weight_per_length=weight_per_length,
            )
        )

        # Adding the component in the end. (It needs to have external
        # nodes before adding to the collection).
        self.model.components.add(component)
        return component

    def _prepare_connection(
        self, node_x: Node, eo_x: numpy_array, component: BarAssembly
    ) -> Node:
        """
        Add auxiliary elements to account for offsets.

        For each end of the bar element, creates a rigid link if an
        offset exists, and returns the node to which the bar element
        should connect to. This function is called twice, once for the
        i-end and once for the j-end.  For purposes of clarity, the
        index x will be used here, meaning either i or j.

        Returns:
          A newly created node, considering the specified offset.

        Raises:
          ValueError: If the model's dimensionality parameter is not
          supported.
        """
        # if there is an offset at the x-end, create an internal node
        # and add a rigidlink element to the component assembly

        if np.linalg.norm(eo_x) < EPSILON:
            # No action needed.
            return node_x

        # '2D Truss', '2D Frame', '3D Truss', '3D Frame'
        material_creators: dict[int, MaterialCreator]
        if self.model.dimensionality == '2D Truss':
            material_creators = {
                1: ElasticMaterialCreator(self.model, STIFF),
                2: ElasticMaterialCreator(self.model, STIFF),
            }
        elif self.model.dimensionality == '2D Frame':
            material_creators = {
                1: ElasticMaterialCreator(self.model, STIFF),
                2: ElasticMaterialCreator(self.model, STIFF),
                3: ElasticMaterialCreator(self.model, STIFF_ROT),
            }
        elif self.model.dimensionality == '3D Truss':
            material_creators = {
                1: ElasticMaterialCreator(self.model, STIFF),
                2: ElasticMaterialCreator(self.model, STIFF),
                3: ElasticMaterialCreator(self.model, STIFF),
            }
        elif self.model.dimensionality == '3D Frame':
            material_creators = {
                1: ElasticMaterialCreator(self.model, STIFF),
                2: ElasticMaterialCreator(self.model, STIFF),
                3: ElasticMaterialCreator(self.model, STIFF),
                4: ElasticMaterialCreator(self.model, STIFF_ROT),
                5: ElasticMaterialCreator(self.model, STIFF_ROT),
                6: ElasticMaterialCreator(self.model, STIFF_ROT),
            }
        else:
            msg = 'Invalid model dimensionality setting: {model.dimensionality}'
            raise ValueError(msg)

        n_x = Node(
            uid_generator=self.model.uid_generator,
            coordinates=tuple(x1 + x2 for x1, x2 in zip(node_x.coordinates, eo_x)),
        )
        component.internal_nodes.add(n_x)

        directions, materials = ZeroLengthCreator(
            uid_generator=self.model.uid_generator,
            material_creators=material_creators,
        ).generate()

        # flip the nodes if the element is about to be defined
        # upside down
        if (
            np.allclose(
                np.array(node_x.coordinates[0:2]),
                np.array(n_x.coordinates[0:2]),
            )
            and n_x.coordinates[2] > node_x.coordinates[2]
        ):
            first_node = n_x
            second_node = node_x
        else:
            first_node = node_x
            second_node = n_x

        # Note: We don't orient the TwoNodeLink with `vecx`,
        # `vecy`, since it's simply rigid. This also avoids having
        # to separate cases for 2D/3D.
        component.elements.add(
            TwoNodeLink(
                uid_generator=self.model.uid_generator,
                nodes=[first_node, second_node],
                materials=materials,
                directions=directions,
            )
        )
        return n_x


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
        offset_i: numpy_array,
        offset_j: numpy_array,
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
        component: BeamColumnAssembly,
        node_i: Node,
        node_j: Node,
        eo_i: numpy_array,
        eo_j: numpy_array,
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
                camber_offset: numpy_array = np.column_stack(
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
        eo_i: numpy_array,
        eo_j: numpy_array,
        section: ElasticSection | FiberSection,
        transf_type: Literal['Linear', 'Corotational', 'PDelta'],
        angle: float = 0.00,
        initial_deformation_config: InitialDeformationConfig | None = None,
    ) -> BeamColumnAssembly:
        """
        Plain component assembly.

        Generates a plain component assembly with line elements in
        series.

        Returns:
          The created component assembly.
        """
        assert node_i.uid != node_j.uid, 'Nodes need to be different.'
        # TODO(JVM): restore this check
        # uids = [node.uid for node in (node_i, node_j)]
        # uids.sort()
        # uids_tuple = (*uids,)
        # assert uids_tuple not in self.model.component_connectivity()

        # instantiate a component assembly
        component = BeamColumnAssembly(self.model.uid_generator, tags=tags)
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

        # Adding the component in the end. (It needs to have external
        # nodes before adding to the collection).
        self.model.components.add(component)
        return component


@dataclass(repr=False)
class TrussCreator:
    """Introduces truss elements to a model."""

    model: Model = field(repr=False)
