"""objects that create component assemblies for a model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from osmg.core.common import EPSILON, STIFF, STIFF_ROT
from osmg.core.osmg_collections import (
    BarAssembly,
    BeamColumnAssembly,
    ComponentAssembly,
)
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
class BaseCreator:
    """Base class for component creators."""

    model: Model = field(repr=False)

    def define_zerolength_fixed_material_creators(
        self,
    ) -> dict[int, MaterialCreator]:
        """
        Define material creators for rigid links.

        Returns:
          Material creator for a rigid uniaxial material for each DOF.

        Raises:
          ValueError: If the model's dimensionality setting is
          not recognized.
        """
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
        return material_creators


@dataclass(repr=False)
class LinkCreator(BaseCreator):
    """Defines TwoNodeLink element components."""

    def add(
        self,
        tags: set[str],
        node_i: Node,
        node_j: Node,
        materials: list[UniaxialMaterial],
        directions: list[int],
        angle=0.00,
    ) -> ComponentAssembly:
        """
        Add a TwoNodeLink component.

        Returns:
          The added component.
        """
        assert node_i.uid != node_j.uid, 'Nodes need to be different.'
        component = ComponentAssembly(
            uid_generator=self.model.uid_generator,
            tags=tags,
        )
        p_i = np.array(node_i.coordinates)
        p_j = np.array(node_j.coordinates)
        axes = local_axes_from_points_and_angle(p_i, p_j, angle)  # type: ignore
        component.elements.add(
            TwoNodeLink(
                uid_generator=self.model.uid_generator,
                nodes=[node_i, node_j],
                materials=materials,
                directions=directions,
                vecyp=axes[1],
            )
        )
        component.external_nodes.add(node_i)
        component.external_nodes.add(node_j)

        # Adding the component in the end. (It needs to have external
        # nodes before adding to the collection).
        self.model.components.add(component)
        return component


@dataclass(repr=False)
class BarCreator(BaseCreator):
    """
    Bar creator object.

    Introduces bar elements to a model.
    Bar elements are linear elements that can only carry axial load.
    """

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
        Add a bar component.

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
        """
        # if there is an offset at the x-end, create an internal node
        # and add a rigidlink element to the component assembly

        if np.linalg.norm(eo_x) < EPSILON:
            # No action needed.
            return node_x

        # '2D Truss', '2D Frame', '3D Truss', '3D Frame'
        n_x = Node(
            uid_generator=self.model.uid_generator,
            coordinates=tuple(x1 + x2 for x1, x2 in zip(node_x.coordinates, eo_x)),
        )
        component.internal_nodes.add(n_x)

        material_creators = self.define_zerolength_fixed_material_creators()
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
    """
    Configuration for `generate_hinged_component_assembly`.

    Only ZeroLength elements at the ends of the beamcolumn sequence.
    """

    zerolength_creator: ZeroLengthCreator


@dataclass(repr=False)
class HingeConfigWithBeamColumnElement(HingeConfig):
    """
    Configuration for `generate_hinged_component_assembly`.

    Two additional beamcolumn sequences, one before the first and one
    after the second ZeroLength element.
    """

    distance: float
    n_sub: int


@dataclass(repr=False)
class HingeConfigWithLink(HingeConfig):
    """
    Configuration for `generate_hinged_component_assembly`.

    Two rigid links, one before the first and one after the second
    ZeroLength element.
    """

    distance: float


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
class BeamColumnCreator(BaseCreator):
    """Introduces beamcolumn elements to a model."""

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

    def generate_hinged_component_assembly(
        self,
        tags: set[str],
        node_i: Node,
        node_j: Node,
        n_sub: int,
        eo_i: numpy_array,
        eo_j: numpy_array,
        section: ElasticSection | FiberSection,
        transf_type: Literal['Linear', 'Corotational', 'PDelta'],
        angle: float,
        initial_deformation_config: InitialDeformationConfig | None = None,
        modified_stiffness_config: ModifiedStiffnessParameterConfig | None = None,
        hinge_config_i: HingeConfig
        | HingeConfigWithBeamColumnElement
        | HingeConfigWithLink
        | None = None,
        hinge_config_j: HingeConfig
        | HingeConfigWithBeamColumnElement
        | HingeConfigWithLink
        | None = None,
    ) -> BeamColumnAssembly:
        """
        Component assembly with hinges at the ends.

        Generates a component assembly that is comprised of beam-column
        elements connected in series with nonlinear springs attached
        at the ends.

        Returns:
            The defined component.
        """
        assert node_i.uid != node_j.uid, 'Nodes need to be different.'

        # Instantiate the component assembly
        component = BeamColumnAssembly(self.model.uid_generator, tags=tags)
        component.external_nodes.add(node_i)
        component.external_nodes.add(node_j)

        p_i = np.array(node_i.coordinates) + eo_i
        p_j = np.array(node_j.coordinates) + eo_j
        x_axis, y_axis, _ = local_axes_from_points_and_angle(p_i, p_j, angle)

        def process_hinge_config(
            node: Node,
            offset: numpy_array,
            hinge_config: HingeConfig
            | HingeConfigWithBeamColumnElement
            | HingeConfigWithLink
            | None,
            *,
            is_end: bool,
        ) -> tuple[Node, numpy_array]:
            """
            Process hinge configuration for a given node.

            This method handles the hinge configuration by creating
            nodes, assigning offsets, and adding elements to the
            component assembly based on the type of hinge
            configuration provided.

            Args:
                node: The node at which the hinge is to be
                  applied.
                offset: The offset for the node's location.
                hinge_config: The hinge configuration for the
                  node. Can be one of the supported hinge types or
                  `None` if no hinge is applied.
                is_end: Whether the node is at the end of the
                  beam-column assembly.

            Returns:
                tuple
                - The connection node after processing the hinge configuration.
                - The offset to be applied to the connection node.

            Raises:
                ValueError: If an offset is specified with
                  `HingeConfig` (not possible).
                TypeError: If the provided `hinge_config` is of an unsupported type.
            """
            if hinge_config is None:
                return node, offset

            location = p_i if not is_end else p_j
            direction = x_axis if not is_end else -x_axis

            nh_out, nh_in = node, None

            if isinstance(hinge_config, HingeConfigWithBeamColumnElement):
                hinge_location = location + direction * hinge_config.distance
                nh_out = Node(
                    uid_generator=self.model.uid_generator,
                    coordinates=tuple(hinge_location),
                )
                nh_in = Node(
                    uid_generator=self.model.uid_generator,
                    coordinates=tuple(hinge_location),
                )
                component.internal_nodes.add(nh_out)
                component.internal_nodes.add(nh_in)

                self.add_beamcolumn_elements_in_series(
                    component=component,
                    node_i=node,
                    node_j=nh_out,
                    eo_i=offset,
                    eo_j=np.zeros(3),
                    n_sub=hinge_config.n_sub,
                    transf_type=transf_type,
                    section=section,
                    angle=angle,
                )
                offset = np.zeros(3)

            elif isinstance(hinge_config, HingeConfigWithLink):
                hinge_location = location + direction * hinge_config.distance
                nh_out = Node(
                    uid_generator=self.model.uid_generator,
                    coordinates=tuple(hinge_location),
                )
                nh_in = Node(
                    uid_generator=self.model.uid_generator,
                    coordinates=tuple(hinge_location),
                )
                component.internal_nodes.add(nh_out)
                component.internal_nodes.add(nh_in)

                material_creators = self.define_zerolength_fixed_material_creators()
                directions, materials = ZeroLengthCreator(
                    uid_generator=self.model.uid_generator,
                    material_creators=material_creators,
                ).generate()

                component.elements.add(
                    TwoNodeLink(
                        uid_generator=self.model.uid_generator,
                        nodes=[node, nh_out],
                        materials=materials,
                        directions=directions,
                    )
                )
                offset = np.zeros(3)

            elif isinstance(hinge_config, HingeConfig):
                if np.linalg.norm(offset) > EPSILON:
                    msg = "Can't have offset with `HingeConfig`. Use `HingeConfigWithElement`."
                    raise ValueError(msg)
                hinge_location = location
                nh_out = node
                nh_in = Node(
                    uid_generator=self.model.uid_generator,
                    coordinates=tuple(hinge_location),
                )
                component.internal_nodes.add(nh_in)
            else:
                msg = f'Invalid hinge_config type: {type(hinge_config)}'
                raise TypeError(msg)

            component.elements.add(
                hinge_config.zerolength_creator.define_element(
                    node_i=nh_out,
                    node_j=nh_in,
                    x_axis=direction,
                    y_axis=y_axis,
                )
            )

            if nh_in is not None:
                nh_return = nh_in
            else:
                nh_return = nh_out

            return nh_return, offset

        # Process hinge configurations for node_i and node_j
        conn_node_i, conn_eo_i = process_hinge_config(
            node_i, eo_i, hinge_config_i, is_end=False
        )
        conn_node_j, conn_eo_j = process_hinge_config(
            node_j, eo_j, hinge_config_j, is_end=True
        )

        # Add the final beam-column elements
        self.add_beamcolumn_elements_in_series(
            component=component,
            node_i=conn_node_i,
            node_j=conn_node_j,
            eo_i=conn_eo_i,
            eo_j=conn_eo_j,
            n_sub=n_sub,
            transf_type=transf_type,
            section=section,
            angle=angle,
            initial_deformation_config=initial_deformation_config,
            modified_stiffness_config=modified_stiffness_config,
        )

        # Adding the component in the end. (It needs to have external
        # nodes before adding to the collection).
        self.model.components.add(component)
        return component
