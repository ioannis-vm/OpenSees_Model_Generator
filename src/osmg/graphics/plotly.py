"""Create plotly graphics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import plotly.graph_objects as go  # type: ignore
import random

from osmg.core.common import EPSILON, THREE_DIMENSIONAL, TWO_DIMENSIONAL, numpy_array
from osmg.core.osmg_collections import BeamColumnAssembly
from osmg.geometry.transformations import (
    local_axes_from_points_and_angle,
    offset_transformation_2d,
    offset_transformation_3d,
    transformation_matrix,
)
from osmg.graphics.objects import positioned_arrow
from osmg.model_objects.element import (
    Bar,
    BeamColumnElement,
    DispBeamColumn,
    ElasticBeamColumn,
    TwoNodeLink,
    ZeroLength,
)

if TYPE_CHECKING:
    import pandas as pd

    from osmg.analysis.common import UDL, PointLoad
    from osmg.analysis.supports import ElasticSupport, FixedSupport
    from osmg.core.osmg_collections import ComponentAssembly
    from osmg.model_objects.element import Element
    from osmg.model_objects.node import Node


def _default_camera() -> dict[str, object]:
    """
    Return the default camera configuration for 3D plots.

    Returns:
        A dictionary defining the default camera settings.
    """
    return {
        'up': {'x': 0, 'y': 0, 'z': 1},
        'center': {'x': 0, 'y': 0, 'z': 0},
        'eye': {'x': 0.00, 'y': -10.00, 'z': 0.00},
        # 'projection': {'type': 'perspective'},
        'projection': {'type': 'orthographic'},
    }


@dataclass
class Figure3DConfiguration:
    """Configuration for 3D figures."""

    camera: dict[str, object] = field(default_factory=_default_camera)
    ndm: Literal[2, 3] = field(default=3)


@dataclass
class PlotConfiguration:
    """Base plot configuration class."""

    reference_length: float
    ndm: int
    ndf: int
    data: pd.DataFrame
    step: int


@dataclass
class DeformationConfiguration(PlotConfiguration):
    """Configuration for plotting deformed shapes."""

    amplification_factor: float | None = field(default=None)

    def __post_init__(self) -> None:
        """
        Post-initialization.

        Calculates an amplification factor if not specified.

        Raises:
          ValueError: If the requested step does not exist.
          ValueError: If the model's `NDF` is unsupported.
        """
        # Determine amplification factor
        if self.amplification_factor is not None:
            return
        if self.data.shape[0] < self.step:
            msg = (
                f'The requested step ({self.step}) does not exist. '
                f'The last step is ({self.data.shape[0] - 1})'
            )
            raise ValueError(msg)
        case_2d = 3
        case_3d = 6
        max_deformation = {}
        for i_dof in range(self.ndf):
            max_deformation[i_dof] = (
                self.data.iloc[self.step, i_dof :: self.ndf].abs().max()
            )
        if self.ndf == case_2d:
            max_displacement = np.max((max_deformation[0], max_deformation[1]))
            max_rotation = max_deformation[2]
        elif self.ndf == case_3d:
            max_displacement = np.max(
                (max_deformation[0], max_deformation[1], max_deformation[2])
            )
            max_rotation = np.max(
                (max_deformation[3], max_deformation[4], max_deformation[5])
            )
        else:
            msg = 'Unsupported model NDF: {model.ndf}.'
            raise ValueError(msg)
        # 10% of reference length, or at most an approx. 30 degree rotation angle.
        if np.abs(max_displacement + max_rotation) < EPSILON:
            print('No deformations.')  # noqa: T201
            # TODO(JVM): change to a warning
            self.amplification_factor = 1.0
        elif np.abs(max_rotation) < EPSILON:
            self.amplification_factor = (
                self.reference_length / max_displacement * 0.10
            )
        elif np.abs(max_displacement) < EPSILON:
            self.amplification_factor = 0.50 / max_rotation
        else:
            self.amplification_factor = min(
                self.reference_length / max_displacement * 0.10, 0.50 / max_rotation
            )


@dataclass
class BasicForceConfiguration(PlotConfiguration):
    """Configuration for plotting basic forces."""

    force_to_length_factor: float
    moment_to_length_factor: float


@dataclass(repr=False)
class Figure3D:
    """3D Figure of the model."""

    configuration: Figure3DConfiguration
    data: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post-initialization."""
        self.layout = go.Layout(
            scene={
                'xaxis_visible': False,
                'yaxis_visible': False,
                'zaxis_visible': False,
                'bgcolor': 'white',
                'camera': self.configuration.camera,
                'aspectmode': 'data',
            }
        )

    def find_data_by_name(self, name: str) -> dict[str, object] | None:
        """
        Find a data dictionary with a given name.

        Returns:
          The data dictionary if found, else None.
        """
        for item in self.data:
            if item['name'] == name:
                return item
        return None

    def add_nodes(
        self,
        nodes: list[Node],
        designation: Literal['primary', 'parent', 'internal'],
        deformation_configuration: DeformationConfiguration | None = None,
        *,
        overlay: bool = False,
        random_length: float = 0.00,
    ) -> None:
        """Draw nodes."""
        if deformation_configuration is None:
            self._add_nodes_undeformed(
                nodes, designation, overlay=overlay, random_length=random_length
            )
        else:
            self._add_nodes_deformed(
                nodes, designation, deformation_configuration, overlay=overlay
            )

    def add_components(
        self,
        components: list[ComponentAssembly],
        deformation_configuration: DeformationConfiguration | None = None,
        *,
        overlay: bool = False,
        random_length: float = 0.00
    ) -> None:
        """Add components to the figure."""
        for component in components:
            internal_nodes = list(component.internal_nodes.values())
            self.add_nodes(
                internal_nodes,
                'internal',
                deformation_configuration,
                overlay=overlay,
               random_length=random_length,
            )
            elements = list(component.elements.values())
            self.add_elements(elements, deformation_configuration, overlay=overlay)

    def add_elements(
        self,
        elements: list[Element],
        deformation_configuration: DeformationConfiguration | None = None,
        *,
        overlay: bool = False,
    ) -> None:
        """Add elements to the figure."""
        elastic_beamcolumn_elements: list[ElasticBeamColumn] = []
        disp_beamcolumn_elements: list[DispBeamColumn] = []
        bar_elements: list[Bar] = []
        two_node_link_elements: list[TwoNodeLink] = []
        zerolength_elements: list[ZeroLength] = []
        unknown_types: set[str] = set()
        for element in elements:
            if isinstance(element, ElasticBeamColumn):
                elastic_beamcolumn_elements.append(element)
            elif isinstance(element, DispBeamColumn):
                disp_beamcolumn_elements.append(element)
            elif isinstance(element, Bar):
                bar_elements.append(element)
            elif isinstance(element, TwoNodeLink):
                two_node_link_elements.append(element)
            elif isinstance(element, ZeroLength):
                zerolength_elements.append(element)
            elif element.__class__.__name__ not in unknown_types:
                unknown_types = unknown_types.union({element.__class__.__name__})
        if unknown_types:
            print(  # noqa: T201
                f'WARNING: Skipped the following unknown element types: {unknown_types}.'
            )
            # TODO(JVM): implement warning
        self.add_beamcolumn_elements(
            elastic_beamcolumn_elements,
            deformation_configuration,
            overlay=overlay,
        )
        self.add_beamcolumn_element_offsets(
            elastic_beamcolumn_elements,
            deformation_configuration,
            overlay=overlay,
        )
        self.add_beamcolumn_elements(
            disp_beamcolumn_elements,
            deformation_configuration,
            overlay=overlay,
        )
        self.add_beamcolumn_element_offsets(
            disp_beamcolumn_elements,
            deformation_configuration,
            overlay=overlay,
        )
        self.add_bar_elements(
            bar_elements,
            deformation_configuration,
            overlay=overlay,
        )
        # self.add_two_node_link_elements(
        #     two_node_link_elements,
        #     deformation_configuration,
        #     overlay=overlay,
        # )

    def add_bar_elements(
        self,
        elements: list[Bar],
        deformation_configuration: DeformationConfiguration | None = None,
        *,
        overlay: bool = False,
    ) -> None:
        """
        Add bar elements to the 3D figure.

        Args:
            elements: A list of bar elements to be added to the figure.
            deformation_configuration: Configuration for deformed shape plotting.
                If None, the undeformed geometry is plotted.
            overlay: If True, reduce the opacity of the plotted elements.
        """
        if overlay:
            opacity = 0.5
        else:
            opacity = 1.0

        if not elements:
            return

        coordinates_i: list[tuple[float, ...]] = []
        coordinates_j: list[tuple[float, ...]] = []
        for element in elements:
            coordinates_i.append(element.nodes[0].coordinates)
            coordinates_j.append(element.nodes[1].coordinates)
        coordinates_i_array = np.array(coordinates_i)
        coordinates_j_array = np.array(coordinates_j)
        if deformation_configuration is not None:
            displacements_i: list[tuple[float, ...]] = []
            displacements_j: list[tuple[float, ...]] = []
            for element in elements:
                displacements_i.append(
                    deformation_configuration.data.iloc[
                        deformation_configuration.step
                    ][element.nodes[0].uid]
                )
                displacements_j.append(
                    deformation_configuration.data.iloc[
                        deformation_configuration.step
                    ][element.nodes[1].uid]
                )
            displacements_i_array = np.array(displacements_i)[
                :, 0 : deformation_configuration.ndm
            ]
            displacements_j_array = np.array(displacements_j)[
                :, 0 : deformation_configuration.ndm
            ]
            coordinates_i_array += (
                displacements_i_array
                * deformation_configuration.amplification_factor
            )
            coordinates_j_array += (
                displacements_j_array
                * deformation_configuration.amplification_factor
            )

        # Cast to 3D for plotting.
        if coordinates_i_array.shape[1] == TWO_DIMENSIONAL:
            coordinates_i_array = np.insert(coordinates_i_array, 1, 0.00, axis=1)
            coordinates_j_array = np.insert(coordinates_j_array, 1, 0.00, axis=1)

        if deformation_configuration is not None:
            name = 'Bar Elements (Deformed)'
        else:
            name = 'Bar Elements'
        data = self.find_data_by_name(name)
        if not data:
            # Initialize
            data = {
                'name': name,
                'type': 'scatter3d',
                'mode': 'lines',
                'x': [],
                'y': [],
                'z': [],
                'opacity': opacity,
                'hoverinfo': 'skip',
                'line': {'width': 5, 'color': '#0f24db'},
            }
            self.data.append(data)

        for coordinates_i, coordinates_j in zip(
            coordinates_i_array, coordinates_j_array
        ):
            data['x'].extend((coordinates_i[0], coordinates_j[0], None))  # type: ignore
            data['y'].extend((coordinates_i[1], coordinates_j[1], None))  # type: ignore
            data['z'].extend((coordinates_i[2], coordinates_j[2], None))  # type: ignore

    def add_beamcolumn_elements(
        self,
        elements: list[ElasticBeamColumn] | list[DispBeamColumn],
        deformation_configuration: DeformationConfiguration | None = None,
        *,
        overlay: bool = False,
    ) -> None:
        """Add beamcolumn elements to the figure."""
        if deformation_configuration is None:
            self._add_beamcolumn_elements_undeformed(elements, overlay=overlay)
        else:
            self._add_beamcolumn_elements_deformed(
                elements, deformation_configuration, overlay=overlay
            )

    def add_beamcolumn_element_offsets(
        self,
        elements: list[ElasticBeamColumn] | list[DispBeamColumn],
        deformation_configuration: DeformationConfiguration | None = None,
        *,
        overlay: bool = False,
    ) -> None:
        """Add beamcolumn element offsets to the figure."""
        if deformation_configuration is None:
            self._add_beamcolumn_element_offsets_undeformed(
                elements, overlay=overlay
            )
        else:
            self._add_beamcolumn_element_offsets_deformed(
                elements, deformation_configuration, overlay=overlay
            )

    def add_supports(  # noqa: C901
        self,
        nodes: dict[int, Node],
        supports: dict[int, FixedSupport] | dict[int, ElasticSupport],
        symbol_size: float,
    ) -> None:
        """Show supports."""
        # Verify dimensionality consistency
        assert supports != {}, 'Supports are empty'
        support_iterator = iter(supports)
        uid = next(support_iterator)
        support = supports[uid]
        num_dofs = len(support)
        # sanity check
        # continue iterating and ansure the dimensions are correct
        for uid in support_iterator:
            support = supports[uid]
            assert num_dofs == len(support)

        # Verify dimensionality consistency
        node_iterator = iter(nodes)
        uid = next(node_iterator)
        node = nodes[uid]
        num_dimensions = len(node.coordinates)
        # sanity check
        # continue iterating and ansure the dimensions are correct
        for uid in node_iterator:
            node = nodes[uid]
            assert num_dimensions == len(node.coordinates)

        if num_dimensions == THREE_DIMENSIONAL:
            directions: dict[int, Literal['x', 'y', 'z', '-x', '-y', '-z']] = {
                0: '-x',
                1: '-y',
                2: '-z',
                3: 'x',
                4: 'y',
                5: 'z',
            }
        else:
            assert num_dimensions == TWO_DIMENSIONAL
            directions = {0: '-x', 1: '-z', 2: 'y'}

        data = self.find_data_by_name('Supports')
        if not data:
            # Initialize
            data = {
                'name': 'Supports',
                'type': 'mesh3d',
                'x': [],
                'y': [],
                'z': [],
                'i': [],
                'j': [],
                'k': [],
                'color': '#00f0ff',
                'hoverinfo': 'skip',
                'showlegend': True,
            }
            # TODO (JVM): replace template fields with actual info.
            self.data.append(data)

        index_offset = 0
        for uid, support in supports.items():
            node = nodes[uid]
            if self.configuration.ndm == THREE_DIMENSIONAL:
                for i in range(num_dofs):
                    if support[i]:
                        tip_coordinates = (
                            node.coordinates[0],
                            node.coordinates[1],
                            node.coordinates[2],
                        )
                        (
                            vertices_x,
                            vertices_y,
                            vertices_z,
                            faces_i,
                            faces_j,
                            faces_k,
                        ) = self._generate_pyramid_mesh(
                            tip_coordinates,
                            symbol_size,
                            directions[i],
                            index_offset,
                        )
                        index_offset += 5
                        data['x'].extend(vertices_x)  # type: ignore
                        data['y'].extend(vertices_y)  # type: ignore
                        data['z'].extend(vertices_z)  # type: ignore
                        data['i'].extend(faces_i)  # type: ignore
                        data['j'].extend(faces_j)  # type: ignore
                        data['k'].extend(faces_k)  # type: ignore
            else:
                assert self.configuration.ndm == TWO_DIMENSIONAL
                for i in range(num_dofs):
                    if support[i]:
                        tip_coordinates = (
                            node.coordinates[0],
                            0.00,
                            node.coordinates[1],
                        )
                        (
                            vertices_x,
                            vertices_y,
                            vertices_z,
                            faces_i,
                            faces_j,
                            faces_k,
                        ) = self._generate_pyramid_mesh(
                            tip_coordinates,
                            symbol_size,
                            directions[i],
                            index_offset,
                        )
                        index_offset += 5
                        data['x'].extend(vertices_x)  # type: ignore
                        data['y'].extend(vertices_y)  # type: ignore
                        data['z'].extend(vertices_z)  # type: ignore
                        data['i'].extend(faces_i)  # type: ignore
                        data['j'].extend(faces_j)  # type: ignore
                        data['k'].extend(faces_k)  # type: ignore

    def add_udl(
        self,
        udl: dict[int, UDL],
        components: dict[int, ComponentAssembly],
        force_to_length_factor: float,
        offset: float,
    ) -> None:
        """Show uniformly distributed load applied on the components."""
        for component_uid, global_udl in udl.items():
            component = components[component_uid]
            assert isinstance(component, BeamColumnAssembly)
            for element in component.elements.values():
                if not isinstance(element, BeamColumnElement):
                    continue
                start_vec = np.array(element.nodes[0].coordinates)
                end_vec = np.array(element.nodes[1].coordinates)
                udl_vec = np.array(global_udl)

                assert len(start_vec) == len(end_vec) == len(udl_vec)

                if len(start_vec) == TWO_DIMENSIONAL:
                    # Add 0.00 to the Y axis (second element)
                    start_vec = np.insert(start_vec, 1, 0.00)
                    end_vec = np.insert(end_vec, 1, 0.00)
                    udl_vec = np.insert(udl_vec, 1, 0.00)

                else:
                    assert len(start_vec) == THREE_DIMENSIONAL

                udl_vec = -udl_vec  # Reverse direction
                udl_vec_normalized = udl_vec / np.linalg.norm(udl_vec)
                start_vertex = start_vec + udl_vec_normalized * offset
                end_vertex = end_vec + udl_vec_normalized * offset
                start_vertex_top = start_vertex + udl_vec * force_to_length_factor
                end_vertex_top = end_vertex + udl_vec * force_to_length_factor

                self._generate_quadrilateral_mesh(
                    (
                        tuple(start_vertex),
                        tuple(end_vertex),
                        tuple(end_vertex_top),
                        tuple(start_vertex_top),
                    ),
                    name='Uniformly Distributed Loads',
                    value=(
                        str(global_udl),
                        str(global_udl),
                        str(global_udl),
                        str(global_udl),
                    ),
                    color='#7ac4b7',
                    opacity=0.5,
                )

    def add_loads(
        self,
        load: dict[int, PointLoad],
        nodes: dict[int, Node],
        force_to_length_factor: float,
        offset: float,
        head_length: float,
        head_width: float,
        base_width: float,
    ) -> None:
        """
        Show point loads applied on nodes.

        Moments are ignored.
        """
        for node_uid, point_load in load.items():
            node = nodes[node_uid]
            start_vec = np.array(node.coordinates)
            load_vec = np.array(point_load[0 : len(start_vec)])
            load_magnitude = np.linalg.norm(load_vec)
            load_direction = load_vec / load_magnitude
            end_vec = start_vec - load_vec * force_to_length_factor
            start_vec -= load_direction * offset
            end_vec -= load_direction * offset

            assert len(start_vec) == len(end_vec)

            if len(start_vec) == TWO_DIMENSIONAL:
                # Add 0.00 to the Y axis (second element)
                start_vec = np.insert(start_vec, 1, 0.00)
                end_vec = np.insert(end_vec, 1, 0.00)
            else:
                assert len(start_vec) == THREE_DIMENSIONAL

            self._generate_arrow(
                tuple(start_vec),
                tuple(end_vec),
                head_length,
                head_width,
                base_width,
                name='Nodal Loads',
                value=str(point_load),
            )

    def add_basic_forces(  # noqa: C901
        self,
        components: list[ComponentAssembly],
        basic_force_configuration: BasicForceConfiguration,
    ) -> None:
        """Add basic forces on linear elements."""
        elastic_beamcolumn_elements: list[ElasticBeamColumn] = []
        disp_beamcolumn_elements: list[DispBeamColumn] = []
        bar_elements: list[Bar] = []
        for component in components:
            elements = list(component.elements.values())
            for element in elements:
                if element.visibility.skip_opensees_definition:
                    continue
                if isinstance(element, ElasticBeamColumn):
                    elastic_beamcolumn_elements.append(element)
                elif isinstance(element, DispBeamColumn):
                    disp_beamcolumn_elements.append(element)
                elif isinstance(element, Bar):
                    if element.transf_type != 'Corotational':
                        # Crurently `localForce` is not a valid
                        # recorder argument for corotational truss
                        # elements, so we don't capture their basic
                        # forces. A dedicated recorder with
                        # `axialForce` can be used instead if needed.
                        bar_elements.append(element)

        axial, shear_y, shear_z, torsion, moment_y, moment_z = (
            basic_force_configuration.data
        )
        axial = axial.iloc[basic_force_configuration.step, :]
        shear_y = shear_y.iloc[basic_force_configuration.step, :]
        shear_z = shear_z.iloc[basic_force_configuration.step, :]
        torsion = torsion.iloc[basic_force_configuration.step, :]
        moment_y = moment_y.iloc[basic_force_configuration.step, :]
        moment_z = moment_z.iloc[basic_force_configuration.step, :]
        force_factor = basic_force_configuration.force_to_length_factor
        moment_factor = basic_force_configuration.moment_to_length_factor
        ignored_uids = []
        for element in (
            elastic_beamcolumn_elements + disp_beamcolumn_elements + bar_elements
        ):
            if element.uid not in axial:
                ignored_uids.append(element.uid)
                continue
            element_length = element.clear_length()
            stations = (
                axial[element.uid].index.get_level_values('station').to_numpy()
            )
            axial_values = axial[element.uid].to_numpy() * force_factor
            torsion_values = torsion[element.uid].to_numpy() * moment_factor
            shear_y_values = shear_y[element.uid].to_numpy() * force_factor
            shear_z_values = shear_z[element.uid].to_numpy() * force_factor
            moment_y_values = moment_y[element.uid].to_numpy() * moment_factor
            moment_z_values = moment_z[element.uid].to_numpy() * moment_factor
            x_values = element_length * stations

            # Define vertices positioned at the local axis origin.
            start_point = np.array(element.nodes[0].coordinates)

            if isinstance(element, Bar):
                x_axis, y_axis, z_axis = local_axes_from_points_and_angle(
                    np.array(element.nodes[0].coordinates),
                    np.array(element.nodes[1].coordinates),
                    ang=0.00,
                )
            else:
                x_axis = element.geomtransf.x_axis
                y_axis = element.geomtransf.y_axis
                z_axis = element.geomtransf.z_axis
                start_point += element.geomtransf.offset_i

            if y_axis is None:  # 2D case
                x_axis = np.insert(x_axis, 1, 0.00)
                y_axis = np.insert(z_axis, 1, 0.00)
                z_axis = np.cross(x_axis, y_axis)
                start_point = np.insert(start_point, 1, 0.00)

            tm = np.column_stack((x_axis, y_axis, z_axis))  # transformation matrix

            # Add axial load

            for description, values_local, plot_direction, color, factor in (
                ('N', axial_values, 'local_2', '#c47b04', force_factor),
                ('T', torsion_values, 'local_2', '#38a89d', moment_factor),
                ('Vy', shear_y_values, 'local_2', '#148231', force_factor),
                ('Vz', shear_z_values, 'local_3', '#148231', force_factor),
                ('My', -moment_y_values, 'local_3', '#3f1e8a', moment_factor),  # *
                ('Mz', -moment_z_values, 'local_2', '#3f1e8a', moment_factor),  # *
            ):
                # * We flip the signs for the moments by convention.
                for i in range(len(stations) - 1):
                    if plot_direction == 'local_2':
                        local_vertices = np.array(
                            (
                                (x_values[i], 0.00, 0.00),
                                (x_values[i + 1], 0.00, 0.00),
                                (x_values[i + 1], values_local[i + 1], 0.00),
                                (x_values[i], values_local[i], 0.00),
                            )
                        )
                    elif plot_direction == 'local_3':
                        local_vertices = np.array(
                            (
                                (x_values[i], 0.00, 0.00),
                                (x_values[i + 1], 0.00, 0.00),
                                (x_values[i + 1], 0.00, values_local[i + 1]),
                                (x_values[i], 0.00, values_local[i]),
                            )
                        )
                    global_vertices = (tm @ local_vertices.T).T + start_point
                    self._generate_quadrilateral_mesh(
                        tuple(tuple(row) for row in global_vertices),  # type: ignore
                        name=description,
                        value=(
                            '',
                            '',
                            f'{values_local[i + 1] / factor:.2f}',
                            f'{values_local[i] / factor:.2f}',
                        ),
                        color=color,
                        opacity=0.80,
                        start_hidden=True,
                    )
        if ignored_uids:
            print(  # noqa: T201
                f'Ignored the following elements with missing data: {set(ignored_uids)}'
            )

    def _add_nodes_undeformed(
        self,
        nodes: list[Node],
        designation: Literal['primary', 'parent', 'internal'],
        *,
        overlay: bool = False,
        random_length: float = 0.00,
    ) -> None:
        """Draw nodes."""
        expanded_designation = {
            'primary': 'Primary Nodes',
            'parent': 'Parent Nodes',
            'internal': 'Internal Nodes',
        }
        marker = {
            'primary': ('circle', 3, '#7ac4b7'),
            'parent': ('circle-open', 15, '#7ac4b7'),
            'internal': ('x', 2, '#a832a8'),
        }
        if overlay:
            opacity = 0.5
        else:
            opacity = 1.0
        data = self.find_data_by_name(expanded_designation[designation])
        if not data:
            # Initialize
            data = {
                'name': expanded_designation[designation],
                'type': 'scatter3d',
                'mode': 'markers',
                'x': [],
                'y': [],
                'z': [],
                'customdata': [],
                'text': [],
                'hovertemplate': (
                    'Node ID: %{customdata[0]:d}<br>'
                    'Coordinates: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>'
                    '<extra></extra>'
                ),
                'marker': {
                    'symbol': marker[designation][0],
                    'color': marker[designation][2],
                    'size': marker[designation][1],
                    'opacity': opacity,
                    'line': {
                        'color': marker[designation][2],
                        'width': 4,
                    },
                },
            }
            self.data.append(data)
        for node in nodes:
            if self.configuration.ndm == THREE_DIMENSIONAL:
                data['x'].append(
                    node.coordinates[0] + random.uniform(-random_length/2.0, random_length/2.0)
                )  # type: ignore
                data['y'].append(
                    node.coordinates[1] + random.uniform(-random_length/2.0, random_length/2.0)
                )  # type: ignore
                data['z'].append(
                    node.coordinates[2] + random.uniform(-random_length/2.0, random_length/2.0)
                )  # type: ignore
                data['customdata'].append([node.uid, None])  # type: ignore
            else:
                assert self.configuration.ndm == TWO_DIMENSIONAL
                data['x'].append(
                    node.coordinates[0] + random.uniform(-random_length/2.0, random_length/2.0)
                )  # type: ignore
                data['y'].append(0.00)  # type: ignore
                data['z'].append(
                    node.coordinates[1] + random.uniform(-random_length/2.0, random_length/2.0)
                )  # type: ignore
                data['customdata'].append([node.uid, None])  # type: ignore

    def _add_nodes_deformed(
        self,
        nodes: list[Node],
        designation: Literal['primary', 'parent', 'internal'],
        deformation_configuration: DeformationConfiguration,
        *,
        overlay: bool = False,
    ) -> None:
        """
        Draw nodes in their displaced location.

        Raises:
          ValueError: If displacement data is unavailable for some node.
        """
        expanded_designation = {
            'primary': 'Primary Nodes (Deformed)',
            'parent': 'Parent Nodes (Deformed)',
            'internal': 'Internal Nodes (Deformed)',
        }
        marker = {
            'primary': ('circle', 3, '#7ac4b7'),
            'parent': ('circle-open', 15, '#7ac4b7'),
            'internal': ('x', 2, '#a832a8'),
        }
        if overlay:
            opacity = 0.5
        else:
            opacity = 1.0
        data = self.find_data_by_name(expanded_designation[designation])
        if not data:
            # Initialize
            data = {
                'name': expanded_designation[designation],
                'type': 'scatter3d',
                'mode': 'markers',
                'x': [],
                'y': [],
                'z': [],
                'customdata': [],
                'text': [],
                'hovertemplate': (
                    'Node ID: %{customdata[0]:d}<br>'
                    'Deformations: %{customdata[1]}<br>'
                    '<extra></extra>'
                ),
                'marker': {
                    'symbol': marker[designation][0],
                    'color': marker[designation][2],
                    'size': marker[designation][1],
                    'opacity': opacity,
                    'line': {
                        'color': marker[designation][2],
                        'width': 4,
                    },
                },
            }
            self.data.append(data)

        all_node_deformations = deformation_configuration.data
        for node in nodes:
            if node.uid not in all_node_deformations.columns:
                msg = f'Results not available for node: {node.uid}.'
                raise ValueError(msg)
            if self.configuration.ndm == THREE_DIMENSIONAL:
                node_deformations = (
                    all_node_deformations.iloc[deformation_configuration.step, :]
                    .loc[node.uid]
                    .to_numpy()
                )
                data['x'].append(  # type: ignore
                    node.coordinates[0]
                    + node_deformations[0]
                    * deformation_configuration.amplification_factor
                )
                data['y'].append(  # type: ignore
                    node.coordinates[1]
                    + node_deformations[1]
                    * deformation_configuration.amplification_factor
                )
                data['z'].append(  # type: ignore
                    node.coordinates[2]
                    + node_deformations[2]
                    * deformation_configuration.amplification_factor
                )
                data['customdata'].append(  # type: ignore
                    [node.uid, [f'{v:.2f}' for v in node_deformations]]
                )
            else:
                assert self.configuration.ndm == TWO_DIMENSIONAL
                node_deformations = (
                    all_node_deformations.iloc[deformation_configuration.step, :]
                    .loc[node.uid]
                    .to_numpy()
                )
                data['x'].append(  # type: ignore
                    node.coordinates[0]
                    + node_deformations[0]
                    * deformation_configuration.amplification_factor
                )
                data['y'].append(0.00)  # type: ignore
                data['z'].append(  # type: ignore
                    node.coordinates[1]
                    + node_deformations[1]
                    * deformation_configuration.amplification_factor
                )
                data['customdata'].append(  # type: ignore
                    [node.uid, [f'{v:.2f}' for v in node_deformations]]
                )

    def _add_beamcolumn_elements_undeformed(
        self,
        elements: list[ElasticBeamColumn] | list[DispBeamColumn],
        *,
        overlay: bool = False,
    ) -> None:
        """Add beamcolumn elements to the figure."""
        if overlay:
            opacity = 0.5
        else:
            opacity = 1.0
        data = self.find_data_by_name('Beamcolumn Elements')
        if not data:
            # Initialize
            data = {
                'name': 'Beamcolumn Elements',
                'type': 'scatter3d',
                'mode': 'lines',
                'x': [],
                'y': [],
                'z': [],
                'text': [],
                'customdata': [],
                'opacity': opacity,
                'hovertemplate': (
                    'Section: %{text}<br>'
                    '<extra>Element: %{customdata[0]:d}<br>'
                    'Node @ this end: %{customdata[1]:d}</extra>'
                ),
                'line': {'width': 5, 'color': '#0f24db'},
            }
            self.data.append(data)

        x_list: list[float | None] = []
        y_list: list[float | None] = []
        z_list: list[float | None] = []
        customdata_list = []  # type: ignore
        section_names = []  # type: ignore
        for element in elements:
            if element.visibility.hidden_at_line_plots:
                continue
            p_i = (
                np.array(element.nodes[0].coordinates) + element.geomtransf.offset_i
            )
            p_j = (
                np.array(element.nodes[1].coordinates) + element.geomtransf.offset_j
            )
            section_name = element.section.name
            section_names.extend([section_name] * 3)
            if self.configuration.ndm == THREE_DIMENSIONAL:
                x_list.extend((p_i[0], p_j[0], None))
                y_list.extend((p_i[1], p_j[1], None))
                z_list.extend((p_i[2], p_j[2], None))
            else:
                assert self.configuration.ndm == TWO_DIMENSIONAL
                x_list.extend((p_i[0], p_j[0], None))
                y_list.extend((0.0, 0.0, None))
                z_list.extend((p_i[1], p_j[1], None))

            customdata_list.extend(
                (
                    (
                        element.uid,
                        element.nodes[0].uid,
                    ),
                    (
                        element.uid,
                        element.nodes[1].uid,
                    ),
                    (None,) * 3,
                )
            )

        data['x'].extend(x_list)  # type: ignore
        data['y'].extend(y_list)  # type: ignore
        data['z'].extend(z_list)  # type: ignore
        data['customdata'].extend(customdata_list)  # type: ignore
        data['text'].extend(section_names)  # type: ignore

    def _add_beamcolumn_elements_deformed(
        self,
        elements: list[ElasticBeamColumn] | list[DispBeamColumn],
        deformation_configuration: DeformationConfiguration,
        *,
        overlay: bool = False,
    ) -> None:
        """
        Add deformed beamcolumn elements.

        Raises:
          ValueError: If displacement data is unavailable for some node.
        """
        if overlay:
            opacity = 0.5
        else:
            opacity = 1.0
        data = self.find_data_by_name('Beamcolumn Elements (Deformed)')
        if not data:
            # Initialize
            data = {
                'name': 'Beamcolumn Elements (Deformed)',
                'type': 'scatter3d',
                'mode': 'lines',
                'x': [],
                'y': [],
                'z': [],
                'opacity': opacity,
                'hoverinfo': 'skip',
                'line': {'width': 5, 'color': '#0f24db'},
            }
            self.data.append(data)

        x_list: list[float | None] = []
        y_list: list[float | None] = []
        z_list: list[float | None] = []
        num_points = 10
        node_deformations = deformation_configuration.data
        amplification_factor = deformation_configuration.amplification_factor
        assert amplification_factor is not None
        step = deformation_configuration.step
        for element in elements:
            if element.visibility.hidden_at_line_plots:
                continue

            node_i = element.nodes[0].uid
            node_j = element.nodes[1].uid
            for node in (node_i, node_j):
                if node not in node_deformations.columns:
                    msg = 'Results not available for node: {node.uid}.'
                    raise ValueError(msg)

            if self.configuration.ndm == THREE_DIMENSIONAL:
                u_i = node_deformations.iloc[step][node_i].to_numpy()[0:3]
                r_i = node_deformations.iloc[step][node_i].to_numpy()[3:6]
                u_j = node_deformations.iloc[step][node_j].to_numpy()[0:3]
                r_j = node_deformations.iloc[step][node_j].to_numpy()[3:6]
                offset_i = element.geomtransf.offset_i
                offset_j = element.geomtransf.offset_j
                u_i_o = offset_transformation_3d(offset_i, u_i, r_i)
                u_j_o = offset_transformation_3d(offset_j, u_j, r_j)
                d_global, _ = self._interpolate_deformation_3d(
                    element, u_i_o, r_i, u_j_o, r_j, num_points
                )
                interpolation_points = self._interpolate_points(
                    element, d_global, num_points, amplification_factor
                )
            elif self.configuration.ndm == TWO_DIMENSIONAL:
                u_i = node_deformations.iloc[step][node_i].to_numpy()[0:2]
                r_i = node_deformations.iloc[step][node_i].to_numpy()[2]
                u_j = node_deformations.iloc[step][node_j].to_numpy()[0:2]
                r_j = node_deformations.iloc[step][node_j].to_numpy()[2]
                offset_i = element.geomtransf.offset_i
                offset_j = element.geomtransf.offset_j
                u_i_o = offset_transformation_2d(offset_i, u_i, r_i)
                u_j_o = offset_transformation_2d(offset_j, u_j, r_j)
                d_global, _ = self._interpolate_deformation_2d(
                    element, u_i_o, r_i, u_j_o, r_j, num_points
                )
                interpolation_points = self._interpolate_points(
                    element, d_global, num_points, amplification_factor
                )
                interpolation_points = np.insert(
                    interpolation_points, 1, 0.00, axis=1
                )
            else:
                msg = 'Unsupported NDM: {self.configuration.ndm}.'
                raise ValueError(msg)

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

        data['x'].extend(x_list)  # type: ignore
        data['y'].extend(y_list)  # type: ignore
        data['z'].extend(z_list)  # type: ignore

    def _add_beamcolumn_element_offsets_undeformed(
        self,
        elements: list[ElasticBeamColumn] | list[DispBeamColumn],
        *,
        overlay: bool = False,
    ) -> None:
        """Add undeformed beamcolumn element offsets."""
        if overlay:
            opacity = 0.5
        else:
            opacity = 1.0
        data = self.find_data_by_name('Rigid Offsets')
        if not data:
            # Initialize
            data = {
                'name': 'Rigid Offsets',
                'type': 'scatter3d',
                'mode': 'lines',
                'x': [],
                'y': [],
                'z': [],
                'opacity': opacity,
                'hoverinfo': 'skip',
                'line': {'width': 8, 'color': '#a83256'},
            }
            self.data.append(data)

        x_list: list[float | None] = []
        y_list: list[float | None] = []
        z_list: list[float | None] = []
        for element in elements:
            if element.visibility.hidden_at_line_plots:
                continue
            p_i = np.array(element.nodes[0].coordinates)
            p_io = (
                np.array(element.nodes[0].coordinates) + element.geomtransf.offset_i
            )
            p_j = np.array(element.nodes[1].coordinates)
            p_jo = (
                np.array(element.nodes[1].coordinates) + element.geomtransf.offset_j
            )

            if self.configuration.ndm == THREE_DIMENSIONAL:
                x_list.extend((p_i[0], p_io[0], None))
                y_list.extend((p_i[1], p_io[1], None))
                z_list.extend((p_i[2], p_io[2], None))
                x_list.extend((p_j[0], p_jo[0], None))
                y_list.extend((p_j[1], p_jo[1], None))
                z_list.extend((p_j[2], p_jo[2], None))
            if self.configuration.ndm == TWO_DIMENSIONAL:
                x_list.extend((p_i[0], p_io[0], None))
                y_list.extend((0.0, 0.0, None))
                z_list.extend((p_i[1], p_io[1], None))
                x_list.extend((p_j[0], p_jo[0], None))
                y_list.extend((0.0, 0.0, None))
                z_list.extend((p_j[1], p_jo[1], None))

        data['x'].extend(x_list)  # type: ignore
        data['y'].extend(y_list)  # type: ignore
        data['z'].extend(z_list)  # type: ignore

    def _add_beamcolumn_element_offsets_deformed(
        self,
        elements: list[ElasticBeamColumn] | list[DispBeamColumn],
        deformation_configuration: DeformationConfiguration,
        *,
        overlay: bool = False,
    ) -> None:
        """
        Add deformed beamcolumn element offsets.

        Raises:
          ValueError: If displacement data is unavailable for some node.
        """
        if overlay:
            opacity = 0.5
        else:
            opacity = 1.0
        data = self.find_data_by_name('Rigid Offsets (Deformed)')
        if not data:
            # Initialize
            data = {
                'name': 'Rigid Offsets (Deformed)',
                'type': 'scatter3d',
                'mode': 'lines',
                'x': [],
                'y': [],
                'z': [],
                'opacity': opacity,
                'hoverinfo': 'skip',
                'line': {'width': 8, 'color': '#a83256'},
            }
            self.data.append(data)

        x_list: list[float | None] = []
        y_list: list[float | None] = []
        z_list: list[float | None] = []

        node_deformations = deformation_configuration.data
        amplification_factor = deformation_configuration.amplification_factor
        assert amplification_factor is not None
        step = deformation_configuration.step
        for element in elements:
            if element.visibility.hidden_at_line_plots:
                continue

            p_i = np.array(element.nodes[0].coordinates)
            p_io = (
                np.array(element.nodes[0].coordinates) + element.geomtransf.offset_i
            )
            p_j = np.array(element.nodes[1].coordinates)
            p_jo = (
                np.array(element.nodes[1].coordinates) + element.geomtransf.offset_j
            )

            node_i = element.nodes[0].uid
            node_j = element.nodes[1].uid
            for node in (node_i, node_j):
                if node not in node_deformations.columns:
                    msg = 'Results not available for node: {node.uid}.'
                    raise ValueError(msg)

            if self.configuration.ndm == THREE_DIMENSIONAL:
                u_i = node_deformations.iloc[step][node_i].to_numpy()[0:3]
                r_i = node_deformations.iloc[step][node_i].to_numpy()[3:6]
                u_j = node_deformations.iloc[step][node_j].to_numpy()[0:3]
                r_j = node_deformations.iloc[step][node_j].to_numpy()[3:6]
                offset_i = element.geomtransf.offset_i
                offset_j = element.geomtransf.offset_j
                u_i_o = offset_transformation_3d(offset_i, u_i, r_i)
                u_j_o = offset_transformation_3d(offset_j, u_j, r_j)
            elif self.configuration.ndm == TWO_DIMENSIONAL:
                u_i = node_deformations.iloc[step][node_i].to_numpy()[0:2]
                r_i = node_deformations.iloc[step][node_i].to_numpy()[2]
                u_j = node_deformations.iloc[step][node_j].to_numpy()[0:2]
                r_j = node_deformations.iloc[step][node_j].to_numpy()[2]
                offset_i = element.geomtransf.offset_i
                offset_j = element.geomtransf.offset_j
                u_i_o = offset_transformation_2d(offset_i, u_i, r_i)
                u_j_o = offset_transformation_2d(offset_j, u_j, r_j)

                u_i = np.insert(u_i, 1, 0.00)
                u_j = np.insert(u_j, 1, 0.00)
                u_i_o = np.insert(u_i_o, 1, 0.00)
                u_j_o = np.insert(u_j_o, 1, 0.00)
                p_i = np.insert(p_i, 1, 0.00)
                p_io = np.insert(p_io, 1, 0.00)
                p_j = np.insert(p_j, 1, 0.00)
                p_jo = np.insert(p_jo, 1, 0.00)

            else:
                msg = 'Unsupported NDM: {self.configuration.ndm}.'
                raise ValueError(msg)

            x_list.extend(
                (
                    p_i[0] + u_i[0] * amplification_factor,
                    p_io[0] + u_i_o[0] * amplification_factor,
                    None,
                    p_j[0] + u_j[0] * amplification_factor,
                    p_jo[0] + u_j_o[0] * amplification_factor,
                    None,
                )
            )
            y_list.extend(
                (
                    p_i[1] + u_i[1] * amplification_factor,
                    p_io[1] + u_i_o[1] * amplification_factor,
                    None,
                    p_j[1] + u_j[1] * amplification_factor,
                    p_jo[1] + u_j_o[1] * amplification_factor,
                    None,
                )
            )
            z_list.extend(
                (
                    p_i[2] + u_i[2] * amplification_factor,
                    p_io[2] + u_i_o[2] * amplification_factor,
                    None,
                    p_j[2] + u_j[2] * amplification_factor,
                    p_jo[2] + u_j_o[2] * amplification_factor,
                    None,
                )
            )

        data['x'].extend(x_list)  # type: ignore
        data['y'].extend(y_list)  # type: ignore
        data['z'].extend(z_list)  # type: ignore

    @staticmethod
    def _interpolate_deformation_2d(
        element: BeamColumnElement,
        u_i: numpy_array,
        r_i: numpy_array,
        u_j: numpy_array,
        r_j: numpy_array,
        num_points: int,
    ) -> tuple[numpy_array, numpy_array]:
        """
        Interpolate deformation in 2D.

        Given the deformations of the ends of a beamcolumn element,
        use its shape functions to obtain intermediate points.
        Assumes sections remain orthogonal to the deformed centroidal
        axis and ignores applied loads. The primary intent of this
        method is to estimate the deformation for visualization
        purposes.

        Args:
          element: A line element
          u_i: 2 displacements at end i, global system
          r_i: rotation at end i
          u_j: 2 displacements at end j, global system
          r_j: rotation at end j
          num_points: Number of interpolation points

        Returns:
          Displacements (global system) and rotations. The rotations
            are needed for plotting the deformed shape with extruded
            frame elements.

        """
        x_vec: numpy_array = element.geomtransf.x_axis
        y_vec = element.geomtransf.y_axis
        assert y_vec is None
        z_vec: numpy_array = element.geomtransf.z_axis

        # global -> local transformation matrix
        transf_global2local = np.vstack((x_vec, z_vec))
        transf_local2global = transf_global2local.T

        u_i_global = u_i
        u_j_global = u_j
        r_i_global = r_i
        r_j_global = r_j

        u_i_local = transf_global2local @ u_i_global
        u_j_local = transf_global2local @ u_j_global
        r_i_local = r_i_global
        r_j_local = r_j_global

        # discrete sample location parameter
        t_vec = np.linspace(0.00, 1.00, num=num_points)
        p_i = np.array(element.nodes[0].coordinates) + element.geomtransf.offset_i
        p_j = np.array(element.nodes[1].coordinates) + element.geomtransf.offset_j
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
            [u_i_local[1], r_i_local, u_j_local[1], r_j_local]
        )

        # bending rotation around the local z axis
        r_z_local = (
            nyz_derivative_mat
            @ np.array([u_i_local[1], r_i_local, u_j_local[1], r_j_local])
            / len_clr
        )

        # all deformations
        d_local = np.column_stack((d_x_local, d_y_local))

        # all rotations
        r_local = r_z_local

        d_global = (transf_local2global @ d_local.T).T

        return d_global, r_local

    @staticmethod
    def _interpolate_deformation_3d(
        element: BeamColumnElement,
        u_i: numpy_array,
        r_i: numpy_array,
        u_j: numpy_array,
        r_j: numpy_array,
        num_points: int,
    ) -> tuple[numpy_array, numpy_array]:
        """
        Interpolate deformation in 3D.

        Given the deformations of the ends of a beamcolumn element,
        use its shape functions to obtain intermediate points.
        Assumes sections remain orthogonal to the deformed centroidal
        axis and ignores applied loads. The primary intent of this
        method is to estimate the deformation for visualization
        purposes.

        Args:
          element: A line element
          u_i: 3 displacements at end i, global system
          r_i: 3 rotations at end i, global system
          u_j: 3 displacements at end j, global system
          r_j: 3 rotations at end j, global system
          num_points: Number of interpolation points

        Returns:
          Displacements (global system) and rotations (local system). The
            rotations are needed for plotting the deformed shape with
            extruded frame elements.

        """
        x_vec = element.geomtransf.x_axis
        y_vec = element.geomtransf.y_axis
        assert y_vec is not None
        z_vec = np.cross(x_vec, y_vec)

        # global -> local transformation matrix
        transf_global2local = transformation_matrix(x_vec, y_vec, z_vec)
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
        p_i = np.array(element.nodes[0].coordinates) + element.geomtransf.offset_i
        p_j = np.array(element.nodes[1].coordinates) + element.geomtransf.offset_j
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

    @staticmethod
    def _interpolate_points(
        element: BeamColumnElement,
        d_global: numpy_array,
        num_points: int,
        scaling: float,
    ) -> numpy_array:
        """
        Interpolate intermediate points.

        Calculates intermediate points based on end locations and
        deformations.

        Returns:
          The interpolated points.
        """
        p_i = np.array(element.nodes[0].coordinates) + element.geomtransf.offset_i
        p_j = np.array(element.nodes[1].coordinates) + element.geomtransf.offset_j

        assert len(p_j) == len(p_i)

        columns = [
            np.linspace(p_i[i], p_j[i], num=num_points) for i in range(len(p_i))
        ]
        element_point_samples: numpy_array = np.column_stack(columns)

        return element_point_samples + d_global * scaling

    @staticmethod
    def _generate_pyramid_mesh(
        tip_coordinates: tuple[float, float, float],
        height: float,
        direction: Literal['x', 'y', 'z', '-x', '-y', '-z'],
        index_offset: int,
    ) -> tuple[
        list[float], list[float], list[float], list[int], list[int], list[int]
    ]:
        """
        Define data for a scaled and translated pyramid mesh.

        Arguments:
          tip_coordinates: Coordinates of the tip of the pyramid.
          height: The height of the pyramid (scales its size proportionally).
          direction: The global axis direction to which the pyramid
            will extend from tip to base.
          index_offset: Offset the index of the vertices in the data
            dictionary to define additional pyramids.

        Returns:
          The coordinates and sequence of vertices in 3D space.

        Raises:
          ValueError: If an invalid direction is specified
        """
        # Tip coordinates
        tip_x, tip_y, tip_z = tip_coordinates

        # Base dimensions relative to the tip
        base_size = height / 2

        if direction in {'x', '-x'}:
            offset = height if direction == 'x' else -height
            vertices_x = [
                tip_x,
                tip_x + offset,
                tip_x + offset,
                tip_x + offset,
                tip_x + offset,
            ]
            vertices_y = [
                tip_y,
                tip_y + base_size,
                tip_y - base_size,
                tip_y - base_size,
                tip_y + base_size,
            ]
            vertices_z = [
                tip_z,
                tip_z - base_size,
                tip_z - base_size,
                tip_z + base_size,
                tip_z + base_size,
            ]
        elif direction in {'y', '-y'}:
            offset = height if direction == 'y' else -height
            vertices_x = [
                tip_x,
                tip_x - base_size,
                tip_x + base_size,
                tip_x + base_size,
                tip_x - base_size,
            ]
            vertices_y = [
                tip_y,
                tip_y + offset,
                tip_y + offset,
                tip_y + offset,
                tip_y + offset,
            ]
            vertices_z = [
                tip_z,
                tip_z - base_size,
                tip_z - base_size,
                tip_z + base_size,
                tip_z + base_size,
            ]
        elif direction in {'z', '-z'}:
            offset = height if direction == 'z' else -height
            vertices_x = [
                tip_x,
                tip_x - base_size,
                tip_x + base_size,
                tip_x + base_size,
                tip_x - base_size,
            ]
            vertices_y = [
                tip_y,
                tip_y - base_size,
                tip_y - base_size,
                tip_y + base_size,
                tip_y + base_size,
            ]
            vertices_z = [
                tip_z,
                tip_z + offset,
                tip_z + offset,
                tip_z + offset,
                tip_z + offset,
            ]
        else:
            msg = "Invalid direction. Choose 'x', '-x', 'y', '-y', 'z', or '-z'."
            raise ValueError(msg)

        # Define faces using vertex indices
        i_o = index_offset
        faces_i = [
            0 + i_o,
            0 + i_o,
            0 + i_o,
            0 + i_o,
        ]  # Start vertex index for each triangle
        faces_j = [
            1 + i_o,
            2 + i_o,
            3 + i_o,
            4 + i_o,
        ]  # Middle vertex index for each triangle
        faces_k = [
            2 + i_o,
            3 + i_o,
            4 + i_o,
            1 + i_o,
        ]  # End vertex index for each triangle

        # Add the base face
        faces_i.extend([1 + i_o, 2 + i_o])
        faces_j.extend([2 + i_o, 3 + i_o])
        faces_k.extend([4 + i_o, 4 + i_o])

        return vertices_x, vertices_y, vertices_z, faces_i, faces_j, faces_k

    def _generate_arrow(
        self,
        start_location: tuple[float, float, float],
        end_location: tuple[float, float, float],
        head_length: float,
        head_width: float,
        base_width: float,
        name: str,
        value: str,
    ) -> None:
        vertices, faces = positioned_arrow(
            start_location,
            end_location,
            head_length,
            head_width,
            base_width,
        )
        x, y, z = zip(*vertices)

        data = self.find_data_by_name(name)
        if not data:
            # Initialize the mesh3d data structure
            data = {
                'name': name,
                'type': 'mesh3d',
                'x': [],
                'y': [],
                'z': [],
                'i': [],
                'j': [],
                'k': [],
                'text': [],  # For custom hover information
                'hovertemplate': ('Value: %{text}<br>' '<extra></extra>'),
                'color': '#7ac4b7',
                'opacity': 0.5,
                'showlegend': True,
            }
            self.data.append(data)

        # Add vertices
        data['x'].extend(x)  # type: ignore
        data['y'].extend(y)  # type: ignore
        data['z'].extend(z)  # type: ignore

        # Add hover text for each vertex
        data['text'].extend([value] * 13)  # type: ignore

        # Define two triangular faces of the quadrilateral
        # Faces are defined by indices of vertices in the mesh
        index_offset = len(data['x']) - 13  # type: ignore
        faces = tuple(np.array(faces) + index_offset)
        i, j, k = zip(*faces)
        data['i'].extend(i)  # type: ignore
        data['j'].extend(j)  # type: ignore
        data['k'].extend(k)  # type: ignore

    def _generate_quadrilateral_mesh(
        self,
        four_points: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
        name: str,
        value: tuple[str, str, str, str],
        color: str,
        *,
        opacity: float,
        start_hidden: bool = False,
    ) -> None:
        """
        Generate and add a filled quadrilateral to the figure.

        Arguments:
            four_points: A tuple of four vertices defining the quadrilateral.
                         Each vertex is a tuple (x, y, z).
            name: Name of the quadrilateral to retrieve or store in the figure data.
            value: Values to display in the hoverbox at each of the four corners.
            color: Desired mesh color.
            opacity: Desired mesh opacity.
            start_hidden: Have it hidden when the plot is shown, to
              activate through the legend if desired.
        """
        data = self.find_data_by_name(name)
        if not data:
            # Initialize the mesh3d data structure
            data = {
                'name': name,
                'type': 'mesh3d',
                'x': [],
                'y': [],
                'z': [],
                'i': [],
                'j': [],
                'k': [],
                'text': [],  # For custom hover information
                'hovertemplate': 'Value: %{text}<br><extra></extra>',
                'color': color,
                'opacity': opacity,
                'showlegend': True,
            }
            if start_hidden:
                data['visible'] = 'legendonly'
            self.data.append(data)

        # Extract the vertices
        (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4) = four_points

        # Add vertices
        data['x'].extend([x1, x2, x3, x4])  # type: ignore
        data['y'].extend([y1, y2, y3, y4])  # type: ignore
        data['z'].extend([z1, z2, z3, z4])  # type: ignore

        # Add hover text for each vertex
        data['text'].extend(value)  # type: ignore

        # Define two triangular faces of the quadrilateral
        # Faces are defined by indices of vertices in the mesh
        index_offset = len(data['x']) - 4  # type: ignore
        data['i'].extend([index_offset, index_offset])  # type: ignore
        data['j'].extend([index_offset + 1, index_offset + 2])  # type: ignore
        data['k'].extend([index_offset + 2, index_offset + 3])  # type: ignore

    def show(self) -> None:
        """Display the figure."""
        fig = go.Figure({'data': self.data, 'layout': self.layout})
        fig.show()
