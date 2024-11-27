"""Create plotly graphics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import plotly.graph_objects as go  # type: ignore

from osmg.core.osmg_collections import BeamColumnAssembly
from osmg.graphics.objects import positioned_arrow
from osmg.model_objects.element import (
    BeamColumnElement,
    DispBeamColumn,
    ElasticBeamColumn,
)

if TYPE_CHECKING:
    from osmg.analysis.common import UDL, PointLoad
    from osmg.analysis.recorders import NodeRecorder
    from osmg.analysis.supports import ElasticSupport, FixedSupport
    from osmg.core.osmg_collections import ComponentAssembly
    from osmg.model_objects.element import Element
    from osmg.model_objects.node import Node


def _default_camera() -> dict[str, object]:
    return {
        'up': {'x': 0, 'y': 0, 'z': 1},
        'center': {'x': 0, 'y': 0, 'z': 0},
        'eye': {'x': 0.00, 'y': -10.00, 'z': 0.00},
        'projection': {'type': 'perspective'},
    }


@dataclass
class Figure3DConfiguration:
    """Configuration for 3D figures."""

    camera: dict[str, object] = field(default_factory=_default_camera)
    ndm: Literal[2, 3] = field(default=3)


@dataclass
class DeformationConfiguration:
    """Configuration for 3D figures."""

    reference_length: float
    ndf: int
    recorder: NodeRecorder
    step: int
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
        node_deformations = self.recorder.get_data()
        if node_deformations.shape[0] < self.step:
            msg = (
                f'The requested step ({self.step}) does not exist. '
                f'The last step is ({node_deformations.shape[0] - 1})'
            )
            raise ValueError(msg)
        case_2d = 3
        case_3d = 6
        max_deformation = {}
        for i_dof in range(self.ndf):
            max_deformation[i_dof] = (
                node_deformations.iloc[self.step, i_dof :: self.ndf].abs().max()
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
            msg = 'Invalid model NDF: {model.ndf}.'
            raise ValueError(msg)
        # 10% of reference length, or at most an approx. 30 degree rotation angle.
        self.amplification_factor = min(
            self.reference_length / max_displacement * 0.10, max_rotation / 0.50
        )


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
        as_overlay: bool = False,
    ) -> None:
        """Draw nodes."""
        if deformation_configuration is None:
            self._add_nodes_undeformed(nodes, designation, as_overlay=as_overlay)
        else:
            self._add_nodes_deformed(
                nodes, designation, deformation_configuration, as_overlay=as_overlay
            )

    def add_components(
        self,
        components: list[ComponentAssembly],
        deformation_configuration: DeformationConfiguration | None = None,
        *,
        as_overlay: bool = False,
    ) -> None:
        """Add components to the figure."""
        for component in components:
            internal_nodes = list(component.internal_nodes.values())
            self.add_nodes(
                internal_nodes,
                'internal',
                deformation_configuration,
                as_overlay=as_overlay,
            )
            elements = list(component.elements.values())
            self.add_elements(
                elements, deformation_configuration, as_overlay=as_overlay
            )

    def add_elements(
        self,
        elements: list[Element],
        deformation_configuration: DeformationConfiguration | None = None,
        *,
        as_overlay: bool = False,
    ) -> None:
        """Add elements to the figure."""
        elastic_beamcolumn_elements: list[ElasticBeamColumn] = []
        disp_beamcolumn_elements: list[DispBeamColumn] = []
        unknown_types: set[str] = set()
        for element in elements:
            if isinstance(element, ElasticBeamColumn):
                elastic_beamcolumn_elements.append(element)
            elif isinstance(element, DispBeamColumn):
                disp_beamcolumn_elements.append(element)
            elif element.__class__.__name__ not in unknown_types:
                unknown_types = unknown_types.union({element.__class__.__name__})
        if unknown_types:
            print(  # noqa: T201
                f'WARNING: Skipped the following unknown element types: {unknown_types}.'
            )
            # TODO(JVM): implement warning
        if deformation_configuration is None:
            self.add_beamcolumn_elements(
                elastic_beamcolumn_elements,
                as_overlay=as_overlay,
            )
            self.add_beamcolumn_element_offsets(
                elastic_beamcolumn_elements,
                as_overlay=as_overlay,
            )

    def add_beamcolumn_element_offsets(
        self,
        elements: list[ElasticBeamColumn] | list[DispBeamColumn],
        *,
        as_overlay: bool = False,
    ) -> None:
        """Add beamcolumn elements to the figure."""
        if as_overlay:
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
                'defined': set(),
            }
            self.data.append(data)

        x_list: list[float | None] = []
        y_list: list[float | None] = []
        z_list: list[float | None] = []
        for element in elements:
            if (
                element.visibility.hidden_at_line_plots
                or element.uid in data['defined']  # type: ignore
            ):
                continue
            p_i = np.array(element.nodes[0].coordinates)
            p_io = (
                np.array(element.nodes[0].coordinates) + element.geomtransf.offset_i
            )
            p_j = np.array(element.nodes[1].coordinates)
            p_jo = (
                np.array(element.nodes[1].coordinates) + element.geomtransf.offset_j
            )

            three_dimensional = 3
            two_dimensional = 2
            if self.configuration.ndm == three_dimensional:
                x_list.extend((p_i[0], p_io[0], None))
                y_list.extend((p_i[1], p_io[1], None))
                z_list.extend((p_i[2], p_io[2], None))
                x_list.extend((p_j[0], p_jo[0], None))
                y_list.extend((p_j[1], p_jo[1], None))
                z_list.extend((p_j[2], p_jo[2], None))
            if self.configuration.ndm == two_dimensional:
                x_list.extend((p_i[0], p_io[0], None))
                y_list.extend((0.0, 0.0, None))
                z_list.extend((p_i[1], p_io[1], None))
                x_list.extend((p_j[0], p_jo[0], None))
                y_list.extend((0.0, 0.0, None))
                z_list.extend((p_j[1], p_jo[1], None))

        data['x'].extend(x_list)  # type: ignore
        data['y'].extend(y_list)  # type: ignore
        data['z'].extend(z_list)  # type: ignore

        data['defined'] = data['defined'].union(x.uid for x in elements)  # type: ignore

    def add_beamcolumn_elements(
        self,
        elements: list[ElasticBeamColumn] | list[DispBeamColumn],
        *,
        as_overlay: bool = False,
    ) -> None:
        """Add beamcolumn elements to the figure."""
        if as_overlay:
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
                'defined': set(),
            }
            self.data.append(data)

        x_list: list[float | None] = []
        y_list: list[float | None] = []
        z_list: list[float | None] = []
        customdata_list = []  # type: ignore
        section_names = []  # type: ignore
        for element in elements:
            if (
                element.visibility.hidden_at_line_plots
                or element.uid in data['defined']  # type: ignore
            ):
                continue
            p_i = (
                np.array(element.nodes[0].coordinates) + element.geomtransf.offset_i
            )
            p_j = (
                np.array(element.nodes[1].coordinates) + element.geomtransf.offset_j
            )
            section_name = element.section.name
            section_names.extend([section_name] * 3)
            three_dimensional = 3
            two_dimensional = 2
            if self.configuration.ndm == three_dimensional:
                x_list.extend((p_i[0], p_j[0], None))
                y_list.extend((p_i[1], p_j[1], None))
                z_list.extend((p_i[2], p_j[2], None))
            else:
                assert self.configuration.ndm == two_dimensional
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

        data['defined'] = data['defined'].union(x.uid for x in elements)  # type: ignore

    def add_supports(  # noqa: C901  kalamari
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

        three_dimensional = 3
        two_dimensional = 2
        if num_dimensions == three_dimensional:
            directions: dict[int, Literal['x', 'y', 'z', '-x', '-y', '-z']] = {
                0: '-x',
                1: '-y',
                2: '-z',
                3: 'x',
                4: 'y',
                5: 'z',
            }
        else:
            assert num_dimensions == two_dimensional
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
                'defined': set(),
                'showlegend': True,
            }
            # TODO (JVM): replace template fields with actual info.
            self.data.append(data)

        index_offset = 0
        for uid, support in supports.items():
            node = nodes[uid]
            if uid in data['defined']:  # type: ignore
                continue
            if self.configuration.ndm == three_dimensional:
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
                assert self.configuration.ndm == two_dimensional
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
        # Update defined objects.
        data['defined'] = data['defined'].union(supports.keys())  # type: ignore

    def add_udl(
        self,
        udl: dict[int, UDL],
        components: dict[int, ComponentAssembly],
        force_to_length_factor: float,
        offset: float,
    ) -> None:
        """Show uniformly distributed load applied on the components."""
        two_dimensional = 2
        three_dimensional = 3
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

                if len(start_vec) == two_dimensional:
                    # Add 0.00 to the Y axis (second element)
                    start_vec = np.insert(start_vec, 1, 0.00)
                    end_vec = np.insert(end_vec, 1, 0.00)
                    udl_vec = np.insert(udl_vec, 1, 0.00)

                else:
                    assert len(start_vec) == three_dimensional

                udl_vec = -udl_vec  # Reverse direction
                udl_vec_normalized = udl_vec / np.linalg.norm(udl_vec)
                start_vertex = start_vec + udl_vec_normalized * offset
                end_vertex = end_vec + udl_vec_normalized * offset
                start_vertex_top = start_vertex + udl_vec * force_to_length_factor
                end_vertex_top = end_vertex + udl_vec * force_to_length_factor

                self._generate_filled_quadrilateral(
                    (
                        tuple(start_vertex),
                        tuple(end_vertex),
                        tuple(end_vertex_top),
                        tuple(start_vertex_top),
                    ),
                    name='Uniformly Distributed Loads',
                    value=str(global_udl),
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
        two_dimensional = 2
        three_dimensional = 3
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

            if len(start_vec) == two_dimensional:
                # Add 0.00 to the Y axis (second element)
                start_vec = np.insert(start_vec, 1, 0.00)
                end_vec = np.insert(end_vec, 1, 0.00)
            else:
                assert len(start_vec) == three_dimensional

            self._generate_arrow(
                tuple(start_vec),
                tuple(end_vec),
                head_length,
                head_width,
                base_width,
                name='Nodal Loads',
                value=str(point_load),
            )

    def _add_nodes_undeformed(
        self,
        nodes: list[Node],
        designation: Literal['primary', 'parent', 'internal'],
        *,
        as_overlay: bool = False,
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
        if as_overlay:
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
                'defined': set(),
            }
            self.data.append(data)
        for node in nodes:
            if node.uid in data['defined']:  # type: ignore
                continue
            three_dimensional = 3
            two_dimensional = 2
            if self.configuration.ndm == three_dimensional:
                data['x'].append(node.coordinates[0])  # type: ignore
                data['y'].append(node.coordinates[1])  # type: ignore
                data['z'].append(node.coordinates[2])  # type: ignore
                data['customdata'].append([node.uid, None])  # type: ignore
            else:
                assert self.configuration.ndm == two_dimensional
                data['x'].append(node.coordinates[0])  # type: ignore
                data['y'].append(0.00)  # type: ignore
                data['z'].append(node.coordinates[1])  # type: ignore
                data['customdata'].append([node.uid, None])  # type: ignore
        # Update defined objects.
        data['defined'] = data['defined'].union(x.uid for x in nodes)  # type: ignore

    def _add_nodes_deformed(
        self,
        nodes: list[Node],
        designation: Literal['primary', 'parent', 'internal'],
        deformation_configuration: DeformationConfiguration,
        *,
        as_overlay: bool = False,
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
        if as_overlay:
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
                'defined': set(),
            }
            self.data.append(data)
        for node in nodes:
            if node.uid in data['defined']:  # type: ignore
                continue
            if node.uid not in deformation_configuration.recorder.get_data().columns:
                msg = 'Results not available for node: {node.uid}.'
                raise ValueError(msg)
            three_dimensional = 3
            two_dimensional = 2
            if self.configuration.ndm == three_dimensional:
                deformations = deformation_configuration.recorder.get_data()
                node_deformations = deformations.iloc[
                    deformation_configuration.step, :
                ].loc[node.uid]
                data['x'].append(  # type: ignore
                    node.coordinates[0]
                    + node_deformations[0]
                    * deformation_configuration.amplification_factor
                )
                data['y'].append(  # type: ignore
                    node.coordinates[0]
                    + node_deformations[1]
                    * deformation_configuration.amplification_factor
                )
                data['z'].append(  # type: ignore
                    node.coordinates[0]
                    + node_deformations[2]
                    * deformation_configuration.amplification_factor
                )
                data['customdata'].append(  # type: ignore
                    [node.uid, [f'{v:.2f}' for v in node_deformations]]
                )
            else:
                assert self.configuration.ndm == two_dimensional
                deformations = deformation_configuration.recorder.get_data()
                node_deformations = (
                    deformations.iloc[deformation_configuration.step, :]
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
        # Update defined objects.
        data['defined'] = data['defined'].union(x.uid for x in nodes)  # type: ignore

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

    def _generate_filled_quadrilateral(
        self,
        four_points: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
        name: str,
        value: str,
    ) -> None:
        """
        Generate and add a filled quadrilateral to the figure.

        Arguments:
            four_points: A tuple of four vertices defining the quadrilateral.
                         Each vertex is a tuple (x, y, z).
            name: Name of the quadrilateral to retrieve or store in the figure data.
            value: A value to display in the hover box.
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
                'hovertemplate': ('Value: %{text}<br>' '<extra></extra>'),
                'color': '#7ac4b7',
                'opacity': 0.5,
                'showlegend': True,
                'defined': [],
            }
            self.data.append(data)

        # Extract the vertices
        (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4) = four_points

        # Add vertices
        data['x'].extend([x1, x2, x3, x4])  # type: ignore
        data['y'].extend([y1, y2, y3, y4])  # type: ignore
        data['z'].extend([z1, z2, z3, z4])  # type: ignore

        # Add hover text for each vertex
        data['text'].extend([value] * 4)  # type: ignore

        # Define two triangular faces of the quadrilateral
        # Faces are defined by indices of vertices in the mesh
        index_offset = len(data['x']) - 4  # type: ignore
        data['i'].extend([index_offset, index_offset])  # type: ignore
        data['j'].extend([index_offset + 1, index_offset + 2])  # type: ignore
        data['k'].extend([index_offset + 2, index_offset + 3])  # type: ignore

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
                'defined': [],
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

    def show(self) -> None:
        """Display the figure."""
        # Remove `defined` entry from `data`
        for data in self.data:
            data.pop('defined')
        fig = go.Figure({'data': self.data, 'layout': self.layout})
        fig.show()
