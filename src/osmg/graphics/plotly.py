"""Create plotly graphics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import plotly.graph_objects as go  # type: ignore

from osmg.elements.element import DispBeamColumn, ElasticBeamColumn

if TYPE_CHECKING:
    from osmg.core.component_assemblies import ComponentAssembly
    from osmg.elements.element import Element
    from osmg.elements.node import Node


def _default_camera() -> dict[str, object]:
    return {
        'up': {'x': 0, 'y': 0, 'z': 1},
        'center': {'x': 0, 'y': 0, 'z': 0},
        'eye': {'x': 0.7, 'y': 1.00, 'z': 0.30},
        'projection': {'type': 'perspective'},
    }


@dataclass(repr=False)
class Figure3DConfiguration:
    """Configuration for 3D figures."""

    camera: dict[str, object] = field(default_factory=_default_camera)
    num_space_dimensions: Literal[2, 3] = field(default=3)


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
                'aspectratio': {'x': 0.25, 'y': 0.25, 'z': 0.25},
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
                    'Coordinates: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>'
                    'Text field: %{text}<br>'
                    'Node ID: %{customdata[0]:d}<br>'
                    'Custom Field 1: %{customdata[1]}<br>'
                    'Custom Field 2: %{customdata[2]}<br>'
                    '<extra></extra>'
                ),
                'marker': {
                    'symbol': marker[designation][0],
                    'color': marker[designation][2],
                    'size': marker[designation][1],
                    'line': {
                        'color': marker[designation][2],
                        'width': 4,
                    },
                },
                'defined': set(),
            }
            # TODO (JVM): replace template fields with actual info.
            self.data.append(data)
        for node in nodes:
            if node.uid in data['defined']:  # type: ignore
                continue
            three_dimensional = 3
            two_dimensional = 2
            if self.configuration.num_space_dimensions == three_dimensional:
                data['x'].append(node.coordinates[0])  # type: ignore
                data['y'].append(node.coordinates[1])  # type: ignore
                data['z'].append(node.coordinates[2])  # type: ignore
            else:
                assert self.configuration.num_space_dimensions == two_dimensional
                data['x'].append(node.coordinates[0])  # type: ignore
                data['y'].append(0.00)  # type: ignore
                data['z'].append(node.coordinates[1])  # type: ignore
            data['customdata'].append([node.uid, 'field2', 'field3'])  # type: ignore
            data['text'].append('field_text')  # type: ignore
        # Update defined objects.
        data['defined'] = data['defined'].union(x.uid for x in nodes)  # type: ignore

    def add_components(self, components: list[ComponentAssembly]) -> None:
        """Add components to the figure."""
        for component in components:
            internal_nodes = list(component.internal_nodes.values())
            self.add_nodes(internal_nodes, 'internal')
            elements = list(component.elements.values())
            self.add_elements(elements)

    def add_elements(self, elements: list[Element]) -> None:
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
        self.add_beamcolumn_elements(elastic_beamcolumn_elements)
        self.add_beamcolumn_element_offsets(elastic_beamcolumn_elements)

    def add_beamcolumn_element_offsets(
        self, elements: list[ElasticBeamColumn] | list[DispBeamColumn]
    ) -> None:
        """Add beamcolumn elements to the figure."""
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
            if self.configuration.num_space_dimensions == three_dimensional:
                x_list.extend((p_i[0], p_io[0], None))
                y_list.extend((p_i[1], p_io[1], None))
                z_list.extend((p_i[2], p_io[2], None))
                x_list.extend((p_j[0], p_jo[0], None))
                y_list.extend((p_j[1], p_jo[1], None))
                z_list.extend((p_j[2], p_jo[2], None))
            if self.configuration.num_space_dimensions == two_dimensional:
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
        self, elements: list[ElasticBeamColumn] | list[DispBeamColumn]
    ) -> None:
        """Add beamcolumn elements to the figure."""
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
            if self.configuration.num_space_dimensions == three_dimensional:
                x_list.extend((p_i[0], p_j[0], None))
                y_list.extend((p_i[1], p_j[1], None))
                z_list.extend((p_i[2], p_j[2], None))
            else:
                assert self.configuration.num_space_dimensions == two_dimensional
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

    def show(self) -> None:
        """Display the figure."""
        # Remove `defined` entry from `data`
        for data in self.data:
            data.pop('defined')
        fig = go.Figure({'data': self.data, 'layout': self.layout})
        fig.show()
