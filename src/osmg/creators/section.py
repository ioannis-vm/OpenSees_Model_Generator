"""objects that create sections."""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from osmg.geometry.mesh_shapes import w_mesh
from osmg.model_objects.section import ElasticSection

if TYPE_CHECKING:
    from osmg.creators.uid import UIDGenerator


# Base class for sections
class BaseSectionData(BaseModel):
    """Base model for section data."""

    Type: str
    A: float
    Ix: float
    Iy: float
    J: float
    W: float


# W Section model
class WSectionData(BaseSectionData):
    """Model for W sections."""

    bf: float
    d: float
    tw: float
    tf: float
    Zx: float
    Zy: float
    rx: float
    ry: float


# HSS Section model
class HSSSectionData(BaseSectionData):
    """Model for HSS sections."""

    t: float
    D: float
    B: float


# Union for supported section types
SectionData = WSectionData | HSSSectionData


class SectionDatabase(BaseModel):
    """Model for the entire section database."""

    sections: dict[str, SectionData]


class AISC_Database_Section_Creator:
    """
    Create frame member sections from a predefined database.

    Force units are `lb` and length units are `in`.
    """

    def __init__(
        self, uid_generator: UIDGenerator, database_path: str | None = None
    ) -> None:
        """Instantiate object."""
        self.uid_generator = uid_generator
        self.database_path = database_path
        self._load_database()

    def _load_database(self) -> None:
        """Load and validate the JSON database."""
        if self.database_path is None:
            with importlib.resources.open_text('osmg.data', 'sections.json') as f:
                raw_data = f.read()
        else:
            with Path(self.database_path).open('r', encoding='utf-8') as f:
                raw_data = f.read()

        json_data = json.loads(raw_data)
        valid_sections: dict[str, SectionData] = {}
        unsupported_types = []
        for name, section in json_data.items():
            section_type = section['Type']
            if section_type == 'W':
                valid_sections[name] = WSectionData(**section)
            # elif section_type == 'HSS':
            #     valid_sections[name] = HSSSectionData(**section)
            else:
                unsupported_types.append(section_type)
        if unsupported_types:
            print(f'Skipping unsupported section types: {set(unsupported_types)}')  # noqa: T201

        self.section_database = SectionDatabase(sections=valid_sections)

    def get_available(self) -> list[str]:
        """
        Determine which sections are available in the database.

        Returns:
          List of available sections.
        """
        return list(self.section_database.sections.keys())

    def load_elastic_section(
        self, section_label: str, e_modulus: float, g_modulus: float
    ) -> ElasticSection:
        """
        Load a section from the database.

        Args:
            section_label: Label of the section to load.
            e_modulus: Elastic modulus.
            g_modulus: Shear modulus.

        Returns:
            ElasticSection: The section.

        Raises:
            ValueError: If the provided `section_label` is unavailable.
            NotImplementedError: If the section type is unsupported.
        """
        section_data = self.section_database.sections.get(section_label)

        if section_data is None:
            msg = f'The provided `section_label` is unavailable: {section_label}.'
            raise ValueError(msg)

        if isinstance(section_data, WSectionData):
            # Geometry calculations for W sections
            outside_shape = w_mesh(
                section_data.bf,
                section_data.d,
                section_data.tw,
                section_data.tf,
                section_data.A,
            )
            bbox = outside_shape.bounding_box()
            z_min, y_min, z_max, y_max = bbox.flatten()
            snap_points = {
                'centroid': np.array([0.0, 0.0]),
                'top_center': np.array([0.0, -y_max]),
                'top_left': np.array([-z_min, -y_max]),
                'top_right': np.array([-z_max, -y_max]),
                'center_left': np.array([-z_min, 0.0]),
                'center_right': np.array([-z_max, 0.0]),
                'bottom_center': np.array([0.0, -y_min]),
                'bottom_left': np.array([-z_min, -y_min]),
                'bottom_right': np.array([-z_max, -y_min]),
            }

            return ElasticSection(
                self.uid_generator,
                section_label,
                e_modulus,
                section_data.A,
                section_data.Iy,
                section_data.Ix,
                g_modulus,
                section_data.J,
                section_data.W / 12.00,  # lb/in
                outside_shape,
                snap_points,
                properties=section_data,
            )
        msg = f'Section type is unsupported: {section_data.Type}'
        raise NotImplementedError(msg)
