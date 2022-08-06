"""
Model Generator for OpenSees ~ section generator
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
from dataclasses import dataclass
import json
import pkgutil
import numpy as np
import numpy.typing as npt
from ..ops.section import SectionComponent
from ..ops.section import ElasticSection
from ..ops.section import FiberSection
from ..gen import mesh_shapes
from .mesh_shapes import rect_mesh
if TYPE_CHECKING:
    from ..model import Model
    from ..ops.section import Section

nparr = npt.NDArray[np.float64]


@dataclass(repr=False)
class SectionGenerator:
    """
    Used to populate the section repository of a model.
    """
    model: Model

    def generate_generic_elastic(
            self,
            name: str,
            e_times_a: float,
            e_times_i: float,
            g_times_j: float
    ):
        """
        Generates a generic elastic section with the specified
        properties
        """
        sec = ElasticSection(
            name=name,
            uid=self.model.uid_generator.new('section'),
            outside_shape=None,
            snap_points=None,
            e_mod=1.00,
            area=e_times_a,
            i_y=e_times_i,
            i_x=e_times_i,
            g_mod=1.00,
            j_mod=g_times_j,
            sec_w=0.00,
        )
        if self.model.settings.imperial_units:
            y_max = +10.00
            y_min = -10.00
            z_max = +6.00
            z_min = -6.00
            sec.outside_shape = rect_mesh(12.0, 20.0)
        else:
            y_max = +0.25
            y_min = -0.25
            z_max = +0.15
            z_min = -0.15
            sec.outside_shape = rect_mesh(0.30, 0.50)
        snap_points: dict[str, nparr] = {
            'centroid': np.array([0., 0.]),
            'top_center': np.array([0., -y_max]),
            'top_left': np.array([-z_min, -y_max]),
            'top_right': np.array([-z_max, -y_max]),
            'center_left': np.array([-z_min, 0.]),
            'center_right': np.array([-z_max, 0.]),
            'bottom_center': np.array([0., -y_min]),
            'bottom_left': np.array([-z_min, -y_min]),
            'bottom_right': np.array([-z_max, -y_min])
        }
        sec.snap_points = snap_points
        self.model.elastic_sections.add(sec)
        return sec

    def load_aisc_from_database(
            self,
            sec_shape_designation: str, labels: list[str],
            ops_material: str, physical_material: str,
            sec_type: Type[Section]):
        """
        Loads a section from the AISC steel section database.
        """
        ops_mat = self.model.uniaxial_materials.retrieve_by_attr(
            'name', ops_material)
        phs_mat = self.model.physical_materials.retrieve_by_attr(
            'name', physical_material)
        filename = '../sections.json'
        contents = pkgutil.get_data(__name__, filename)
        assert isinstance(contents, bytes)
        section_dictionary = json.loads(contents)
        assert self.model.settings.imperial_units, 'SI not supported'
        for label in labels:
            try:
                sec_data = section_dictionary[label]
            except KeyError:
                raise KeyError(f'Section {label} not found in file.')
            if sec_shape_designation == 'W':
                sec_b = sec_data['bf']
                sec_h = sec_data['d']
                sec_tw = sec_data['tw']
                sec_tf = sec_data['tf']
                area = sec_data['A']
                outside_shape = mesh_shapes.w_mesh(
                    sec_b, sec_h, sec_tw, sec_tf, area)
                bbox = outside_shape.bounding_box()
                z_min, y_min, z_max, y_max = bbox.flatten()
                snap_points: dict[str, nparr] = {
                    'centroid': np.array([0., 0.]),
                    'top_center': np.array([0., -y_max]),
                    'top_left': np.array([-z_min, -y_max]),
                    'top_right': np.array([-z_max, -y_max]),
                    'center_left': np.array([-z_min, 0.]),
                    'center_right': np.array([-z_max, 0.]),
                    'bottom_center': np.array([0., -y_min]),
                    'bottom_left': np.array([-z_min, -y_min]),
                    'bottom_right': np.array([-z_max, -y_min])
                }
                if sec_type.__name__ == 'FiberSection':
                    main_part = SectionComponent(
                        outside_shape,
                        {},
                        ops_mat,
                        phs_mat)
                    sec_fib = FiberSection(
                        label,
                        self.model.uid_generator.new('section'),
                        outside_shape,
                        {'main': main_part},
                        sec_data['J'],
                        snap_points)
                    self.model.fiber_sections.add(sec_fib)
                elif sec_type.__name__ == 'ElasticSection':
                    sec_el = ElasticSection(
                        label,
                        self.model.uid_generator.new('section'),
                        phs_mat.e_mod,
                        sec_data['A'],
                        sec_data['Iy'],
                        sec_data['Ix'],
                        phs_mat.g_mod,
                        sec_data['J'],
                        sec_data['W'] / 12.00,  # lb/in
                        outside_shape,
                        snap_points,
                        properties=sec_data)
                    self.model.elastic_sections.add(sec_el)
                else:
                    raise ValueError(
                        f'Unsupported section type: {sec_type.__name__}')
            else:
                raise ValueError(
                    'Unsupported section designtation:'
                    f' {sec_shape_designation}')
