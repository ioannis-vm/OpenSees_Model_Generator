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
from ..ops.section import SectionComponent
import numpy as np
import numpy.typing as npt
import json
from ..ops.section import ElasticSection
from ..ops.section import FiberSection
from ..gen import mesh_shapes
from .mesh_shapes import rect_mesh
import pkgutil
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
            ea: float,
            ei: float,
            gj: float
    ):
        sec = ElasticSection(
            name=name,
            uid=self.model.uid_generator.new('section'),
            outside_shape=None,
            snap_points=None,
            E=1.00,
            A=ea,
            Iy=ei,
            Ix=ei,
            G=1.00,
            J=gj,
            W=0.00,
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

    def load_AISC_from_database(
            self,
            sec_shape_designation: str, labels: list[str],
            ops_material: str, physical_material: str,
            sec_type: Type[Section]):
        ops_mat = self.model.uniaxial_materials.retrieve_by_attr(
            'name', ops_material)
        phs_mat = self.model.physical_materials.retrieve_by_attr(
            'name', physical_material)
        filename = '../../../section_data/sections.json'
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
                b = sec_data['bf']
                h = sec_data['d']
                tw = sec_data['tw']
                tf = sec_data['tf']
                area = sec_data['A']
                outside_shape = mesh_shapes.w_mesh(b, h, tw, tf, area)
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
                        phs_mat.E,
                        sec_data['A'],
                        sec_data['Iy'],
                        sec_data['Ix'],
                        phs_mat.G,
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
