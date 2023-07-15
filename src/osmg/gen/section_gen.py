"""
Objects that generate sections.

"""

#
#   _|_|      _|_|_|  _|      _|    _|_|_|
# _|    _|  _|        _|_|  _|_|  _|
# _|    _|    _|_|    _|  _|  _|  _|  _|_|
# _|    _|        _|  _|      _|  _|    _|
#   _|_|    _|_|_|    _|      _|    _|_|_|
#
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

    Attributes:
      model: Model to act upon

    """

    model: Model

    def generate_generic_elastic(
        self, name: str, e_times_a: float, e_times_i: float, g_times_j: float
    ) -> ElasticSection:
        """
        Generates an ElasticSection object with the specified properties.

        Arguments:
            name: Name of the section.
            e_times_a:
                The product of the elastic modulus and the area of the
                section.
            e_times_i:
                The product of the elastic modulus and the moment of
                inertia.
            g_times_j:
                The product of the shear modulus and the torsional
                moment of inertia.

        Returns:
            ElasticSection: An ElasticSection object with the
                specified properties.

        Example:
            >>> from osmg.gen.section_gen import SectionGenerator
            >>> from osmg.model import Model
            >>> model = Model('test_model')
            >>> sec_gen = SectionGenerator(model)
            >>> sec = sec_gen.generate_generic_elastic(
            ...     name="My Elastic Section",
            ...     e_times_a=100.00, e_times_i=1000.00, g_times_j=500.00)
            >>> sec.name
            'My Elastic Section'
            >>> sec.area
            100.0
            >>> sec.i_y
            1000.0
            >>> sec.i_x
            1000.0
            >>> sec.j_mod
            500.0

        """

        sec = ElasticSection(
            name=name,
            uid=self.model.uid_generator.new("section"),
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
            "centroid": np.array([0.0, 0.0]),
            "top_center": np.array([0.0, -y_max]),
            "top_left": np.array([-z_min, -y_max]),
            "top_right": np.array([-z_max, -y_max]),
            "center_left": np.array([-z_min, 0.0]),
            "center_right": np.array([-z_max, 0.0]),
            "bottom_center": np.array([0.0, -y_min]),
            "bottom_left": np.array([-z_min, -y_min]),
            "bottom_right": np.array([-z_max, -y_min]),
        }
        sec.snap_points = snap_points
        self.model.elastic_sections.add(sec)
        return sec

    def load_aisc_from_database(
        self,
        sec_shape_designation: str,
        labels: list[str],
        ops_material: str,
        physical_material: str,
        sec_type: Type[Section],
        store_in_model: bool = True,
        return_section: bool = False,
    ) -> dict[str, ElasticSection | FiberSection]:
        """
        Loads a section from the AISC steel section database.

        Arguments:
            sec_shape_designation: Designation of the type of section
                to be loaded.
            labels: List of labels of the sections to be loaded.
            ops_material: Name of the uniaxial material to be
                associated with the section.
            physical_material: Name of the physical material to be
                associated with the section.
            sec_type: Type of section to be created.
            store_in_model: If True, the created sections are stored
                in the model.
            return_section: If True, the created sections are returned.

        Returns:
            If `return_section` is True, a dictionary containing the
            created sections. The keys are the labels of the sections,
            and the values are the sections themselves.

        Example:
            >>> from osmg.gen.section_gen import SectionGenerator
            >>> from osmg.model import Model
            >>> from osmg.defaults import load_default_steel
            >>> from osmg.ops.section import ElasticSection
            >>> model = Model('test_model')
            >>> load_default_steel(model)
            >>> sec_gen = SectionGenerator(model)
            >>> sec_gen.load_aisc_from_database(
            ...     'W', ['W14X90'], 'default steel', 'default steel',
            ...     ElasticSection, store_in_model=True, return_section=True)
            {'W14X90': ElasticSection object
            name: W14X90
            uid: 0
            Properties:  E: 29000000.0
              A: 26.5
              Iy: 362.0
              Ix: 999.0
              G: 11153846.15
              J: 4.06
              W: 7.5
            outside_shape: specified
            snap_points: specified
            }

        """

        ops_mat = self.model.uniaxial_materials.retrieve_by_attr(
            "name", ops_material
        )
        phs_mat = self.model.physical_materials.retrieve_by_attr(
            "name", physical_material
        )
        filename = "../sections.json"
        contents = pkgutil.get_data(__name__, filename)
        assert isinstance(contents, bytes)
        section_dictionary = json.loads(contents)
        assert self.model.settings.imperial_units, "SI not supported"
        returned_sections: dict[str, ElasticSection | FiberSection] = {}
        for label in labels:
            try:
                sec_data = section_dictionary[label]
            except KeyError as exc:
                raise KeyError(f"Section {label} not found in file.") from exc
            if sec_shape_designation == "W":
                assert sec_data["Type"] == "W"
                sec_b = sec_data["bf"]
                sec_h = sec_data["d"]
                sec_tw = sec_data["tw"]
                sec_tf = sec_data["tf"]
                area = sec_data["A"]
                outside_shape = mesh_shapes.w_mesh(
                    sec_b, sec_h, sec_tw, sec_tf, area
                )
                bbox = outside_shape.bounding_box()
                z_min, y_min, z_max, y_max = bbox.flatten()
                snap_points: dict[str, nparr] = {
                    "centroid": np.array([0.0, 0.0]),
                    "top_center": np.array([0.0, -y_max]),
                    "top_left": np.array([-z_min, -y_max]),
                    "top_right": np.array([-z_max, -y_max]),
                    "center_left": np.array([-z_min, 0.0]),
                    "center_right": np.array([-z_max, 0.0]),
                    "bottom_center": np.array([0.0, -y_min]),
                    "bottom_left": np.array([-z_min, -y_min]),
                    "bottom_right": np.array([-z_max, -y_min]),
                }
                if sec_type.__name__ == "FiberSection":
                    main_part = SectionComponent(
                        outside_shape, {}, ops_mat, phs_mat
                    )
                    sec_fib = FiberSection(
                        name=label,
                        uid=self.model.uid_generator.new("section"),
                        outside_shape=outside_shape,
                        section_parts={"main": main_part},
                        j_mod=sec_data["J"],
                        snap_points=snap_points,
                        properties=sec_data,
                        n_x=10,
                        n_y=10,
                    )
                    if store_in_model:
                        self.model.fiber_sections.add(sec_fib)
                    if return_section:
                        returned_sections[sec_fib.name] = sec_fib
                elif sec_type.__name__ == "ElasticSection":
                    sec_el = ElasticSection(
                        label,
                        self.model.uid_generator.new("section"),
                        phs_mat.e_mod,
                        sec_data["A"],
                        sec_data["Iy"],
                        sec_data["Ix"],
                        phs_mat.g_mod,
                        sec_data["J"],
                        sec_data["W"] / 12.00,  # lb/in
                        outside_shape,
                        snap_points,
                        properties=sec_data,
                    )
                    if store_in_model:
                        self.model.elastic_sections.add(sec_el)
                    if return_section:
                        returned_sections[sec_el.name] = sec_el
                else:
                    raise ValueError(
                        f"Unsupported section type: {sec_type.__name__}"
                    )
            elif sec_shape_designation == "HSS_rect":
                assert sec_data["Type"] == "HSS"
                # must be rectangle: name will have 2 X's.
                assert len(label.split("X")) == 3
                sec_ht = sec_data["Ht"]
                sec_b = sec_data["B"]
                sec_t = sec_data["tdes"]
                outside_shape = mesh_shapes.rect_mesh(sec_b, sec_ht)
                hole = mesh_shapes.rect_mesh(
                    sec_b - 2.00 * sec_t, sec_ht - 2.00 * sec_t
                )
                bbox = outside_shape.bounding_box()
                z_min, y_min, z_max, y_max = bbox.flatten()
                snap_points = {
                    "centroid": np.array([0.0, 0.0]),
                    "top_center": np.array([0.0, -y_max]),
                    "top_left": np.array([-z_min, -y_max]),
                    "top_right": np.array([-z_max, -y_max]),
                    "center_left": np.array([-z_min, 0.0]),
                    "center_right": np.array([-z_max, 0.0]),
                    "bottom_center": np.array([0.0, -y_min]),
                    "bottom_left": np.array([-z_min, -y_min]),
                    "bottom_right": np.array([-z_max, -y_min]),
                }
                if sec_type.__name__ == "FiberSection":
                    main_part = SectionComponent(
                        outside_shape, {"hole": hole}, ops_mat, phs_mat
                    )
                    sec_fib = FiberSection(
                        label,
                        self.model.uid_generator.new("section"),
                        outside_shape,
                        {"main": main_part},
                        sec_data["J"],
                        snap_points,
                        sec_data,
                        n_x=10,
                        n_y=10,
                    )
                    if store_in_model:
                        self.model.fiber_sections.add(sec_fib)
                    if return_section:
                        returned_sections[sec_fib.name] = sec_fib
                elif sec_type.__name__ == "ElasticSection":
                    sec_el = ElasticSection(
                        label,
                        self.model.uid_generator.new("section"),
                        phs_mat.e_mod,
                        sec_data["A"],
                        sec_data["Iy"],
                        sec_data["Ix"],
                        phs_mat.g_mod,
                        sec_data["J"],
                        sec_data["W"] / 12.00,  # lb/in
                        outside_shape,
                        snap_points,
                        properties=sec_data,
                    )
                    if store_in_model:
                        self.model.elastic_sections.add(sec_el)
                    if return_section:
                        returned_sections[sec_el.name] = sec_el
                else:
                    raise ValueError(
                        f"Unsupported section type: {sec_type.__name__}"
                    )
            elif sec_shape_designation == "HSS_circ":
                # TODO: eliminate some redundant code here by merging
                # suare and round HSS
                assert sec_data["Type"] == "HSS"
                # must be circular: name will have 1.
                assert len(label.split("X")) == 2
                sec_h = sec_data["OD"]
                sec_t = sec_data["tdes"]
                outside_shape = mesh_shapes.circ_mesh(sec_h)
                hole = mesh_shapes.circ_mesh(
                    sec_h - 2.00 * sec_t
                )
                bbox = outside_shape.bounding_box()
                z_min, y_min, z_max, y_max = bbox.flatten()
                snap_points = {
                    "centroid": np.array([0.0, 0.0]),
                    "top_center": np.array([0.0, -y_max]),
                    "top_left": np.array([-z_min, -y_max]),
                    "top_right": np.array([-z_max, -y_max]),
                    "center_left": np.array([-z_min, 0.0]),
                    "center_right": np.array([-z_max, 0.0]),
                    "bottom_center": np.array([0.0, -y_min]),
                    "bottom_left": np.array([-z_min, -y_min]),
                    "bottom_right": np.array([-z_max, -y_min]),
                }
                if sec_type.__name__ == "FiberSection":
                    main_part = SectionComponent(
                        outside_shape, {"hole": hole}, ops_mat, phs_mat
                    )
                    sec_fib = FiberSection(
                        label,
                        self.model.uid_generator.new("section"),
                        outside_shape,
                        {"main": main_part},
                        sec_data["J"],
                        snap_points,
                        sec_data,
                        n_x=10,
                        n_y=10,
                    )
                    if store_in_model:
                        self.model.fiber_sections.add(sec_fib)
                    if return_section:
                        returned_sections[sec_fib.name] = sec_fib
                elif sec_type.__name__ == "ElasticSection":
                    sec_el = ElasticSection(
                        label,
                        self.model.uid_generator.new("section"),
                        phs_mat.e_mod,
                        sec_data["A"],
                        sec_data["Iy"],
                        sec_data["Ix"],
                        phs_mat.g_mod,
                        sec_data["J"],
                        sec_data["W"] / 12.00,  # lb/in
                        outside_shape,
                        snap_points,
                        properties=sec_data,
                    )
                    if store_in_model:
                        self.model.elastic_sections.add(sec_el)
                    if return_section:
                        returned_sections[sec_el.name] = sec_el
                else:
                    raise ValueError(
                        f"Unsupported section type: {sec_type.__name__}"
                    )
            else:
                raise ValueError(
                    "Unsupported section designtation:"
                    f" {sec_shape_designation}"
                )
        return returned_sections
