"""Objects that generate materials."""

#
#   _|_|      _|_|_|  _|      _|    _|_|_|
# _|    _|  _|        _|_|  _|_|  _|
# _|    _|    _|_|    _|  _|  _|  _|  _|_|
# _|    _|        _|  _|      _|  _|    _|
#   _|_|    _|_|_|    _|      _|    _|_|_|
#
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from typing import overload, Literal

from osmg.model import Model
from osmg.ops.section import ElasticSection, FiberSection
from osmg.ops.uniaxial_material import (
    Elastic,
    Fatigue,
    IMKBilin,
    MaxStrainRange,
    Steel02,
)
from osmg.physical_material import PhysicalMaterial

nparr = npt.NDArray[np.float64]


@dataclass(repr=False)
class MaterialGenerator:
    """This object introduces element-specific materials to a model."""

    model: Model

    def generate_steel_hss_rect_brace_maxstrainrange_mat(
        self,
        section: FiberSection,
        physical_material: PhysicalMaterial,
        brace_length: float,
        node_i_uid: int,
        node_j_uid: int,
    ) -> MaxStrainRange:
        """
        Max strain range material.

        Sen, A. D., Roeder, C. W., Lehman, D. E., & Berman,
        J. W. (2019). Nonlinear modeling of concentrically braced
        frames. Journal of Constructional Steel Research, 157,
        103-120.

        """
        param_b = 0.001
        param_r0 = 15
        param_c_r1 = 0.925
        param_c_r2 = 0.15

        assert section.properties is not None
        sec_b = section.properties['B']
        sec_t = section.properties['tdes']
        var_lc = brace_length
        sec_r = min(section.properties['rx'], section.properties['ry'])
        mat_e = physical_material.e_mod
        mat_g = physical_material.g_mod
        mat_fy = physical_material.f_y
        var_msr = (
            (0.554)
            * (sec_b / sec_t) ** (-0.75)
            * (var_lc / sec_r) ** (-0.47)
            * (mat_e / mat_fy) ** (0.21)
        )

        steel02_mat = Steel02(
            self.model.uid_generator.new('uniaxial material'),
            'auto_steel02_brace_mat',
            mat_fy,
            mat_e,
            mat_g,
            param_b,
            param_r0,
            param_c_r1,
            param_c_r2,
        )

        return MaxStrainRange(
            self.model.uid_generator.new('uniaxial material'),
            'auto_maxstrainrange_brace_mat',
            steel02_mat,
            var_msr,
            tangent_ratio=1.0e-5,
            def_coeff=0.068,
            node_tags=(node_i_uid, node_j_uid),
        )

    def generate_steel_hss_circ_brace_fatigue_mat(
        self,
        section: FiberSection,
        physical_material: PhysicalMaterial,
        brace_length: float,
    ) -> Fatigue:
        """
        Circular HSS brace fatigue material.

        Karamanaci and Lignos (2014). Computational Approach for
        Collapse Assessment of Concentrically Braced Frames in Seismic
        Regions. Journal of Structural Engineering

        """
        param_b = 0.005
        param_r0 = 24.00
        param_c_r1 = 0.925
        param_c_r2 = 0.25

        assert section.properties is not None
        sec_d = section.properties['OD']
        sec_t = section.properties['tdes']
        var_lc = brace_length
        sec_r = min(section.properties['rx'], section.properties['ry'])
        mat_e = physical_material.e_mod
        mat_g = physical_material.g_mod
        mat_fy = physical_material.f_y
        var_e0 = (
            (0.748)
            * (var_lc / sec_r) ** (-0.399)
            * (sec_d / sec_t) ** (-0.628)
            * (mat_e / mat_fy) ** (0.201)
        )
        var_m = -0.300

        steel02_mat = Steel02(
            self.model.uid_generator.new('uniaxial material'),
            'auto_steel02_brace_mat',
            mat_fy,
            mat_e,
            mat_g,
            param_b,
            param_r0,
            param_c_r1,
            param_c_r2,
            a1=0.2,
            a2=1.0,
            a3=0.2,
            a4=1.0,
        )

        return Fatigue(
            self.model.uid_generator.new('uniaxial material'),
            'auto_fatigue_brace_mat',
            steel02_mat,
            var_e0,
            var_m,
        )


class YourClass:
    @overload
    def generate_steel_w_imk_material(
        self,
        *,
        section: ElasticSection,
        physical_material: PhysicalMaterial,
        element_length: float,
        lboverl: float,
        loverh: float,
        rbs_factor: float,
        consider_composite: bool,
        axial_load_ratio: float,
        direction: str = 'strong',
        moment_modifier: float = 1.00,
        n_parameter: float = 0.00,
        only_elastic: Literal[True],
    ) -> Elastic: ...

    def generate_steel_w_imk_material(
        self,
        *,
        section: ElasticSection,
        physical_material: PhysicalMaterial,
        element_length: float,
        lboverl: float,
        loverh: float,
        rbs_factor: float,
        consider_composite: bool,
        axial_load_ratio: float,
        direction: str = 'strong',
        moment_modifier: float = 1.00,
        n_parameter: float = 0.00,
        only_elastic: bool = False,
    ):
        # Implementation
        if only_elastic:
            return Elastic(
                self.model.uid_generator.new('uniaxial material'),
                'auto_IMK',
                stiffness * moment_modifier,
            )
        return IMKBilin(
            self.model.uid_generator.new('uniaxial material'),
            'auto_IMK',
            stiffness * moment_modifier,
            theta_p_plus,
            theta_pc_plus,
            theta_u,
            m_plus * moment_modifier,
            (1.0 + beta_plus),
            residual_plus,
            theta_p_minus,
            theta_pc_minus,
            theta_u,
            -m_minus * moment_modifier,
            (1.0 + beta_minus),
            residual_minus,
            lamda,
            lamda,
            lamda,
            1.00,
            1.00,
            1.00,
            d_plus,
            d_minus,
        )
