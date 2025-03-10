"""objects that create materials."""

from dataclasses import dataclass

import numpy as np

from osmg.core.model import Model
from osmg.model_objects.section import ElasticSection, FiberSection
from osmg.model_objects.uniaxial_material import (
    Elastic,
    Fatigue,
    Hysteretic,
    IMKBilin,
    MaxStrainRange,
    Pinching4,
    Steel02,
    UniaxialMaterial,
)


@dataclass(repr=False)
class MaterialCreator:
    """Base class for uniaxial material creators."""

    model: Model

    def generate(self) -> UniaxialMaterial:  # noqa: D102, PLR6301
        msg = 'Subclasses must implement this method.'
        raise NotImplementedError(msg)


@dataclass(repr=False)
class ElasticMaterialCreator(MaterialCreator):
    """Generates an Elastic uniaxial material."""

    stiffness: float

    def generate(self) -> Elastic:
        """
        Generate a material.

        Returns:
          The material.
        """
        return Elastic(
            self.model.uid_generator,
            'Elastic',
            self.stiffness,
        )


@dataclass(repr=False)
class SteelHSSRectBraceMaxStrainRangeMaterialCreator(MaterialCreator):
    """
    Max strain range material.

    Sen, A. D., Roeder, C. W., Lehman, D. E., & Berman,
    J. W. (2019). Nonlinear modeling of concentrically braced
    frames. Journal of Constructional Steel Research, 157,
    103-120.

    """

    section: FiberSection
    e_mod: float
    g_mod: float
    f_y: float
    brace_length: float
    node_i_uid: int
    node_j_uid: int

    def generate(self) -> MaxStrainRange:
        """
        Generate a material.

        Returns:
          The material.
        """
        param_b = 0.001
        param_r0 = 15
        param_c_r1 = 0.925
        param_c_r2 = 0.15

        assert self.section.properties is not None
        sec_b = self.section.properties.B
        sec_t = self.section.properties.tdes
        var_lc = self.brace_length
        sec_r = min(self.section.properties.rx, self.section.properties.ry)
        mat_e = self.e_mod
        mat_g = self.g_mod
        mat_fy = self.f_y
        var_msr = (
            (0.554)
            * (sec_b / sec_t) ** (-0.75)
            * (var_lc / sec_r) ** (-0.47)
            * (mat_e / mat_fy) ** (0.21)
        )

        steel02_mat = Steel02(
            self.model.uid_generator,
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
            self.model.uid_generator,
            'auto_maxstrainrange_brace_mat',
            steel02_mat,
            var_msr,
            tangent_ratio=1.0e-5,
            def_coeff=0.068,
            node_tags=(self.node_i_uid, self.node_j_uid),
        )


@dataclass(repr=False)
class SteelHSSCircBraceFatigueMaterialCreator(MaterialCreator):
    """
    Circular HSS brace fatigue material.

    Karamanaci and Lignos (2014). Computational Approach for
    Collapse Assessment of Concentrically Braced Frames in Seismic
    Regions. Journal of Structural Engineering

    Returns:
      The defined material.
    """

    section: FiberSection
    e_mod: float
    g_mod: float
    f_y: float
    brace_length: float

    def generate(self) -> Fatigue:
        """
        Generate a material.

        Returns:
          The material.
        """
        param_b = 0.005
        param_r0 = 24.00
        param_c_r1 = 0.925
        param_c_r2 = 0.25

        assert self.section.properties is not None
        sec_d = self.section.properties.OD
        sec_t = self.section.properties.tdes
        var_lc = self.brace_length
        sec_r = min(self.section.properties.rx, self.section.properties.ry)
        mat_e = self.e_mod
        mat_g = self.g_mod
        mat_fy = self.f_y
        var_e0 = (
            (0.748)
            * (var_lc / sec_r) ** (-0.399)
            * (sec_d / sec_t) ** (-0.628)
            * (mat_e / mat_fy) ** (0.201)
        )
        var_m = -0.300

        steel02_mat = Steel02(
            self.model.uid_generator,
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
            self.model.uid_generator,
            'auto_fatigue_brace_mat',
            steel02_mat,
            var_e0,
            var_m,
        )


@dataclass(repr=False)
class SteelWIMKMaterialCreator(MaterialCreator):
    """
    Steel W-shape IMK material.

    Assumes force unit is lb and length unit is inch.

    Lignos, D. G., & Krawinkler, H. (2011). Deterioration modeling of
    steel components in support of collapse prediction of steel moment
    frames under earthquake loading. Journal of Structural
    Engineering-Reston, 137(11), 1291.

    Elkady, A., & Lignos, D. G. (2014). Modeling of the composite
    action in fully restrained beam-to-column connections:
    implications in the seismic design and collapse capacity of steel
    special moment frames. Earthquake Engineering & Structural
    Dynamics, 43(13), 1935-1954.

    """

    section: ElasticSection
    f_y: float
    element_length: float
    lboverl: float
    loverh: float
    rbs_factor: float
    consider_composite: bool
    axial_load_ratio: float
    direction: str = 'strong'
    moment_modifier: float = 1.00
    n_parameter: float = 0.00

    def generate(self) -> IMKBilin:  # noqa: C901
        """
        Generate a material.

        Returns:
          The material.
        """
        # gather necessary data and check interpolation range
        assert self.section.name[0] == 'W', 'Error: Only W sections can be used.'
        assert isinstance(self.section, ElasticSection)
        assert self.section.properties
        # Yield stress
        mat_fy = self.f_y
        # Moment of inertia - strong axis - original section
        if self.direction == 'strong':
            sec_i = self.section.properties.Ix
        else:
            sec_i = self.section.properties.Iy
        # Section depth
        sec_d = self.section.properties.d
        # Flange width
        sec_bf = self.section.properties.bf
        # Flange and web thicknesses
        sec_tf = self.section.properties.tf
        sec_tw = self.section.properties.tw
        # Plastic modulus (unreduced)
        if self.direction == 'strong':
            sec_z = self.section.properties.Zx
        else:
            sec_z = self.section.properties.Zy
        # Radius of gyration
        sec_ry = self.section.properties.ry
        # Clear length
        elm_h = self.element_length
        # Shear span
        elm_l = self.loverh * elm_h
        elm_lb = self.lboverl * elm_l
        lbry = elm_lb / sec_ry

        # consider cases

        # checks ~ acceptable range
        if not 20.00 < sec_d / sec_tw < 55.00:  # noqa: PLR2004
            print(
                f'Warning: sec_d/sec_tw={sec_d / sec_tw:.2f}'
                ' outside regression range'
            )
            print('20.00 < sec_d/sec_tw < 55.00')
            print(self.section.name, '\n')
        if not 20.00 < lbry < 80.00:  # noqa: PLR2004
            print(f'Warning: Lb/ry={lbry:.2f} outside regression range')
            print('20.00 < lbry < 80.00')
            print(self.section.name, '\n')
        if not 4.00 < (sec_bf / (2.0 * sec_tf)) < 8.00:  # noqa: PLR2004
            print(
                f'Warning: bf/(2 tf)={sec_bf / (2. * sec_tf):.2f}'
                ' outside regression range'
            )
            print('4.00 < (sec_bf/(2.*sec_tf)) < 8.00')
            print(self.section.name, '\n')
        if not 2.5 < elm_l / sec_d < 7.0:  # noqa: PLR2004
            print(f'Warning: L/d={elm_l / sec_d:.2f}  outside regression range')
            print('2.5 < elm_l/sec_d < 7.0')
            print(self.section.name, '\n')
        if not 4.00 < sec_d < 36.00:  # noqa: PLR2004
            print(f'Warning: Section d={sec_d:.2f} outside regression range')
            print('4.00 < sec_d < 36.00')
            print(self.section.name, '\n')
        if not 35.00 < mat_fy < 65.00:  # noqa: PLR2004
            print(f'Warning: Fy={mat_fy:.2f} outside regression range')
            print('35.00 < mat_fy < 65.00')
            print(self.section.name, '\n')
        if self.rbs_factor:
            # RBS case
            assert self.direction == 'strong'
            assert self.rbs_factor <= 1.00, 'rbs_factor must be <= 1.00'
            # calculate parameters
            theta_p = (
                0.19
                * (sec_d / sec_tw) ** (-0.314)
                * (sec_bf / (2.0 * sec_tf)) ** (-0.10)
                * lbry ** (-0.185)
                * (elm_l / sec_d) ** 0.113
                * (25.4 * sec_d / 533.0) ** (-0.76)
                * (6.895 * mat_fy / 355.0) ** (-0.07)
            )
            theta_pc = (
                9.52
                * (sec_d / sec_tw) ** (-0.513)
                * (sec_bf / (2.0 * sec_tf)) ** (-0.863)
                * lbry ** (-0.108)
                * (6.895 * mat_fy / 355.0) ** (-0.36)
            )
            lambda_var = (
                585.0
                * (sec_d / sec_tw) ** (-1.14)
                * (sec_bf / (2.0 * sec_tf)) ** (-0.632)
                * lbry ** (-0.205)
                * (6.895 * mat_fy / 355.0) ** (-0.391)
            )
            rbs_c = sec_bf * (1.0 - self.rbs_factor) / 2.0
            z_rbs = sec_z - 2.0 * rbs_c * sec_tf * (sec_d - sec_tf)
            sec_m = 1.06 * z_rbs * mat_fy
            mcmy_plus = 1.10
            mcmy_minus = 1.10

        # Other-than-RBS case
        elif self.axial_load_ratio:
            # column case
            theta_p = (
                294.00
                * (sec_d / sec_tw) ** (-1.70)
                * lbry ** (-0.70)
                * (1.00 - self.axial_load_ratio) ** (1.60)
            )
            theta_pc = (
                90.00
                * (sec_d / sec_tw) ** (-0.80)
                * lbry ** (-0.80)
                * (1.00 - self.axial_load_ratio) ** (2.50)
            )
            theta_p = min(theta_p, 0.20)
            theta_pc = min(theta_pc, 0.30)
            if self.axial_load_ratio <= 0.35:  # noqa: PLR2004
                lambda_var = (
                    25500.00
                    * (sec_d / sec_tw) ** (-2.14)
                    * lbry ** (-0.53)
                    * (1.00 - self.axial_load_ratio) ** (4.29)
                )
            else:
                lambda_var = (
                    268000.00
                    * (sec_d / sec_tw) ** (-2.30)
                    * lbry ** (-1.30)
                    * (1.00 - self.axial_load_ratio) ** (1.19)
                )
            if self.axial_load_ratio <= 0.20:  # noqa: PLR2004
                sec_m = (
                    1.15
                    / 1.10
                    * (sec_z * mat_fy)
                    * (1.00 - self.axial_load_ratio / 2.00)
                )
            else:
                sec_m = (
                    1.15
                    / 1.10
                    * (sec_z * mat_fy)
                    * 9.0
                    / 8.0
                    * (1.00 - self.axial_load_ratio)
                )
            mcmy = (
                12.5
                * (sec_d / sec_tw) ** (-0.20)
                * lbry ** (-0.40)
                * (1.00 - self.axial_load_ratio) ** (0.40)
            )
            mcmy = min(mcmy, 1.00)
            mcmy = max(mcmy, 1.30)
            mcmy_plus = mcmy
            mcmy_minus = mcmy

        else:
            # non-RBS beam case
            theta_p = (
                0.0865
                * (sec_d / sec_tw) ** (-0.365)
                * (sec_bf / (2.0 * sec_tf)) ** (-0.14)
                * (elm_l / sec_d) ** 0.34
                * (25.4 * sec_d / 533.0) ** (-0.721)
                * (6.895 * mat_fy / 355.0) ** (-0.23)
            )
            theta_pc = (
                5.63
                * (sec_d / sec_tw) ** (-0.565)
                * (sec_bf / (2.0 * sec_tf)) ** (-0.800)
                * (25.4 * sec_d / 533.0) ** (-0.28)
                * (6.895 * mat_fy / 355.0) ** (-0.43)
            )
            lambda_var = (
                495.0
                * (sec_d / sec_tw) ** (-1.34)
                * (sec_bf / (2.0 * sec_tf)) ** (-0.595)
                * (6.895 * mat_fy / 355.0) ** (-0.36)
            )
            sec_m = 1.17 * sec_z * mat_fy
            mcmy_plus = 1.10
            mcmy_minus = 1.10

        theta_u = 0.20
        residual_plus = 0.40
        residual_minus = 0.40
        theta_p_plus = theta_p
        theta_p_minus = theta_p
        theta_pc_plus = theta_pc
        theta_pc_minus = theta_pc
        d_plus = 1.00
        d_minus = 1.00
        m_plus = sec_m
        m_minus = -sec_m

        if self.consider_composite:
            # Elkady, A., & Lignos, D. G. (2014). Modeling of the
            # composite action in fully restrained beam-to-column
            # connections: implications in the seismic design and
            # collapse capacity of steel special moment
            # frames. Earthquake Engineering & Structural Dynamics,
            # 43(13), 1935-1954.  Table II

            assert (
                self.axial_load_ratio == 0.00
            ), "Can't consider composite action for columns"
            assert (
                self.direction == 'strong'
            ), 'Composite action affects the behavior in strong-axis bending'

            theta_p_plus *= 1.80
            theta_p_minus *= 0.95
            theta_pc_plus *= 1.35
            theta_pc_minus *= 0.95
            d_plus = 1.15
            d_minus = 1.00
            mcmy_plus = 1.30
            mcmy_minus = 1.05
            m_plus *= 1.35
            m_minus *= 1.25
            residual_plus = 0.30
            residual_minus = 0.20

        # adjust parameters to account for the presence of the elastic element
        stiffness_init = 6.00 * self.section.e_mod * sec_i / elm_h
        stiffness = (self.n_parameter + 1.00) * stiffness_init
        theta_y = sec_m / stiffness_init
        theta_p_plus -= (mcmy_plus - 1.0) * (sec_m / stiffness)
        theta_p_minus -= (mcmy_minus - 1.0) * (sec_m / stiffness)
        theta_pc_plus += theta_y + (mcmy_plus - 1.0) * (sec_m / stiffness)
        theta_pc_plus += theta_y + (mcmy_minus - 1.0) * (sec_m / stiffness)
        beta_plus = (mcmy_plus - 1.0) * m_plus / theta_p_plus / stiffness
        beta_minus = -(mcmy_minus - 1.0) * m_minus / theta_p_minus / stiffness

        # new model
        return IMKBilin(
            self.model.uid_generator,
            'auto_IMK',
            stiffness * self.moment_modifier,
            theta_p_plus,
            theta_pc_plus,
            theta_u,
            m_plus * self.moment_modifier,
            (1.0 + beta_plus),
            residual_plus,
            theta_p_minus,
            theta_pc_minus,
            theta_u,
            -m_minus * self.moment_modifier,
            (1.0 + beta_minus),
            residual_minus,
            lambda_var,
            lambda_var,
            lambda_var,
            1.00,
            1.00,
            1.00,
            d_plus,
            d_minus,
        )


@dataclass(repr=False)
class SteelGravityShearTabCreator(MaterialCreator):
    """
    Gravity shear tab connection hinge.

    Assumes force unit is lb and length unit is inch.

    Elkady, A., & Lignos, D. G. (2015). Effect of gravity framing on
    the overstrength and collapse capacity of steel frame buildings
    with perimeter special moment frames. Earthquake Engineering &
    Structural Dynamics, 44(8), 1289-1307.
    """

    section: ElasticSection
    f_y: float
    consider_composite: bool
    moment_modifier: float = 1.00

    def generate(self) -> Pinching4:
        """
        Generate a material.

        Returns:
          The material.
        """
        assert self.section.name[0] == 'W', 'Error: Only W sections can be used.'
        assert isinstance(self.section, ElasticSection)
        assert self.section.properties

        # Yield stress
        assert isinstance(self.moment_modifier, float)
        mat_fy = self.f_y
        # Plastic modulus (unreduced)
        sec_zx = self.section.properties.Zx
        # Plastic moment of the section
        sec_mp = sec_zx * mat_fy * self.moment_modifier

        if not self.consider_composite:
            m_max_pos = 0.121 * sec_mp
            m_max_neg = 0.121 * sec_mp
            m1_p = +0.521 * m_max_pos
            m1_n = -0.521 * m_max_neg
            m2_p = +0.967 * m_max_pos
            m2_n = -0.967 * m_max_neg
            m3_p = +1.000 * m_max_pos
            m3_n = -1.000 * m_max_pos
            m4_p = +0.901 * m_max_pos
            m4_n = -0.901 * m_max_neg
            th_1_p = 0.0045
            th_1_n = -0.0045
            th_2_p = 0.0465
            th_2_n = -0.0465
            th_3_p = 0.0750
            th_3_n = -0.0750
            th_4_p = 0.1000
            th_4_n = -0.1000
            rdispp = 0.57
            rdispn = 0.57
            rforcep = 0.40
            rforcen = 0.40
            uforcep = 0.05
            uforcen = 0.05
            gklim = 0.2
            gdlim = 0.1
            gflim = 0.0
            g_e = 10
            dmgtype = 'energy'
        else:
            m_max_pos = 0.35 * sec_mp
            m_max_neg = 0.64 * 0.35 * sec_mp
            m1_p = +0.250 * m_max_pos
            m1_n = -0.250 * m_max_neg
            m2_p = +1.000 * m_max_pos
            m2_n = -1.000 * m_max_neg
            m3_p = +1.001 * m_max_pos
            m3_n = -1.001 * m_max_pos
            m4_p = +0.530 * m_max_pos
            m4_n = -0.540 * m_max_neg
            th_1_p = 0.0042
            th_1_n = -0.0042
            th_2_p = 0.0200
            th_2_n = -0.0110
            th_3_p = 0.0390
            th_3_n = -0.0300
            th_4_p = 0.0400
            th_4_n = -0.0550
            rdispp = 0.40
            rdispn = 0.50
            rforcep = 0.13
            rforcen = 0.53
            uforcep = 0.01
            uforcen = 0.05
            gklim = 0.30
            gdlim = 0.05
            gflim = 0.05
            g_e = 10
            dmgtype = 'energy'

        return Pinching4(
            self.model.uid_generator,
            'auto_gravity_shear_tab',
            m1_p,
            th_1_p,
            m2_p,
            th_2_p,
            m3_p,
            th_3_p,
            m4_p,
            th_4_p,
            m1_n,
            th_1_n,
            m2_n,
            th_2_n,
            m3_n,
            th_3_n,
            m4_n,
            th_4_n,
            rdispp,
            rforcep,
            uforcep,
            rdispn,
            rforcen,
            uforcen,
            0.00,
            0.00,
            0.00,
            0.00,
            gklim,
            0.00,
            0.00,
            0.00,
            0.00,
            gdlim,
            0.00,
            0.00,
            0.00,
            0.00,
            gflim,
            g_e,
            dmgtype,
        )


@dataclass(repr=False)
class SteelWColumnPanelZoneCreator(MaterialCreator):
    """
    Panel zone hinge (parallelogram model).

    Assumes force unit is lb and length unit is inch.

    Gupta, A., & Krawinkler, H. (1999). Seismic demands for the
    performance evaluation of steel moment resisting frame
    structures. Rep. No. 132.
    """

    section: ElasticSection
    f_y: float
    g_mod: float
    pz_length: float
    pz_doubler_plate_thickness: float
    pz_hardening: float
    moment_modifier: float = 1.00

    def generate(self) -> Hysteretic:
        """
        Generate a material.

        Returns:
          The material.
        """
        assert self.section.name[0] == 'W', 'Error: Only W sections can be used.'
        assert isinstance(self.section, ElasticSection)
        assert self.section.properties
        f_y = self.f_y
        hardening = self.pz_hardening
        d_c = self.section.properties.d
        bfc = self.section.properties.bf
        t_p = self.section.properties.tw + self.pz_doubler_plate_thickness
        t_f = self.section.properties.tf
        v_y = 0.55 * f_y * d_c * t_p
        g_mod = self.g_mod
        k_e = 0.95 * g_mod * t_p * d_c
        k_p = 0.95 * g_mod * bfc * t_f**2 / self.pz_length
        gamma_1 = v_y / k_e
        gamma_2 = 4.0 * gamma_1
        gamma_3 = 100.0 * gamma_1
        m1y = gamma_1 * k_e * self.pz_length * self.moment_modifier
        m2y = m1y + k_p * self.pz_length * (gamma_2 - gamma_1) * self.moment_modifier
        m3y = (
            m2y
            + (hardening * k_e * self.pz_length)
            * (gamma_3 - gamma_2)
            * self.moment_modifier
        )

        return Hysteretic(
            self.model.uid_generator,
            'auto_steel_W_PZ',
            (m1y, gamma_1),
            (m2y, gamma_2),
            (m3y, gamma_3),
            (-m1y, -gamma_1),
            (-m2y, -gamma_2),
            (-m3y, -gamma_3),
            0.25,
            0.75,
            0.00,
            0.00,
            0.00,
        )


@dataclass(repr=False)
class SteelWColumnPanelZoneUpdatedCreator(MaterialCreator):
    """
    Define updated panel zone hinge (parallelogram model).

    Assumes force unit is lb and length unit is inch.

    Skiadopoulos, A., Elkady, A. and D. G. Lignos (2020). "Proposed
    Panel Zone Model for Seismic Design of Steel Moment-Resisting
    Frames." ASCE Journal of Structural Engineering. DOI:
    10.1061/(ASCE)ST.1943-541X.0002935.

    """

    section: ElasticSection
    e_mod: float
    g_mod: float
    f_y: float
    pz_length: float
    pz_doubler_plate_thickness: float
    axial_load_ratio: float
    slab_depth: float
    location: str
    consider_composite: bool
    moment_modifier: float = 1.00

    def generate(self) -> Hysteretic:  # noqa: PLR0914
        """
        Generate a material.

        Returns:
          The material.

        Raises:
          ValueError: If the specified `location` is invalid.
        """
        assert self.section.name[0] == 'W', 'Error: Only W sections can be used.'
        assert isinstance(self.section, ElasticSection)
        assert self.section.properties
        f_y = self.f_y
        e_mod = self.e_mod
        g_mod = self.g_mod
        tw_Col = self.section.properties.tw  # noqa: N806
        tdp = self.pz_doubler_plate_thickness
        d_Col = self.section.properties.d  # noqa: N806
        d_Beam = self.pz_length  # noqa: N806
        tf_Col = self.section.properties.tf  # noqa: N806
        bf_Col = self.section.properties.bf  # noqa: N806
        Ix_Col = self.section.properties.Ix  # noqa: N806
        ts = self.slab_depth
        n = self.axial_load_ratio
        trib = self.slab_depth

        tpz = tw_Col + tdp  # total PZ thickness
        # effective depth in positive moment
        d_BeamP = d_Beam  # noqa: N806
        if self.consider_composite:
            d_BeamP = d_Beam + trib + 0.50 * ts  # noqa: N806
        # effective depth in negative moment
        d_BeamN = d_Beam  # noqa: N806

        # Stiffness Calculation
        Ks = tpz * (d_Col - tf_Col) * g_mod  # noqa: N806
        Kb = (  # noqa: N806
            12.0
            * e_mod
            * (Ix_Col + tdp * ((d_Col - 2.0 * tf_Col) ** 3) / 12.00)
            / (d_Beam**3)
            * d_Beam
        )
        Ke = Ks * Kb / (Ks + Kb)  # noqa: N806

        # flange stiffness: shear contribution
        Ksf = 2.0 * (bf_Col * tf_Col) * g_mod  # noqa: N806
        # flange stiffness: bending contribution
        Kbf = 2.0 * 12.0 * e_mod * bf_Col * (tf_Col**3) / 12.0 / (d_Beam**3) * d_Beam  # noqa: N806
        # flange stiffness: total contribution
        Kef = (Ksf * Kbf) / (Ksf + Kbf)  # noqa: N806

        ay = (0.58 * Kef / Ke + 0.88) / (1.0 - Kef / Ke)

        aw_eff_4gamma = 1.10
        aw_eff_6gamma = 1.15

        af_eff_4gamma = 0.93 * Kef / Ke + 0.015
        af_eff_6gamma = 1.05 * Kef / Ke + 0.020
        # reduction factor accounting for axial load
        r = np.sqrt(1.0 - (n**2))

        Vy = r * 0.577 * f_y * ay * (d_Col - tf_Col) * tpz  # noqa: N806
        # Plastic Shear Force at 4 gammaY
        Vp_4gamma = (  # noqa: N806
            r
            * 0.577
            * f_y
            * (
                aw_eff_4gamma * (d_Col - tf_Col) * tpz
                + af_eff_4gamma * (bf_Col - tw_Col) * 2 * tf_Col
            )
        )
        # Plastic Shear Force at 6 gammaY
        Vp_6gamma = (  # noqa: N806
            r
            * 0.577
            * f_y
            * (
                aw_eff_6gamma * (d_Col - tf_Col) * tpz
                + af_eff_6gamma * (bf_Col - tw_Col) * 2 * tf_Col
            )
        )

        gamma_y = Vy / Ke
        gamma4_y = 4.0 * gamma_y

        My_P = Vy * d_BeamP  # noqa: N806
        Mp_4gamma_P = Vp_4gamma * d_BeamP  # noqa: N806
        Mp_6gamma_P = Vp_6gamma * d_BeamP  # noqa: N806

        My_N = Vy * d_BeamN  # noqa: N806
        Mp_4gamma_N = Vp_4gamma * d_BeamN  # noqa: N806
        Mp_6gamma_N = Vp_6gamma * d_BeamN  # noqa: N806

        Slope_4to6gamma_y_P = (Mp_6gamma_P - Mp_4gamma_P) / (2.0 * gamma_y)  # noqa: N806
        Slope_4to6gamma_y_N = (Mp_6gamma_N - Mp_4gamma_N) / (2.0 * gamma_y)  # noqa: N806

        # Defining the 3 Points used to construct the trilinear backbone curve
        gamma1 = gamma_y
        gamma2 = gamma4_y
        gamma3 = 100.0 * gamma_y

        M1_P = My_P  # noqa: N806
        M2_P = Mp_4gamma_P  # noqa: N806
        M3_P = Mp_4gamma_P + Slope_4to6gamma_y_P * (100 * gamma_y - gamma4_y)  # noqa: N806

        M1_N = My_N  # noqa: N806
        M2_N = Mp_4gamma_N  # noqa: N806
        M3_N = Mp_4gamma_N + Slope_4to6gamma_y_N * (100 * gamma_y - gamma4_y)  # noqa: N806

        # apply moment modifier
        M1_P *= self.moment_modifier  # noqa: N806
        M2_P *= self.moment_modifier  # noqa: N806
        M3_P *= self.moment_modifier  # noqa: N806
        M1_N *= self.moment_modifier  # noqa: N806
        M2_N *= self.moment_modifier  # noqa: N806
        M3_N *= self.moment_modifier  # noqa: N806

        if not self.consider_composite:
            args = (
                (M1_N, gamma1),
                (M2_N, gamma2),
                (M3_N, gamma3),
                (-M1_N, -gamma1),
                (-M2_N, -gamma2),
                (-M3_N, -gamma3),
                0.25,
                0.75,
                0.0,
                0.0,
                0.0,
            )
        elif self.location == 'interior':
            args = (
                (M1_P, gamma1),
                (M2_P, gamma2),
                (M3_P, gamma3),
                (-M1_P, -gamma1),
                (-M2_P, -gamma2),
                (-M3_P, -gamma3),
                0.25,
                0.75,
                0.0,
                0.0,
                0.0,
            )
        elif self.location == 'exterior_first':
            args = (
                (M1_N, gamma1),
                (M2_N, gamma2),
                (M3_N, gamma3),
                (-M1_P, -gamma1),
                (-M2_P, -gamma2),
                (-M3_P, -gamma3),
                0.25,
                0.75,
                0.0,
                0.0,
                0.0,
            )
        elif self.location == 'exterior_last':
            args = (
                (M1_P, gamma1),
                (M2_P, gamma2),
                (M3_P, gamma3),
                (-M1_N, -gamma1),
                (-M2_N, -gamma2),
                (-M3_N, -gamma3),
                0.25,
                0.75,
                0.0,
                0.0,
                0.0,
            )
        else:
            msg = f'Invalid Location: {self.location}'
            raise ValueError(msg)

        return Hysteretic(
            self.model.uid_generator,
            'auto_steel_W_pz_updated',
            *args,
        )


class SteelBraceGussetCreator(MaterialCreator):
    """
    Steel brace gusset plate hinge.

    Hsiao, P-C., Lehman, D.E., and Roeder, C.W., 2012, Improved
    analytical model for special concentrically braced frames, Journal
    of Constructional Steel Research, Vol. 73, pp 80-94.

    Arguments:
      model: Model object.
      d_brace: brace section height.
      l_c: brace-to-gusset connection length.
      t_p: gusset plate thickness.
      l_b: gusset plate average buckling length.
      **kwargs: Other keyword arguments.

    """

    model: Model
    e_mod: float
    g_mod: float
    f_y: float
    d_brace: float
    l_c: float
    t_p: float
    l_b: float

    def generate(self) -> Steel02:
        """
        Generate a material.

        Returns:
          The material.
        """
        var_w = self.d_brace + 2.00 * self.l_c * np.tan(30.00 / 180.00 * np.pi)
        var_i = var_w * self.t_p**3 / 12.00
        var_z = var_w * self.t_p**2 / 6.00
        f_y = self.f_y
        var_e = self.e_mod
        var_g = self.g_mod
        var_my = var_z * f_y
        var_k_rot = var_e * var_i / self.l_b
        var_b = 0.01
        return Steel02(
            self.model.uid_generator,
            'auto_steel_gusset',
            var_my,
            var_k_rot,
            var_g,
            var_b,
            20.00,
            0.925,
            0.15,
            0.0005,
            0.014,
            0.0005,
            0.01,
            0.00,
        )
