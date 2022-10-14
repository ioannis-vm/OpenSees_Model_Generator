"""
Model Generator for OpenSees ~ material generator
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator


from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from ..ops.section import FiberSection
from ..physical_material import PhysicalMaterial
from ..model import Model
from ..ops.uniaxial_material import Steel02
from ..ops.uniaxial_material import MaxStrainRange
from ..ops.uniaxial_material import Bilin
from ..ops.section import ElasticSection
nparr = npt.NDArray[np.float64]


@dataclass(repr=False)
class MaterialGenerator:
    """
    This object introduces element-specific materials to a model.
    """
    model: Model

    def generate_steel_hss_rect_brace_maxstrainrange_mat(
            self,
            section: FiberSection,
            physical_material: PhysicalMaterial,
            brace_length: float,
            node_i_uid: int,
            node_j_uid: int
    ):
        """
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
        var_msr = ((0.554) * (sec_b/sec_t)**(-0.75)
                   * (var_lc/sec_r)**(-0.47)
                   * (mat_e/mat_fy)**(0.21))

        steel02_mat = Steel02(
            self.model.uid_generator.new('uniaxial material'),
            'auto_steel02_brace_mat',
            mat_fy, mat_e, mat_g, param_b, param_r0, param_c_r1, param_c_r2)

        maxstrainrange_mat = MaxStrainRange(
            self.model.uid_generator.new('uniaxial material'),
            'auto_maxstrainrange_brace_mat',
            steel02_mat,
            var_msr, tangent_ratio=1.0e-5, def_coeff=0.068,
            node_tags=(node_i_uid, node_j_uid))

        return maxstrainrange_mat

    def generate_steel_w_imk_material(
            self, section, physical_material, element_length, lboverl,
            loverh, rbs_factor, consider_composite,
            direction='strong'):
        """
        Lignos, D. G., & Krawinkler, H. (2011). Deterioration modeling of
        steel components in support of collapse prediction of steel moment
        frames under earthquake loading. Journal of Structural
        Engineering-Reston, 137(11), 1291.

        Elkady, A., & Lignos, D. G. (2014). Modeling of the composite
        action in fully restrained beam‐to‐column connections:
        implications in the seismic design and collapse capacity of steel
        special moment frames. Earthquake Engineering & Structural
        Dynamics, 43(13), 1935-1954.
        """
        assert section.name[0] == 'W', \
            "Error: Only W sections can be used."
        assert isinstance(section, ElasticSection)
        assert self.model.settings.imperial_units, \
            "Error: Only imperial units supported."
        assert section.properties
        # Young's modulus
        mat_e = section.e_mod / 1.e3
        # Yield stress
        mat_fy = physical_material.f_y / 1.e3
        # Moment of inertia - strong axis - original section
        if direction == 'strong':
            sec_i = section.properties['Ix']
        else:
            sec_i = section.properties['Iy']
        # Section depth
        sec_d = section.properties['d']
        # Flange width
        sec_bf = section.properties['bf']
        # Flange and web thicknesses
        sec_tf = section.properties['tf']
        sec_tw = section.properties['tw']
        # Plastic modulus (unreduced)
        if direction == 'strong':
            sec_z = section.properties['Zx']
        else:
            sec_z = section.properties['Zy']
        # Radius of gyration
        sec_ry = section.properties['ry']
        # Clear length
        elm_h = element_length
        # Shear span
        elm_l = loverh * elm_h
        elm_lb = lboverl * elm_l
        lbry = elm_lb / sec_ry
        if rbs_factor:
            # RBS case
            assert direction == 'strong'
            assert rbs_factor <= 1.00, 'rbs_factor must be <= 1.00'
            # checks ~ acceptable range
            if not 20.00 < sec_d/sec_tw < 55.00:
                print(f'Warning: sec_d/sec_tw={sec_d/sec_tw:.2f}'
                      ' outside regression range')
                print('20.00 < sec_d/sec_tw < 55.00')
                print(section.name, '\n')
            if not 20.00 < lbry < 80.00:
                print(f'Warning: Lb/ry={lbry:.2f} outside regression range')
                print('20.00 < lbry < 80.00')
                print(section.name, '\n')
            if not 4.00 < (sec_bf/(2.*sec_tf)) < 8.00:
                print(f'Warning: bf/(2 tf)={sec_bf/(2.*sec_tf):.2f}'
                      ' outside regression range')
                print('4.00 < (sec_bf/(2.*sec_tf)) < 8.00')
                print(section.name, '\n')
            if not 2.5 < elm_l/sec_d < 7.0:
                print(f'Warning: L/d={elm_l/sec_d:.2f}'
                      '  outside regression range')
                print('2.5 < elm_l/sec_d < 7.0')
                print(section.name, '\n')
            if not 4.00 < sec_d < 36.00:
                print(f'Warning: Section d={sec_d:.2f} '
                      'outside regression range')
                print('4.00 < sec_d < 36.00')
                print(section.name, '\n')
            if not 35.00 < mat_fy < 65.00:
                print(f'Warning: Fy={mat_fy:.2f} outside regression range')
                print('35.00 < mat_fy < 65.00')
                print(section.name, '\n')
            # calculate parameters
            theta_p = 0.19 * (sec_d/sec_tw)**(-0.314) * \
                (sec_bf/(2.*sec_tf))**(-0.10) * \
                lbry**(-0.185) * \
                (elm_l/sec_d)**0.113 * \
                (25.4 * sec_d / 533.)**(-0.76) * \
                (6.895 * mat_fy / 355.)**(-0.07)
            theta_pc = 9.52 * (sec_d/sec_tw)**(-0.513) * \
                (sec_bf/(2.*sec_tf))**(-0.863) * \
                lbry**(-0.108) * \
                (6.895 * mat_fy / 355.)**(-0.36)
            lamda = 585. * (sec_d/sec_tw)**(-1.14) * \
                (sec_bf/(2.*sec_tf))**(-0.632) * \
                lbry**(-0.205) * \
                (6.895 * mat_fy / 355.)**(-0.391)
            rbs_c = sec_bf * (1. - rbs_factor) / 2.
            z_rbs = sec_z - 2. * rbs_c * sec_tf * (sec_d - sec_tf)
            sec_m = 1.06 * z_rbs * mat_fy * 1.e3
        else:
            # Other-than-RBS case
            theta_p = 0.0865 * (sec_d/sec_tw)**(-0.365) * \
                (sec_bf/(2.*sec_tf))**(-0.14) * \
                (elm_l/sec_d)**0.34 * \
                (25.4 * sec_d / 533.)**(-0.721) * \
                (6.895 * mat_fy / 355.)**(-0.23)
            theta_pc = 5.63 * (sec_d/sec_tw)**(-0.565) * \
                (sec_bf/(2.*sec_tf))**(-0.800) * \
                (25.4 * sec_d / 533.)**(-0.28) *  \
                (6.895 * mat_fy / 355.)**(-0.43)
            lamda = 495. * (sec_d/sec_tw)**(-1.34) * \
                (sec_bf/(2.*sec_tf))**(-0.595) * \
                (6.895 * mat_fy / 355.)**(-0.36)
            sec_m = 1.17 * sec_z * mat_fy * 1.e3
        theta_u = 0.20
        residual_plus = 0.40
        residual_minus = 0.40
        theta_p_plus = theta_p
        theta_p_minus = theta_p
        theta_pc_plus = theta_pc
        theta_pc_minus = theta_pc
        d_plus = 1.00
        d_minus = 1.00
        mcmy_plus = 1.0001
        mcmy_minus = 1.0001
        m_plus = sec_m
        m_minus = -sec_m
        if consider_composite:
            # Elkady, A., & Lignos, D. G. (2014). Modeling of the
            # composite action in fully restrained beam‐to‐column
            # connections: implications in the seismic design and
            # collapse capacity of steel special moment
            # frames. Earthquake Engineering & Structural Dynamics,
            # 43(13), 1935-1954.  Table II
            assert direction == 'strong'
            theta_p_plus *= 1.80
            theta_p_minus *= 0.95
            theta_pc_plus *= 1.35
            theta_pc_minus *= 0.95
            d_plus *= 1.15
            d_minus *= 1.00
            mcmy_plus *= 1.30
            mcmy_minus *= 1.05
            m_plus *= 1.35
            m_minus *= 1.25
            residual_plus = 0.30
            residual_minus = 0.20
        stiffness = 6.00 * mat_e * sec_i / elm_h * 1e4
        beta_plus = (mcmy_plus - 1.) * m_plus / (theta_p_plus) / stiffness
        beta_minus = - (mcmy_minus - 1.) * m_minus \
            / (theta_p_minus) / stiffness
        bilin_mat = Bilin(
            self.model.uid_generator.new('uniaxial material'),
            'auto_IMK',
            stiffness,
            beta_plus, beta_minus,
            m_plus, m_minus,
            lamda, lamda, lamda, lamda,
            1.00, 1.00, 1.00, 1.00,
            theta_p_plus, theta_p_minus,
            theta_pc_plus, theta_pc_minus,
            residual_plus, residual_minus,
            theta_u, theta_u,
            d_plus, d_minus,
            0.00
        )
        return bilin_mat
