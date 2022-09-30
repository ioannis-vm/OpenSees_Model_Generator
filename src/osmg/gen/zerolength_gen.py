"""
Model Generator for OpenSees ~ zero length element uniaxial materials
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

# pylint: disable=unused-argument


from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Optional
import numpy as np
from ..ops.section import ElasticSection
from ..ops.uniaxial_material import Steel02
from ..ops.uniaxial_material import Pinching4
from ..ops.uniaxial_material import Hysteretic
from .material_gen import MaterialGenerator

if TYPE_CHECKING:
    from ..model import Model
    from ..physical_material import PhysicalMaterial


def fix_all(model: Model, **kwargs):
    """
    Fixed in all directions
    """
    dirs = [1, 2, 3, 4, 5, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr('name', 'fix')
    mats = [fix_mat]*6
    return dirs, mats


def release_6(model: Model, **kwargs):
    """
    Frees strong axis bending
    """
    dirs = [1, 2, 3, 4, 5]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr('name', 'fix')
    mats = [fix_mat]*5
    return dirs, mats


def release_5(model: Model, **kwargs):
    """
    Frees weak axis bending
    """
    dirs = [1, 2, 3, 4, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr('name', 'fix')
    mats = [fix_mat]*5
    return dirs, mats


def release_56(model: Model, **kwargs):
    """
    Frees both strong and weak axis bending
    """
    dirs = [1, 2, 3, 4]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr('name', 'fix')
    mats = [fix_mat]*4
    return dirs, mats


def imk_6(
        model: Model,
        element_length: float,
        lboverl: float,
        loverh: float,
        rbs_factor: Optional[float],
        consider_composite: bool,
        section: ElasticSection,
        physical_material: PhysicalMaterial,
        **kwargs):
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
    mat_generator = MaterialGenerator(model)
    mat = mat_generator.generate_steel_w_imk_material(
        section, physical_material, element_length,
        lboverl, loverh, rbs_factor, consider_composite,
        direction='strong')
    dirs = [1, 2, 3, 4, 5, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr('name', 'fix')
    mats = [fix_mat]*5 + [mat]
    return dirs, mats


def imk_56(
        model: Model,
        element_length: float,
        lboverl: float,
        loverh: float,
        rbs_factor: Optional[float],
        consider_composite: bool,
        section: ElasticSection,
        physical_material: PhysicalMaterial,
        **kwargs):
    """
    release in the weak axis bending direction,
    imk (see imk docstring) in the strong axis bending direction
    """
    mat_generator = MaterialGenerator(model)
    mat_strong = mat_generator.generate_steel_w_imk_material(
        section, physical_material, element_length,
        lboverl, loverh, rbs_factor, consider_composite,
        direction='strong')
    mat_weak = mat_generator.generate_steel_w_imk_material(
        section, physical_material, element_length,
        lboverl, loverh, rbs_factor, consider_composite,
        direction='weak')
    dirs = [1, 2, 3, 4, 5, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr('name', 'fix')
    mats = [fix_mat]*4 + [mat_weak, mat_strong]
    return dirs, mats


def gravity_shear_tab(
        model: Model,
        consider_composite: bool,
        section: ElasticSection,
        physical_material: PhysicalMaterial,
        **kwargs):
    """
    Elkady, A., & Lignos, D. G. (2015). Effect of gravity framing on
    the overstrength and collapse capacity of steel frame buildings
    with perimeter special moment frames. Earthquake Engineering &
    Structural Dynamics, 44(8), 1289-1307.
    """
    assert section.name[0] == 'W', \
        "Error: Only W sections can be used."
    assert isinstance(section, ElasticSection)
    assert model.settings.imperial_units, \
        "Error: Only imperial units supported."
    assert section.properties

    # Yield stress
    mat_fy = physical_material.f_y / 1.e3
    # Plastic modulus (unreduced)
    sec_zx = section.properties['Zx']
    # Plastic moment of the section
    sec_mp = sec_zx * mat_fy * 1.e3
    if not consider_composite:
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
        m_max_neg = 0.64*0.35 * sec_mp
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

    mat = Pinching4(
        model.uid_generator.new('uniaxial material'),
        'auto_gravity_shear_tab',
        m1_p, th_1_p, m2_p, th_2_p,
        m3_p, th_3_p, m4_p, th_4_p,
        m1_n, th_1_n, m2_n, th_2_n,
        m3_n, th_3_n, m4_n, th_4_n,
        rdispp, rforcep, uforcep,
        rdispn, rforcen, uforcen,
        0.00, 0.00, 0.00, 0.00, gklim,
        0.00, 0.00, 0.00, 0.00, gdlim,
        0.00, 0.00, 0.00, 0.00, gflim,
        g_e, dmgtype)
    dirs = [1, 2, 3, 4, 5, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr('name', 'fix')
    mats = [fix_mat]*5 + [mat]
    return dirs, mats


def steel_w_col_pz(
        model: Model,
        section: ElasticSection,
        physical_material: PhysicalMaterial,
        pz_length: float,
        pz_doubler_plate_thickness: float,
        pz_hardening: float,
        **kwargs):
    """
    Gupta, A., & Krawinkler, H. (1999). Seismic demands for the
    performance evaluation of steel moment resisting frame
    structures. Rep. No. 132.
    """
    assert section.name[0] == 'W', \
        "Error: Only W sections can be used."
    assert isinstance(section, ElasticSection)
    assert model.settings.imperial_units, \
        "Error: Only imperial units supported."
    assert section.properties
    f_y = physical_material.f_y
    hardening = pz_hardening
    d_c = section.properties['d']
    bfc = section.properties['bf']
    t_p = section.properties['tw'] + pz_doubler_plate_thickness
    t_f = section.properties['tf']
    v_y = 0.55 * f_y * d_c * t_p
    g_mod = physical_material.g_mod
    k_e = 0.95 * g_mod * t_p * d_c
    k_p = 0.95 * g_mod * bfc * t_f**2 / pz_length
    gamma_1 = v_y / k_e
    gamma_2 = 4.0 * gamma_1
    gamma_3 = 100. * gamma_1
    m1y = gamma_1 * k_e * pz_length
    m2y = m1y + k_p * pz_length * (gamma_2 - gamma_1)
    m3y = m2y + (hardening * k_e
                 * pz_length) * (gamma_3 - gamma_2)

    # account for the fact that our panel zones have four nonlinear
    # springs
    m1y /= 4.00
    m2y /= 4.00
    m3y /= 4.00

    mat = Hysteretic(
        model.uid_generator.new('uniaxial material'),
        'auto_steel_W_PZ',
        (m1y, gamma_1),
        (m2y, gamma_2),
        (m3y, gamma_3),
        (-m1y, -gamma_1),
        (-m2y, -gamma_2),
        (-m3y, -gamma_3),
        1.00, 1.00,
        0.00, 0.00,
        0.00
    )
    dirs = [1, 2, 3, 4, 5, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr('name', 'fix')
    mats = [fix_mat]*5 + [mat]
    return dirs, mats


def steel_brace_gusset(
        model: Model,
        physical_mat: PhysicalMaterial,
        d_brace: float,
        l_c: float,
        t_p: float,
        l_b: float,
        **kwargs):
    """
    Hsiao, P-C., Lehman, D.E., and Roeder, C.W., 2012, Improved
    analysis model for special concentrically braced frames, Journal
    of Constructional Steel Research, Vol. 73, pp 80-94.
    Arguments:
      model (Model): Model object
      physical_mat (PhysicalMaterial): physical material object
      d_brace (float): brace section height
      l_c (float): brace-to-gusset connection length
      t_p (float): gusset plate thickness
      l_b (float): gusset plate average buckling length
    """
    var_w = d_brace + 2.00 * l_c * np.tan(30.00/180.00*np.pi)
    var_i = var_w * t_p**3 / 12.00
    var_z = var_w * t_p**2 / 6.00
    f_y = physical_mat.f_y
    var_e = physical_mat.e_mod
    var_g = physical_mat.g_mod
    var_my = var_z * f_y
    var_k_rot = var_e * var_i / l_b
    var_b = 0.01
    gusset_mat = Steel02(
        model.uid_generator.new('uniaxial material'),
        'auto_steel_gusset',
        var_my, var_k_rot, var_g, var_b, 20.00, 0.925, 0.15,
        0.0005, 0.014, 0.0005, 0.01, 0.00)
    dirs = [1, 2, 3, 4, 5, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr('name', 'fix')
    mats = [fix_mat]*4 + [gusset_mat, fix_mat]
    return dirs, mats
