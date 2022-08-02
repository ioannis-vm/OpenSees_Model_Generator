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
from ..ops.section import ElasticSection
from ..ops.uniaxial_material import Bilin
from ..ops.uniaxial_material import Pinching4
from ..ops.uniaxial_material import Hysteretic

if TYPE_CHECKING:
    from ..model import Model
    from ..physical_material import PhysicalMaterial


def fix_all(model: Model, **kwargs):
    """
    useful for debugging.
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
        lbry: float,
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
    assert section.name[0] == 'W', \
        "Error: Only W sections can be used."
    assert isinstance(section, ElasticSection)
    assert model.settings.imperial_units, \
        "Error: Only imperial units supported."
    assert section.properties
    # Young's modulus
    mat_e = section.e_mod / 1.e3
    # Yield stress
    mat_fy = physical_material.f_y / 1.e3
    # Moment of inertia - strong axis - original section
    sec_ix = section.i_x
    # Section depth
    sec_d = section.properties['d']
    # Flange width
    sec_bf = section.properties['bf']
    # Flange and web thicknesses
    sec_tf = section.properties['tf']
    sec_tw = section.properties['tw']
    # Plastic modulus (unreduced)
    sec_zx = section.properties['Zx']
    # Clear length
    elm_h = element_length
    # Shear span - 0.5 * elm_H typically.
    elm_l = loverh * elm_h
    if rbs_factor:
        # RBS case
        assert rbs_factor <= 1.00, 'rbs_factor must be <= 1.00'
        # checks ~ acceptable range
        if not 20.00 < sec_d/sec_tw < 55.00:
            print('Warning: sec_d/sec_tw outside regression range')
            print(section.name, '\n')
        if not 20.00 < lbry < 80.00:
            print('Warning: Lb/ry outside regression range')
            print(section.name, '\n')
        if not 4.00 < (sec_bf/(2.*sec_tf)) < 8.00:
            print('Warning: bf/(2 tf) outside regression range')
            print(section.name, '\n')
        if not 2.5 < elm_l/sec_d < 7.0:
            print('Warning: L/d  outside regression range')
            print(section.name, '\n')
        if not 4.00 < sec_d < 36.00:
            print('Warning: Section d outside regression range')
            print(section.name, '\n')
        if not 35.00 < mat_fy < 65.00:
            print('Warning: Fy outside regression range')
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
        z_rbs = sec_zx - 2. * rbs_c * sec_tf * (sec_d - sec_tf)
        sec_my = 1.06 * z_rbs * mat_fy * 1.e3
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
        sec_my = 1.17 * sec_zx * mat_fy * 1.e3
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
    my_plus = sec_my
    my_minus = -sec_my
    if consider_composite:
        # Elkady, A., & Lignos, D. G. (2014). Modeling of the
        # composite action in fully restrained beam‐to‐column
        # connections: implications in the seismic design and
        # collapse capacity of steel special moment
        # frames. Earthquake Engineering & Structural Dynamics,
        # 43(13), 1935-1954.  Table II
        theta_p_plus *= 1.80
        theta_p_minus *= 0.95
        theta_pc_plus *= 1.35
        theta_pc_minus *= 0.95
        d_plus *= 1.15
        d_minus *= 1.00
        mcmy_plus *= 1.30
        mcmy_minus *= 1.05
        my_plus *= 1.35
        my_minus *= 1.25
        residual_plus = 0.30
        residual_minus = 0.20
    stiffness = 6.00 * mat_e * sec_ix / elm_h * 1e4
    beta_plus = (mcmy_plus - 1.) * my_plus / (theta_p_plus) / stiffness
    beta_minus = - (mcmy_minus - 1.) * my_minus \
        / (theta_p_minus) / stiffness
    mat = Bilin(
        model.uid_generator.new('element'),
        'auto_IMK',
        stiffness,
        beta_plus, beta_minus,
        my_plus, my_minus,
        lamda, lamda, lamda, lamda,
        1.00, 1.00, 1.00, 1.00,
        theta_p_plus, theta_p_minus,
        theta_pc_plus, theta_pc_minus,
        residual_plus, residual_minus,
        theta_u, theta_u,
        d_plus, d_minus,
        0.00
    )
    dirs = [1, 2, 3, 4, 5, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr('name', 'fix')
    mats = [fix_mat]*5 + [mat]
    return dirs, mats


def release_5_imk_6(
        model: Model,
        element_length: float,
        lbry: float,
        loverh: float,
        rbs_factor: Optional[float],
        consider_composite: bool,
        section: ElasticSection,
        physical_material: PhysicalMaterial,
        **kwargs):
    """
    release in the weak axis bending direciton,
    imk (see imk docstring) in the strong axis bending direction
    """
    # TODO: avoid code repetition
    assert section.name[0] == 'W', \
        "Error: Only W sections can be used."
    assert isinstance(section, ElasticSection)
    assert model.settings.imperial_units, \
        "Error: Only imperial units supported."
    assert section.properties
    # Young's modulus
    mat_e = section.e_mod / 1.e3
    # Yield stress
    mat_fy = physical_material.f_y / 1.e3
    # Moment of inertia - strong axis - original section
    sec_ix = section.i_x
    # Section depth
    sec_d = section.properties['d']
    # Flange width
    sec_bf = section.properties['bf']
    # Flange and web thicknesses
    sec_tf = section.properties['tf']
    sec_tw = section.properties['tw']
    # Plastic modulus (unreduced)
    sec_zx = section.properties['Zx']
    # Clear length
    elm_h = element_length
    # Shear span - 0.5 * elm_H typically.
    elm_l = loverh * elm_h
    if rbs_factor:
        # RBS case
        assert rbs_factor <= 1.00, 'rbs_factor must be <= 1.00'
        # checks ~ acceptable range
        if not 20.00 < sec_d/sec_tw < 55.00:
            print('Warning: sec_d/sec_tw outside regression range')
            print(section.name, '\n')
        if not 20.00 < lbry < 80.00:
            print('Warning: Lb/ry outside regression range')
            print(section.name, '\n')
        if not 4.00 < (sec_bf/(2.*sec_tf)) < 8.00:
            print('Warning: bf/(2 tf) outside regression range')
            print(section.name, '\n')
        if not 2.5 < elm_l/sec_d < 7.0:
            print('Warning: L/d  outside regression range')
            print(section.name, '\n')
        if not 4.00 < sec_d < 36.00:
            print('Warning: Section d outside regression range')
            print(section.name, '\n')
        if not 35.00 < mat_fy < 65.00:
            print('Warning: Fy outside regression range')
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
        z_rbs = sec_zx - 2. * rbs_c * sec_tf * (sec_d - sec_tf)
        sec_my = 1.06 * z_rbs * mat_fy * 1.e3
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
        sec_my = 1.17 * sec_zx * mat_fy * 1.e3
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
    my_plus = sec_my
    my_minus = -sec_my
    if consider_composite:
        # Elkady, A., & Lignos, D. G. (2014). Modeling of the
        # composite action in fully restrained beam‐to‐column
        # connections: implications in the seismic design and
        # collapse capacity of steel special moment
        # frames. Earthquake Engineering & Structural Dynamics,
        # 43(13), 1935-1954.  Table II
        theta_p_plus *= 1.80
        theta_p_minus *= 0.95
        theta_pc_plus *= 1.35
        theta_pc_minus *= 0.95
        d_plus *= 1.15
        d_minus *= 1.00
        mcmy_plus *= 1.30
        mcmy_minus *= 1.05
        my_plus *= 1.35
        my_minus *= 1.25
        residual_plus = 0.30
        residual_minus = 0.20
    stiffness = 6.00 * mat_e * sec_ix / elm_h * 1e4
    beta_plus = (mcmy_plus - 1.) * my_plus / (theta_p_plus) / stiffness
    beta_minus = - (mcmy_minus - 1.) * my_minus \
        / (theta_p_minus) / stiffness
    mat = Bilin(
        model.uid_generator.new('element'),
        'auto_IMK',
        stiffness,
        beta_plus, beta_minus,
        my_plus, my_minus,
        lamda, lamda, lamda, lamda,
        1.00, 1.00, 1.00, 1.00,
        theta_p_plus, theta_p_minus,
        theta_pc_plus, theta_pc_minus,
        residual_plus, residual_minus,
        theta_u, theta_u,
        d_plus, d_minus,
        0.00
    )
    dirs = [1, 2, 3, 4, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr('name', 'fix')
    mats = [fix_mat]*4 + [mat]
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
        model.uid_generator.new('element'),
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
        model.uid_generator.new('element'),
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
