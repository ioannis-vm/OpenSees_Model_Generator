"""
Objects that generate ZeroLength elements.

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

# pylint: disable=unused-argument


from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Optional
import numpy as np
from ..ops.section import ElasticSection
from ..ops.uniaxial_material import UniaxialMaterial
from ..ops.uniaxial_material import Elastic
from ..ops.uniaxial_material import Steel02
from ..ops.uniaxial_material import Pinching4
from ..ops.uniaxial_material import Hysteretic
from ..ops.uniaxial_material import MinMax
from .material_gen import MaterialGenerator

if TYPE_CHECKING:
    from ..model import Model
    from ..physical_material import PhysicalMaterial


def fix_all(
        model: Model, **kwargs: dict[object, object]) \
        -> tuple[list[int], list[UniaxialMaterial]]:
    """
    Fixed in all directions.

    """

    dirs = [1, 2, 3, 4, 5, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr("name", "fix")
    mats = [fix_mat] * 6
    return dirs, mats


def release_6(
        model: Model, **kwargs: dict[object, object]) \
        -> tuple[list[int], list[UniaxialMaterial]]:
    """
    Frees strong axis bending.

    """

    dirs = [1, 2, 3, 4, 5]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr("name", "fix")
    mats = [fix_mat] * 5
    return dirs, mats


def release_5(
        model: Model, **kwargs: dict[object, object]) \
        -> tuple[list[int], list[UniaxialMaterial]]:
    """
    Frees weak axis bending.

    """

    dirs = [1, 2, 3, 4, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr("name", "fix")
    mats = [fix_mat] * 5
    return dirs, mats


def release_56(
        model: Model, **kwargs: dict[object, object]) \
        -> tuple[list[int], list[UniaxialMaterial]]:
    """
    Frees both strong and weak axis bending.

    """

    dirs = [1, 2, 3, 4]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr("name", "fix")
    mats = [fix_mat] * 4
    return dirs, mats


def imk_6(
        model: Model,
        element_length: float,
        lboverl: float,
        loverh: float,
        rbs_factor: Optional[float],
        consider_composite: bool,
        axial_load_ratio: float,
        section: ElasticSection,
        physical_material: PhysicalMaterial,
        **kwargs: dict[object, object]) \
        -> tuple[list[int], list[UniaxialMaterial]]:
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

    moment_modifier = kwargs.get('moment_modifier', 1.00)
    n_parameter = kwargs.get('n_parameter', 0.00)

    mat_generator = MaterialGenerator(model)
    mat = mat_generator.generate_steel_w_imk_material(
        section,
        physical_material,
        element_length,
        lboverl,
        loverh,
        rbs_factor,
        consider_composite,
        axial_load_ratio,
        direction="strong",
        moment_modifier=moment_modifier,
        n_parameter=n_parameter
    )
    dirs = [1, 2, 3, 4, 5, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr("name", "fix")
    mats = [fix_mat] * 5 + [mat]
    return dirs, mats


def imk_56(
        model: Model,
        element_length: float,
        lboverl: float,
        loverh: float,
        rbs_factor: Optional[float],
        consider_composite: bool,
        axial_load_ratio: float,
        section: ElasticSection,
        physical_material: PhysicalMaterial,
        **kwargs:  dict[object, object]) \
        -> tuple[list[int], list[UniaxialMaterial]]:
    """
    release in the weak axis bending direction,
    :func:`~osmg.gen.zerolength_gen.imk_6` in the strong axis bending
    direction

    """

    moment_modifier = kwargs.get('moment_modifier', 1.00)
    n_parameter = kwargs.get('n_parameter', 0.00)

    mat_generator = MaterialGenerator(model)
    mat_strong = mat_generator.generate_steel_w_imk_material(
        section,
        physical_material,
        element_length,
        lboverl,
        loverh,
        rbs_factor,
        consider_composite,
        axial_load_ratio,
        direction="strong",
        moment_modifier=moment_modifier,
        n_parameter=n_parameter
    )
    mat_weak = mat_generator.generate_steel_w_imk_material(
        section,
        physical_material,
        element_length,
        lboverl,
        loverh,
        rbs_factor,
        consider_composite,
        axial_load_ratio,
        direction="weak",
        moment_modifier=moment_modifier
    )
    dirs = [1, 2, 3, 4, 5, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr("name", "fix")
    mats = [fix_mat] * 4 + [mat_weak, mat_strong]
    return dirs, mats


def gravity_shear_tab(
        model: Model,
        consider_composite: bool,
        section: ElasticSection,
        physical_material: PhysicalMaterial,
        **kwargs:  dict[object, object]) \
        -> tuple[list[int], list[UniaxialMaterial]]:
    """
    Elkady, A., & Lignos, D. G. (2015). Effect of gravity framing on
    the overstrength and collapse capacity of steel frame buildings
    with perimeter special moment frames. Earthquake Engineering &
    Structural Dynamics, 44(8), 1289-1307.

    """

    assert section.name[0] == "W", "Error: Only W sections can be used."
    assert isinstance(section, ElasticSection)
    assert (
        model.settings.imperial_units
    ), "Error: Only imperial units supported."
    assert section.properties

    moment_modifier = kwargs.get('moment_modifier', 1.00)

    # Yield stress
    assert isinstance(moment_modifier, float)
    mat_fy = physical_material.f_y / 1.0e3
    # Plastic modulus (unreduced)
    sec_zx = section.properties["Zx"]
    # Plastic moment of the section
    sec_mp = sec_zx * mat_fy * 1.0e3 * moment_modifier

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
        dmgtype = "energy"
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
        dmgtype = "energy"

    mat = Pinching4(
        model.uid_generator.new("uniaxial material"),
        "auto_gravity_shear_tab",
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
    dirs = [1, 2, 3, 4, 5, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr("name", "fix")
    release_mat = mat_repo.retrieve_by_attr("name", "release")
    mats = [fix_mat] * 4 + [release_mat] + [mat]
    return dirs, mats


def steel_w_col_pz(
        model: Model,
        section: ElasticSection,
        physical_material: PhysicalMaterial,
        pz_length: float,
        pz_doubler_plate_thickness: float,
        pz_hardening: float,
        only_elastic: bool = False,
        moment_modifier: float = 1.00,
        **kwargs:  dict[object, object]) \
        -> tuple[list[int], list[UniaxialMaterial]]:
    """
    Gupta, A., & Krawinkler, H. (1999). Seismic demands for the
    performance evaluation of steel moment resisting frame
    structures. Rep. No. 132.

    """

    assert section.name[0] == "W", "Error: Only W sections can be used."
    assert isinstance(section, ElasticSection)
    assert (
        model.settings.imperial_units
    ), "Error: Only imperial units supported."
    assert section.properties
    f_y = physical_material.f_y
    hardening = pz_hardening
    d_c = section.properties["d"]
    bfc = section.properties["bf"]
    t_p = section.properties["tw"] + pz_doubler_plate_thickness
    t_f = section.properties["tf"]
    v_y = 0.55 * f_y * d_c * t_p
    g_mod = physical_material.g_mod
    k_e = 0.95 * g_mod * t_p * d_c
    k_p = 0.95 * g_mod * bfc * t_f**2 / pz_length
    gamma_1 = v_y / k_e
    gamma_2 = 4.0 * gamma_1
    gamma_3 = 100.0 * gamma_1
    m1y = (gamma_1 * k_e * pz_length
           * moment_modifier)
    m2y = (m1y + k_p * pz_length * (gamma_2 - gamma_1)
           * moment_modifier)
    m3y = (m2y + (hardening * k_e * pz_length) * (gamma_3 - gamma_2)
           * moment_modifier)

    if only_elastic:
        mat: UniaxialMaterial = Elastic(
            model.uid_generator.new("uniaxial material"),
            "auto_steel_W_PZ",
            m1y/gamma_1
        )
    else:
        mat = Hysteretic(
            model.uid_generator.new("uniaxial material"),
            "auto_steel_W_PZ",
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
    dirs = [1, 2, 3, 4, 5, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr("name", "fix")
    mats = [fix_mat] * 5 + [mat]
    return dirs, mats


def steel_w_col_pz_updated(
    model: Model,
    section: ElasticSection,
    physical_material: PhysicalMaterial,
    pz_length: float,
    pz_doubler_plate_thickness: float,
    axial_load_ratio: float,
    slab_depth: float,
    consider_composite: bool,
    location: str,
    only_elastic: bool = False,
    moment_modifier: float = 1.00,
    **kwargs:  dict[object, object]) \
        -> tuple[list[int], list[UniaxialMaterial]]:
    """
    Skiadopoulos, A., Elkady, A. and D. G. Lignos (2020). "Proposed
    Panel Zone Model for Seismic Design of Steel Moment-Resisting
    Frames." ASCE Journal of Structural Engineering. DOI:
    10.1061/(ASCE)ST.1943-541X.0002935.

    """

    assert section.name[0] == "W", "Error: Only W sections can be used."
    assert isinstance(section, ElasticSection)
    assert (
        model.settings.imperial_units
    ), "Error: Only imperial units supported."
    assert section.properties
    f_y = physical_material.f_y
    e_mod = physical_material.e_mod
    g_mod = physical_material.g_mod
    tw_Col = section.properties["tw"]
    tdp = pz_doubler_plate_thickness
    d_Col = section.properties["d"]
    d_Beam = pz_length
    tf_Col = section.properties["tf"]
    bf_Col = section.properties["bf"]
    Ix_Col = section.properties["Ix"]
    ts = slab_depth
    n = axial_load_ratio
    trib = slab_depth

    tpz = tw_Col + tdp  # total PZ thickness
    # effective depth in positive moment
    d_BeamP = d_Beam
    if consider_composite:
        d_BeamP = d_Beam + trib + 0.50 * ts
    # effective depth in negative moment
    d_BeamN = d_Beam

    # Stiffness Calculation
    Ks = tpz * (d_Col - tf_Col) * g_mod
    Kb = (12.0 * e_mod * (Ix_Col + tdp *
          ((d_Col - 2.0 * tf_Col)**3) / 12.00) / (d_Beam**3) * d_Beam)
    Ke = Ks * Kb / (Ks + Kb)

    # flange stiffness: shear contribution
    Ksf = 2.0 * (bf_Col * tf_Col) * g_mod
    # flange stiffness: bending contribution
    Kbf = (2.0 * 12.0 * e_mod * bf_Col
           * (tf_Col**3) / 12.0 / (d_Beam**3) * d_Beam)
    # flange stiffness: total contribution
    Kef = (Ksf * Kbf) / (Ksf + Kbf)

    ay = (0.58 * Kef / Ke + 0.88) / (1.0 - Kef / Ke)

    aw_eff_4gamma = 1.10
    aw_eff_6gamma = 1.15

    af_eff_4gamma = 0.93 * Kef / Ke + 0.015
    af_eff_6gamma = 1.05 * Kef / Ke + 0.020
    # reduction factor accounting for axial load
    r = np.sqrt(1.0 - (n**2))

    Vy = r * 0.577 * f_y * ay * (d_Col - tf_Col) * tpz
    # Plastic Shear Force at 4 gammaY
    Vp_4gamma = (
        r * 0.577 * f_y * (
            aw_eff_4gamma * (d_Col - tf_Col)
            * tpz + af_eff_4gamma * (bf_Col - tw_Col) * 2*tf_Col))
    # Plastic Shear Force at 6 gammaY
    Vp_6gamma = (
        r * 0.577 * f_y * (
            aw_eff_6gamma * (d_Col - tf_Col)
            * tpz + af_eff_6gamma * (bf_Col - tw_Col) * 2*tf_Col))

    gamma_y = Vy/Ke
    gamma4_y = 4.0 * gamma_y

    My_P = Vy * d_BeamP
    Mp_4gamma_P = Vp_4gamma * d_BeamP
    Mp_6gamma_P = Vp_6gamma * d_BeamP

    My_N = Vy * d_BeamN
    Mp_4gamma_N = Vp_4gamma * d_BeamN
    Mp_6gamma_N = Vp_6gamma * d_BeamN

    Slope_4to6gamma_y_P = (Mp_6gamma_P - Mp_4gamma_P) / (2.0 * gamma_y)
    Slope_4to6gamma_y_N = (Mp_6gamma_N - Mp_4gamma_N) / (2.0 * gamma_y)

    # Defining the 3 Points used to construct the trilinear backbone curve
    gamma1 = gamma_y
    gamma2 = gamma4_y
    gamma3 = 100.0 * gamma_y

    M1_P = My_P
    M2_P = Mp_4gamma_P
    M3_P = Mp_4gamma_P + Slope_4to6gamma_y_P * (100 * gamma_y - gamma4_y)

    M1_N = My_N
    M2_N = Mp_4gamma_N
    M3_N = Mp_4gamma_N + Slope_4to6gamma_y_N * (100 * gamma_y - gamma4_y)

    # apply moment modifier
    M1_P *= moment_modifier
    M2_P *= moment_modifier
    M3_P *= moment_modifier
    M1_N *= moment_modifier
    M2_N *= moment_modifier
    M3_N *= moment_modifier

    gammaU_P = 0.3
    gammaU_N = -0.3

    if not consider_composite:
        args = ((M1_N, gamma1),
                (M2_N, gamma2),
                (M3_N, gamma3),
                (-M1_N, -gamma1),
                (-M2_N, -gamma2),
                (-M3_N, -gamma3),
                0.25, 0.75, 0.0, 0.0, 0.0)
    elif location == 'interior':
        args = ((M1_P, gamma1),
                (M2_P, gamma2),
                (M3_P, gamma3),
                (-M1_P, -gamma1),
                (-M2_P, -gamma2),
                (-M3_P, -gamma3),
                0.25, 0.75, 0.0, 0.0, 0.0)
    elif location == 'exterior_first':
        args = ((M1_N, gamma1),
                (M2_N, gamma2),
                (M3_N, gamma3),
                (-M1_P, -gamma1),
                (-M2_P, -gamma2),
                (-M3_P, -gamma3),
                0.25, 0.75, 0.0, 0.0, 0.0)
    elif location == 'exterior_last':
        args = ((M1_P, gamma1),
                (M2_P, gamma2),
                (M3_P, gamma3),
                (-M1_N, -gamma1),
                (-M2_N, -gamma2),
                (-M3_N, -gamma3),
                0.25, 0.75, 0.0, 0.0, 0.0)
    else:
        raise ValueError(f'Invalid Location: {location}')

    if only_elastic:
        m1y, gamma_1 = args[0]  # type: ignore
        mat: UniaxialMaterial = Elastic(
            model.uid_generator.new("uniaxial material"),
            "auto_steel_W_PZ",
            m1y/gamma_1  # type: ignore
        )
    else:
        mat = Hysteretic(
            model.uid_generator.new("uniaxial material"),
            "auto_steel_W_pz_updated",
            *args
        )
    minmaxmat = MinMax(
        model.uid_generator.new("uniaxial material"),
        "auto_steel_W_pz_updated_minmax",
        mat, gammaU_N, gammaU_P)
    dirs = [1, 2, 3, 4, 5, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr("name", "fix")
    mats = [fix_mat] * 5 + [minmaxmat]
    return dirs, mats


def steel_brace_gusset(
        model: Model,
        physical_mat: PhysicalMaterial,
        d_brace: float,
        l_c: float,
        t_p: float,
        l_b: float,
        **kwargs:  dict[object, object]) \
        -> tuple[list[int], list[UniaxialMaterial]]:
    """
    Hsiao, P-C., Lehman, D.E., and Roeder, C.W., 2012, Improved
    analysis model for special concentrically braced frames, Journal
    of Constructional Steel Research, Vol. 73, pp 80-94.

    Arguments:
      model: Model object
      physical_mat: physical material object
      d_brace: brace section height
      l_c: brace-to-gusset connection length
      t_p: gusset plate thickness
      l_b: gusset plate average buckling length

    """

    var_w = d_brace + 2.00 * l_c * np.tan(30.00 / 180.00 * np.pi)
    var_i = var_w * t_p**3 / 12.00
    var_z = var_w * t_p**2 / 6.00
    f_y = physical_mat.f_y
    var_e = physical_mat.e_mod
    var_g = physical_mat.g_mod
    var_my = var_z * f_y
    var_k_rot = var_e * var_i / l_b
    var_b = 0.01
    gusset_mat = Steel02(
        model.uid_generator.new("uniaxial material"),
        "auto_steel_gusset",
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
    dirs = [1, 2, 3, 4, 5, 6]
    mat_repo = model.uniaxial_materials
    fix_mat = mat_repo.retrieve_by_attr("name", "fix")
    mats = [fix_mat] * 4 + [gusset_mat, fix_mat]
    return dirs, mats
