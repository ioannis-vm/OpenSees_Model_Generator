"""
Model Generator for OpenSees ~ steel design checks
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
from typing import Optional


def smrf_scwb(
    col_sec_properties: dict[str, float],
    col_sec_properties_above: dict[str, float],
    beam_i_sec_properties: dict[str, float],
    col_axial_load: float,
    beam_udl_i: float,
    rbs_proportion_i: float,
    level_height: float,
    bay_length: float,
    beam_j_sec_properties: Optional[dict[str, float]],
    beam_udl_j: Optional[float],
    rbs_proportion_j: Optional[float],
    s_h: float,
    f_y: float,
    ry_coeff: float = 1.15,
    omega_coeff: float = 1.10,
) -> float:
    """
    SMRF strong column weak beam check
    """
    # in3
    zc_below = col_sec_properties["Zx"]
    # in2
    ac_below = col_sec_properties["A"]
    # in3
    zc_above = col_sec_properties_above["Zx"]
    # in2
    ac_above = col_sec_properties_above["A"]
    # lb-in
    mc_below = zc_below * (f_y - col_axial_load / ac_below)
    # lb-in
    mc_above = zc_above * (f_y - col_axial_load / ac_above)
    # lb
    if beam_j_sec_properties:
        max_beam_d = max(
            beam_i_sec_properties["d"], beam_j_sec_properties["d"]
        )
    else:
        max_beam_d = beam_i_sec_properties["d"]
    vc_star = (mc_below + mc_above) / ((level_height - max_beam_d))
    # lb-in
    sigma_mc_star = (mc_below + mc_above) + vc_star * max_beam_d / 2.00
    # in
    c_rbs_i = beam_i_sec_properties["bf"] * (1.0 - rbs_proportion_i) / 2.0
    # in3
    z_rbs_i = beam_i_sec_properties[
        "Zx"
    ] - 2.0 * c_rbs_i * beam_i_sec_properties["tf"] * (
        beam_i_sec_properties["d"] - beam_i_sec_properties["tf"]
    )
    # lb-in
    m_pr_i = ry_coeff * omega_coeff * f_y * z_rbs_i
    # lb
    v_e_i = (2 * m_pr_i) / (bay_length - 2.0 * s_h)
    # lb
    v_g_i = beam_udl_i * (bay_length - 2.0 * s_h) / 2.0
    d_c = col_sec_properties["d"]

    if beam_j_sec_properties:
        assert beam_udl_j is not None
        assert rbs_proportion_j is not None
        c_rbs_j = beam_j_sec_properties["bf"] * (1.0 - rbs_proportion_j) / 2.0
        z_rbs_j = beam_j_sec_properties[
            "Zx"
        ] - 2.0 * c_rbs_j * beam_j_sec_properties["tf"] * (
            beam_j_sec_properties["d"] - beam_j_sec_properties["tf"]
        )
        m_pr_j = ry_coeff * omega_coeff * f_y * z_rbs_j
        v_e_j = (2 * m_pr_j) / (bay_length - 2.0 * s_h)
        v_g_j = beam_udl_j * (bay_length - 2.0 * s_h) / 2.0

        sigm_mb_star = (
            m_pr_i
            + v_e_i * (s_h + d_c / 2.0)
            + v_g_i * (s_h + d_c / 2.0)
            + m_pr_j
            + v_e_j * (s_h + d_c / 2.0)
            - v_g_j * (s_h + d_c / 2.0)
        )
        capacity = sigma_mc_star / sigm_mb_star

    else:
        sigm_mb_star = 1.00 * (
            m_pr_i + v_e_i * (s_h + d_c / 2.0) + v_g_i * (s_h + d_c / 2.0)
        )
        capacity = sigma_mc_star / sigm_mb_star

    return capacity


def smrf_pz_doubler_plate_requirement(
    col_sec_properties: dict[str, float],
    beam_sec_properties: dict[str, float],
    rbs_proportion: float,
    bay_length: float,
    place: str,
    s_h: float,
    f_y: float,
    ry_coeff: float = 1.15,
    omega_coeff: float = 1.10,
) -> float:
    """
    Calculates the required doubler plate thickness
    """
    # in
    c_rbs = beam_sec_properties["bf"] * (1.0 - rbs_proportion) / 2.0
    # in3
    z_rbs = beam_sec_properties["Zx"] - 2.0 * c_rbs * beam_sec_properties[
        "tf"
    ] * (beam_sec_properties["d"] - beam_sec_properties["tf"])
    # lb-in
    m_pr = ry_coeff * omega_coeff * f_y * z_rbs
    # lb
    v_e = 2.0 * m_pr / (bay_length - 2.0 * s_h)
    # lb-in
    m_f = m_pr + v_e * s_h
    # lb
    r_n = (
        0.60
        * f_y
        * col_sec_properties["d"]
        * col_sec_properties["tw"]
        * (
            1.00
            + (
                3.0
                * col_sec_properties["bf"]
                * (col_sec_properties["tf"]) ** 2
            )
            / (
                col_sec_properties["d"]
                * beam_sec_properties["d"]
                * col_sec_properties["tw"]
            )
        )
    )
    if place == "interior":
        r_u = 2 * m_f / (beam_sec_properties["d"] - beam_sec_properties["tf"])
        tdoub = (r_u - r_n) / (0.60 * f_y * col_sec_properties["d"])
        tdoub = max(tdoub, 0.00)
    else:
        r_u = m_f / (beam_sec_properties["d"] - beam_sec_properties["tf"])
        tdoub = (r_u - r_n) / (0.60 * f_y * col_sec_properties["d"])
        tdoub = max(tdoub, 0.00)
    return tdoub


# def steel_W_sec_strength_check(sec_properties, loads):
#     pass
