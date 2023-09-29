"""
This module generates the nonlinear time-history analysis models of
the archetypes considerd in the study.
"""

from copy import deepcopy
from osmg.model import Model
from osmg.gen.component_gen import BeamColumnGenerator
from osmg.gen.section_gen import SectionGenerator
from osmg.gen.steel.brb import BRBGenerator
from osmg.gen.material_gen import MaterialGenerator
from osmg import defaults
from osmg.preprocessing.self_weight_mass import self_weight
from osmg.preprocessing.self_weight_mass import self_mass
from osmg.ops.section import ElasticSection
from osmg.ops.section import FiberSection
from osmg.ops.element import ElasticBeamColumn
from osmg.ops.element import TwoNodeLink
from osmg.ops.element import DispBeamColumn
from osmg.gen.zerolength_gen import release_56
from osmg.gen.zerolength_gen import imk_56
from osmg.gen.zerolength_gen import imk_6
from osmg.gen.zerolength_gen import gravity_shear_tab
from osmg.gen.zerolength_gen import steel_brace_gusset
from osmg.load_case import LoadCase
from osmg.preprocessing.tributary_area_analysis import PolygonLoad
from osmg.common import G_CONST_IMPERIAL
import numpy as np
import pandas as pd  # type: ignore
import scipy as sp  # type: ignore


# pylint:disable=too-many-locals
# pylint:disable=too-many-branches
# pylint:disable=too-many-statements
# pylint:disable=too-many-arguments
# pylint:disable=too-many-lines
# pylint:disable=consider-using-enumerate
# pylint:disable=use-dict-literal


def generate_archetype(
    level_elevs,
    sections,
    metadata,
    surf_loads,
    surf_loads_massless,
    lateral_system,
    risk_category,
):
    """
    Generates an archetype building
    """

    n_parameter = 10.00

    interp_smf = None  # stiffness modification factor for BRBFs

    assert lateral_system in {"smrf", "scbf", "brbf"}
    mdl = Model("model")
    bcg = BeamColumnGenerator(mdl)
    secg = SectionGenerator(mdl)
    mtlg = MaterialGenerator(mdl)

    num_levels = len(level_elevs)

    mdl.add_level(0, 0.00)
    for i, height in enumerate(level_elevs):
        mdl.add_level(i + 1, height)

    hi_diff = np.diff(np.array((0.00, *level_elevs)))

    defaults.load_default_steel(mdl)
    defaults.load_default_fix_release(mdl)
    defaults.load_util_rigid_elastic(mdl)

    steel_phys_mat = mdl.physical_materials.retrieve_by_attr("name", "default steel")

    # define sections
    wsections = set()
    if risk_category == "iv":
        frame_locs = ("outer_frame", "inner_frame")
    else:
        frame_locs = ("outer_frame",)
    for lvl_tag in [f"level_{i+1}" for i in range(num_levels)]:
        wsections.add(sections["gravity_beams_a"][lvl_tag])
        wsections.add(sections["gravity_beams_b"][lvl_tag])
        wsections.add(sections["gravity_cols"][lvl_tag])
    if lateral_system == "smrf":
        for frame_loc in frame_locs:
            for lvl_tag in [f"level_{i+1}" for i in range(num_levels)]:
                wsections.add(sections[frame_loc]["lateral_beams"][lvl_tag])
        for function in ["exterior", "interior"]:
            for lvl_tag in [f"level_{i+1}" for i in range(num_levels)]:
                for frame_loc in frame_locs:
                    wsections.add(
                        sections[frame_loc]["lateral_cols"][function][lvl_tag]
                    )

    elif lateral_system in {"scbf", "brbf"}:
        for lvl_tag in [f"level_{i+1}" for i in range(num_levels)]:
            wsections.add(sections["lateral_cols"][lvl_tag])
            wsections.add(sections["lateral_beams"][lvl_tag])
    else:
        raise ValueError("Invalid lateral system")

    section_type = ElasticSection
    element_type = ElasticBeamColumn
    sec_collection = mdl.elastic_sections

    for sec in wsections:
        secg.load_aisc_from_database(
            "W", [sec], "default steel", "default steel", section_type
        )

    if lateral_system == "scbf":
        hss_secs = set()
        for lvl_tag in [f"level_{i+1}" for i in range(num_levels)]:
            hss_secs.add(sections["braces"][lvl_tag])
        for sec in hss_secs:
            secg.load_aisc_from_database(
                "HSS_circ", [sec], "default steel", "default steel", FiberSection
            )

    if lateral_system == "brbf":
        # load the stiffness modification factor (smf) data
        df_smf = pd.read_csv(
            "brbf_stiffness_modification_factors.csv",
            skiprows=6,
            index_col=(0),
        )
        df_smf = (
            df_smf.assign(
                WorkPtLen_ft=np.sqrt(df_smf.Bay_ft**2 + df_smf.Height_ft**2)
            )
            .drop(columns=["Bay_ft", "Height_ft"])
            .reset_index()
            .set_index(["Asc_in2", "WorkPtLen_ft"])
        )

        # cast to numpy arrays
        points_smf = np.array(df_smf.index.to_list()).tolist()
        values_smf = df_smf.to_numpy().reshape(-1)
        # generate interpolation function
        interp_smf = sp.interpolate.LinearNDInterpolator(points_smf, values_smf)

        df_acs = pd.read_csv(
            "brbf_approximate_casing_sizes.csv",
            skiprows=6,
            index_col=(0),
        )
        df_acs = (
            df_acs.assign(
                WorkPtLen_ft=np.sqrt(df_acs.Bay_ft**2 + df_acs.Height_ft**2)
            )
            .drop(columns=["Bay_ft", "Height_ft"])
            .reset_index()
            .set_index(["Asc_in2", "WorkPtLen_ft"])
        )

        # cast to numpy arrays
        points_acs = np.array(df_acs.index.to_list()).tolist()
        values_acs = np.array(
            [
                float(x.replace("t", "").replace("p", ""))
                for x in df_acs.to_numpy().reshape(-1)
            ]
        )
        # yes, we will draw everything as squares for our
        # visualization purposes..
        # generate interpolation function
        interp_acs = sp.interpolate.LinearNDInterpolator(points_acs, values_acs)

    # generate a dictionary containing coordinates given gridline tag names
    point = {}
    x_grd_tags = ["A", "B", "C", "D", "E", "F", "G"]
    y_grd_tags = ["5", "4", "3", "2", "1"]
    x_grd_locs = (
        np.array([0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0]) * 12.00 + 10.00
    )  # (in)
    y_grd_locs = np.array([0.0, 25.0, 50.0, 75.0, 100.0]) * 12.00 + 10.00  # (in)

    n_sub = 1  # linear elastic element subdivision

    for i in range(len(x_grd_tags)):
        point[x_grd_tags[i]] = {}
        for j in range(len(y_grd_tags)):
            point[x_grd_tags[i]][y_grd_tags[j]] = np.array(
                [x_grd_locs[i], y_grd_locs[j]]
            )

    col_gtransf = "Corotational"

    # component definition

    if lateral_system in {"scbf", "brbf"}:
        plate_a = metadata["plate_a"]
        plate_b = metadata["plate_b"]

        if lateral_system == "scbf":
            brace_subdiv = 8
            brace_lens = metadata["brace_buckling_length"]
            brace_l_c = metadata["brace_l_c"]
            gusset_t_p = metadata["gusset_t_p"]
            gusset_avg_buckl_len = metadata["gusset_avg_buckl_len"]
            hinge_dist = metadata["hinge_dist"]

    for level_counter in range(num_levels):
        level_tag = "level_" + str(level_counter + 1)
        mdl.levels.set_active([level_counter + 1])

        # define gravity columns
        sec = sec_collection.retrieve_by_attr(
            "name", sections["gravity_cols"][level_tag]
        )

        placement_df = pd.read_excel(
            f"{lateral_system}_{risk_category}_member_placement.xls",
            dtype={"x_tag": str, "y_tag": str, "ang": float},
            sheet_name="grv_cols",
        )

        for _, row in placement_df.iterrows():
            tag1, tag2, ang = row
            plcmt_pt = point[tag1][tag2]
            bcg.add_vertical_active(
                plcmt_pt[0],
                plcmt_pt[1],
                np.zeros(3),
                np.zeros(3),
                col_gtransf,
                n_sub,
                sec,
                element_type,
                "centroid",
                ang,
                method="generate_hinged_component_assembly",
                additional_args={
                    "zerolength_gen_i": None,
                    "zerolength_gen_args_i": {},
                    "zerolength_gen_j": release_56,
                    "zerolength_gen_args_j": {
                        "distance": sec.properties["d"] / 24.00,
                        "n_sub": 1,
                    },
                },
            )

    # define frame columns
    for level_counter in range(num_levels):
        level_tag = "level_" + str(level_counter + 1)
        if level_counter % 2 == 0:
            even_story_num = False  # (odd because of zero-indexing)
        else:
            even_story_num = True
        mdl.levels.set_active([level_counter + 1])

        if lateral_system == "smrf":
            placement_df = pd.read_excel(
                f"{lateral_system}_{risk_category}_member_placement.xls",
                dtype={
                    "x_tag": str,
                    "y_tag": str,
                    "ang": float,
                    "frame": str,
                    "col": str,
                    "panel_zone_odd": bool,
                    "panel_zone_even": bool,
                },
                sheet_name="lat_cols",
            )
        else:
            placement_df = pd.read_excel(
                f"{lateral_system}_{risk_category}_member_placement.xls",
                dtype={
                    "x_tag": str,
                    "y_tag": str,
                    "ang": float,
                    "panel_zone_odd": bool,
                    "panel_zone_even": bool,
                },
                sheet_name="lat_cols",
            )

        for _, row in placement_df.iterrows():
            plcmt_pt = point[tag1][tag2]
            if lateral_system == "smrf":
                tag1, tag2, ang, frame, col, panel_zone_odd, panel_zone_even = row
                plcmt_pt = point[tag1][tag2]
                sec = sec_collection.retrieve_by_attr(
                    "name", sections[frame]["lateral_cols"][col][level_tag]
                )
                sec_cp = deepcopy(sec)
                sec_cp.i_x *= (n_parameter + 1) / n_parameter
                column_depth = sec.properties["d"]
                beam_depth = sec_collection.retrieve_by_attr(
                    "name", sections[frame]["lateral_beams"][level_tag]
                ).properties["d"]
                doubler = metadata[frame][col][level_tag]
            else:
                tag1, tag2, ang, panel_zone_odd, panel_zone_even = row
                plcmt_pt = point[tag1][tag2]
                sec = sec_collection.retrieve_by_attr(
                    "name", sections["lateral_cols"][level_tag]
                )
                sec_cp = deepcopy(sec)
                sec_cp.i_x *= (n_parameter + 1) / n_parameter
                column_depth = sec.properties["d"]
                beam_depth = sec_collection.retrieve_by_attr(
                    "name", sections["lateral_beams"][level_tag]
                ).properties["d"]
                doubler = 0.00
            if even_story_num and panel_zone_even:
                add_pz = True
            elif (not even_story_num) and panel_zone_odd:
                add_pz = True
            else:
                add_pz = False
            if lateral_system != "smrf":
                if add_pz:
                    top_offset = 0.00
                    bot_offset = +plate_b[level_counter + 1]
                else:
                    top_offset = -beam_depth - plate_b[level_counter + 1]
                    bot_offset = 0.00
            else:
                top_offset = 0.00
                bot_offset = 0.00

            if add_pz:
                bcg.add_pz_active(
                    plcmt_pt[0],
                    plcmt_pt[1],
                    sec,
                    steel_phys_mat,
                    ang,
                    column_depth,
                    beam_depth,
                    "steel_w_col_pz_updated",
                    {
                        "pz_doubler_plate_thickness": doubler,
                        "axial_load_ratio": 0.00,
                        "slab_depth": 4.00,
                        "consider_composite": True,
                        "location": "interior",
                        "only_elastic": False,
                        "moment_modifier": 1.00,
                    },
                )
            bcg.add_vertical_active(
                plcmt_pt[0],
                plcmt_pt[1],
                np.array((0.00, 0.00, top_offset)),
                np.array((0.00, 0.00, bot_offset)),
                col_gtransf,
                n_sub,
                sec_cp,
                element_type,
                "centroid",
                ang,
                method="generate_hinged_component_assembly",
                additional_args={
                    "zerolength_gen_i": imk_6,
                    "zerolength_gen_args_i": {
                        "lboverl": 1.00,
                        "loverh": 0.50,
                        "rbs_factor": None,
                        "consider_composite": False,
                        "axial_load_ratio": 0.00,
                        "section": sec,
                        "n_parameter": n_parameter,
                        "physical_material": steel_phys_mat,
                        "distance": sec.properties["d"] / 24.00,
                        "n_sub": 1,
                        "element_type": TwoNodeLink,
                    },
                    "zerolength_gen_j": imk_56,
                    "zerolength_gen_args_j": {
                        "lboverl": 1.00,
                        "loverh": 0.50,
                        "rbs_factor": None,
                        "consider_composite": False,
                        "axial_load_ratio": 0.00,
                        "section": sec,
                        "n_parameter": n_parameter,
                        "physical_material": steel_phys_mat,
                        "distance": sec.properties["d"] / 24.00,
                        "n_sub": 1,
                        "element_type": TwoNodeLink,
                    },
                },
            )

    # define frame beams
    for level_counter in range(num_levels):
        level_tag = "level_" + str(level_counter + 1)
        if level_counter % 2 == 0:
            even_story_num = False  # (odd because of zero-indexing)
        else:
            even_story_num = True
        mdl.levels.set_active([level_counter + 1])

        placement_df = pd.read_excel(
            f"{lateral_system}_{risk_category}_member_placement.xls",
            dtype={
                "xi_tag": str,
                "yi_tag": str,
                "xj_tag": str,
                "yj_tag": str,
                "snap_i_even": str,
                "snap_j_even": str,
                "snap_i_odd": str,
                "snap_j_odd": str,
                "frame": str,
            },
            sheet_name="lat_bms",
        )

        for _, row in placement_df.iterrows():
            if lateral_system == "smrf":
                (
                    tag1_i,
                    tag2_i,
                    tag1_j,
                    tag2_j,
                    snap_i_even,
                    snap_j_even,
                    snap_i_odd,
                    snap_j_odd,
                    frame,
                ) = row
                sec = sec_collection.retrieve_by_attr(
                    "name", sections[frame]["lateral_beams"][level_tag]
                )
            else:
                (
                    tag1_i,
                    tag2_i,
                    tag1_j,
                    tag2_j,
                    snap_i_even,
                    snap_j_even,
                    snap_i_odd,
                    snap_j_odd,
                ) = row
                sec = sec_collection.retrieve_by_attr(
                    "name", sections["lateral_beams"][level_tag]
                )
            sec_cp = deepcopy(sec)
            sec_cp.i_x *= (n_parameter + 1) / n_parameter
            if even_story_num:
                snap_i = snap_i_even
                snap_j = snap_j_even
            else:
                snap_i = snap_i_odd
                snap_j = snap_j_odd
            if snap_i in ["middle_back", "middle_front"]:
                h_offset_i = 0.00
            else:
                h_offset_i = 0.75 * plate_a[level_counter + 1]
            if snap_j in ["middle_back", "middle_front"]:
                h_offset_j = 0.00
            else:
                h_offset_j = 0.75 * plate_a[level_counter + 1]
            bcg.add_horizontal_active(
                point[tag1_i][tag2_i][0],
                point[tag1_i][tag2_i][1],
                point[tag1_j][tag2_j][0],
                point[tag1_j][tag2_j][1],
                np.array((0.0, 0.0, 0.0)),
                np.array((0.0, 0.0, 0.0)),
                snap_i,
                snap_j,
                "Corotational",
                n_sub,
                sec_cp,
                element_type,
                "top_center",
                h_offset_i=h_offset_i,
                h_offset_j=h_offset_j,
                method="generate_hinged_component_assembly",
                additional_args={
                    "zerolength_gen_i": imk_6,
                    "zerolength_gen_args_i": {
                        "lboverl": 0.75,
                        "loverh": 0.50,
                        "rbs_factor": None,
                        "consider_composite": True,
                        "axial_load_ratio": 0.00,
                        "section": sec,
                        "n_parameter": n_parameter,
                        "physical_material": steel_phys_mat,
                        "distance": 0.01,
                        "n_sub": 1,
                        "element_type": TwoNodeLink,
                    },
                    "zerolength_gen_j": imk_6,
                    "zerolength_gen_args_j": {
                        "lboverl": 0.75,
                        "loverh": 0.50,
                        "rbs_factor": None,
                        "consider_composite": True,
                        "axial_load_ratio": 0.00,
                        "section": sec,
                        "n_parameter": n_parameter,
                        "physical_material": steel_phys_mat,
                        "distance": 0.01,
                        "n_sub": 1,
                        "element_type": TwoNodeLink,
                    },
                },
            )

    # define gravity beams
    for level_counter in range(num_levels):
        level_tag = "level_" + str(level_counter + 1)
        if level_counter % 2 == 0:
            even_story_num = False  # (odd because of zero-indexing)
        else:
            even_story_num = True
        mdl.levels.set_active([level_counter + 1])

        placement_df = pd.read_excel(
            f"{lateral_system}_{risk_category}_member_placement.xls",
            dtype={
                "xi_tag": str,
                "yi_tag": str,
                "xj_tag": str,
                "yj_tag": str,
                "snap_i_even": str,
                "snap_j_even": str,
                "snap_i_odd": str,
                "snap_j_odd": str,
                "designation": str,
            },
            sheet_name="grv_bms",
        )

        for _, row in placement_df.iterrows():
            (
                tag1_i,
                tag2_i,
                tag1_j,
                tag2_j,
                snap_i_even,
                snap_j_even,
                snap_i_odd,
                snap_j_odd,
                designation,
            ) = row

            sec = sec_collection.retrieve_by_attr(
                "name", sections[designation][level_tag]
            )
            sec_cp = deepcopy(sec)
            sec_cp.i_x *= (n_parameter + 1) / n_parameter
            if even_story_num:
                snap_i = snap_i_even
                snap_j = snap_j_even
            else:
                snap_i = snap_i_odd
                snap_j = snap_j_odd
            bcg.add_horizontal_active(
                point[tag1_i][tag2_i][0],
                point[tag1_i][tag2_i][1],
                point[tag1_j][tag2_j][0],
                point[tag1_j][tag2_j][1],
                np.array((0.0, 0.0, 0.0)),
                np.array((0.0, 0.0, 0.0)),
                snap_i,
                snap_j,
                "Corotational",
                n_sub,
                sec_cp,
                element_type,
                "top_center",
                method="generate_hinged_component_assembly",
                additional_args={
                    "zerolength_gen_i": gravity_shear_tab,
                    "zerolength_gen_args_i": {
                        "consider_composite": True,
                        "section": sec,
                        "n_parameter": n_parameter,
                        "physical_material": steel_phys_mat,
                        "distance": 0.01,
                        "n_sub": 1,
                        "element_type": TwoNodeLink,
                    },
                    "zerolength_gen_j": gravity_shear_tab,
                    "zerolength_gen_args_j": {
                        "consider_composite": True,
                        "section": sec,
                        "n_parameter": n_parameter,
                        "physical_material": steel_phys_mat,
                        "distance": 0.01,
                        "n_sub": 1,
                        "element_type": TwoNodeLink,
                    },
                },
            )

    if lateral_system in {"scbf", "brbf"}:
        sec = sec_collection.retrieve_by_attr(
            "name", sections["lateral_beams"]["level_1"]
        )
        vertical_offsets = [-sec.properties["d"] / 2.00]
        for level_counter in range(num_levels):
            level_tag = f"level_{level_counter+1}"
            sec = sec_collection.retrieve_by_attr(
                "name", sections["lateral_beams"][f"level_{level_counter+1}"]
            )
            vertical_offsets.append(-sec.properties["d"] / 2.00)

        for level_counter in range(num_levels):
            level_tag = "level_" + str(level_counter + 1)
            if level_counter % 2 == 0:
                even_story_num = False  # (odd because of zero-indexing)
            else:
                even_story_num = True
            mdl.levels.set_active([level_counter + 1])
            brace_sec_name = sections["braces"][level_tag]

            placement_df = pd.read_excel(
                f"{lateral_system}_{risk_category}_member_placement.xls",
                dtype={
                    "xi_tag": str,
                    "yi_tag": str,
                    "xj_tag": str,
                    "yj_tag": str,
                    "snap_i_even": str,
                    "snap_j_even": str,
                    "snap_i_odd": str,
                    "snap_j_odd": str,
                    "designation": str,
                },
                sheet_name="braces",
            )

            for _, row in placement_df.iterrows():
                (tag1_i, tag2_i, tag1_j, tag2_j) = row

                if not even_story_num:
                    t1i = tag1_j
                    t1j = tag1_i
                    t2i = tag2_j
                    t2j = tag2_i
                else:
                    t1i = tag1_i
                    t1j = tag1_j
                    t2i = tag2_i
                    t2j = tag2_j

                if lateral_system == "scbf":
                    brace_sec = mdl.fiber_sections.retrieve_by_attr(
                        "name", brace_sec_name
                    )
                    brace_phys_mat = deepcopy(steel_phys_mat)
                    brace_phys_mat.f_y = 50.4 * 1000.00  # for round HSS
                    brace_mat = mtlg.generate_steel_hss_circ_brace_fatigue_mat(
                        brace_sec, brace_phys_mat, brace_lens[level_counter + 1]
                    )

                    bsec = brace_sec.copy_alter_material(
                        brace_mat, mdl.uid_generator.new("section")
                    )

                    bcg.add_diagonal_active(
                        point[t1i][t2i][0],
                        point[t1i][t2i][1],
                        point[t1j][t2j][0],
                        point[t1j][t2j][1],
                        np.array((0.00, 0.00, vertical_offsets[level_counter])),
                        np.array((0.00, 0.00, vertical_offsets[level_counter])),
                        "centroid",
                        "centroid",
                        "Corotational",
                        brace_subdiv,
                        bsec,
                        DispBeamColumn,
                        "centroid",
                        0.00,
                        0.00,
                        0.1 / 100.00,
                        None,
                        None,
                        "generate_hinged_component_assembly",
                        {
                            "zerolength_gen_i": steel_brace_gusset,
                            "zerolength_gen_args_i": {
                                "distance": hinge_dist[level_counter + 1],
                                "element_type": TwoNodeLink,
                                "physical_mat": steel_phys_mat,
                                "d_brace": bsec.properties["OD"],
                                "l_c": brace_l_c[level_counter + 1],
                                "t_p": gusset_t_p[level_counter + 1],
                                "l_b": gusset_avg_buckl_len[level_counter + 1],
                            },
                            "zerolength_gen_j": steel_brace_gusset,
                            "zerolength_gen_args_j": {
                                "distance": hinge_dist[level_counter + 1],
                                "element_type": TwoNodeLink,
                                "physical_mat": steel_phys_mat,
                                "d_brace": bsec.properties["OD"],
                                "l_c": brace_l_c[level_counter + 1],
                                "t_p": gusset_t_p[level_counter + 1],
                                "l_b": gusset_avg_buckl_len[level_counter + 1],
                            },
                        },
                    )

                elif lateral_system == "brbf":
                    brbg = BRBGenerator(mdl)
                    brace_sec_name = sections["braces"][level_tag]
                    area = float(brace_sec_name)
                    workpoint_length = np.sqrt(
                        (25.00 * 12.00) ** 2 + (hi_diff[level_counter]) ** 2
                    )  # in
                    trial_point = np.array((area, workpoint_length / 12.00))
                    stiffness_mod_factor = interp_smf(trial_point)[0]  # type: ignore
                    casing_size = interp_acs(trial_point)[0]  # type: ignore

                    brbg.add_brb(
                        point[t1i][t2i][0],
                        point[t1i][t2i][1],
                        level_counter + 1,
                        np.array((0.00, 0.00, vertical_offsets[level_counter])),
                        "centroid",
                        point[t1j][t2j][0],
                        point[t1j][t2j][1],
                        level_counter,
                        np.array((0.00, 0.00, vertical_offsets[level_counter])),
                        "centroid",
                        area,
                        38000.00 * 1.10,
                        29000 * 1e3 * stiffness_mod_factor,
                        casing_size,
                        150.00 / (12.00) ** 3,  # lb/in3, approximate brb weight
                    )

    # fix base
    for node in mdl.levels[0].nodes.values():
        node.restraint = [True] * 6

    # ~~~~~~~~~~~~ #
    # assign loads #
    # ~~~~~~~~~~~~ #

    loadcase = LoadCase("1.2D+0.25L+-E", mdl)
    self_weight(mdl, loadcase, factor=1.20)
    self_mass(mdl, loadcase)

    # surface loads
    for key in range(1, 1 + num_levels):
        loadcase.tributary_area_analysis[key].polygon_loads.append(
            PolygonLoad("dead", surf_loads[key], None, None, False)
        )
        loadcase.tributary_area_analysis[key].polygon_loads.append(
            PolygonLoad("dead", surf_loads_massless[key], None, None, True)
        )
        loadcase.tributary_area_analysis[key].run(
            load_factor=1.20, massless_load_factor=0.25, perform_checks=False
        )

    # cladding loads
    def apply_cladding_load(coords, surf_load, surf_area, factor):
        subset_model = mdl.initialize_empty_copy("subset_1")
        mdl.transfer_by_polygon_selection(subset_model, coords)
        elms = {}
        elm_lens = {}
        for comp in subset_model.list_of_components():
            if comp.component_purpose != "steel_W_panel_zone":
                for elm in [
                    elm
                    for elm in comp.elements.values()
                    if isinstance(elm, (ElasticBeamColumn, DispBeamColumn))
                ]:
                    elms[elm.uid] = elm
                    elm_lens[elm.uid] = elm.clear_length()
        len_tot = sum(elm_lens.values())
        load = surf_load * surf_area
        line_load = load / len_tot
        for key, elm in elms.items():
            loadcase.line_element_udl[key].add_glob(
                np.array((0.00, 0.00, -line_load * factor))
            )
            half_mass = line_load * elm_lens[key] / G_CONST_IMPERIAL
            for nid in range(2):
                loadcase.node_mass[elm.nodes[nid].uid].add(
                    np.array((half_mass, half_mass, half_mass, 0.00, 0.00, 0.00))
                )

    apply_cladding_load(
        np.array(
            [[-50.00, -50.00], [+1850.00, -50.00], [+1850.00, +50.00], [-50.00, +50.00]]
        ),
        15.00 / 12.00**2,
        150.00 * (15.00 + 13.00 + 13.00) * 12.00**2,
        1.2,
    )
    apply_cladding_load(
        np.array(
            [
                [-50.00, +1150.00],
                [+1850.00, +1150.00],
                [+1850.00, +1250.00],
                [-50.00, +1250.00],
            ]
        ),
        15.00 / 12.00**2,
        150.00 * (15.00 + 13.00 + 13.00) * 12.00**2,
        1.2,
    )
    apply_cladding_load(
        np.array(
            [[-50.00, -50.00], [-50.00, 1250.00], [+50.00, 1250.00], [+50.00, -50.00]]
        ),
        15.00 / 12.00**2,
        100.00 * (15.00 + 13.00 + 13.00) * 12.00**2,
        1.2,
    )
    apply_cladding_load(
        np.array(
            [
                [+1800.00 - 50.00, -50.00],
                [+1800.00 - 50.00, 1200.00 + 50.00],
                [+1800.00 + 50.00, 1200.00 + 50.00],
                [+1800.00 + 50.00, -50.00],
            ]
        ),
        15.00 / 12.00**2,
        100.00 * (15.00 + 13.00 + 13.00) * 12.00**2,
        1.2,
    )

    loadcase.rigid_diaphragms(list(range(1, num_levels + 1)), gather_mass=True)

    # from osmg.graphics.preprocessing_3d import show
    # show(mdl, loadcase, extrude=True)

    return mdl, loadcase


def smrf_3_ii() -> tuple[Model, LoadCase]:
    """
    3 story special moment frame risk category II archetype
    """

    heights = np.array((15.00, 13.00 + 15.00, 13.00 + 13.00 + 15.00)) * 12.00

    sections = dict(
        gravity_cols=dict(level_1="W14X48", level_2="W14X48", level_3="W14X48"),
        gravity_beams_a=dict(level_1="W16X31", level_2="W16X31", level_3="W16X31"),
        gravity_beams_b=dict(level_1="W16X31", level_2="W16X31", level_3="W16X31"),
        outer_frame=dict(
            lateral_cols=dict(
                exterior=dict(level_1="W24X94", level_2="W24X94", level_3="W24X94"),
                interior=dict(level_1="W24X176", level_2="W24X176", level_3="W24X176"),
            ),
            lateral_beams=dict(level_1="W24X131", level_2="W24X84", level_3="W24X76"),
        ),
    )

    doubler_plate_thicknesses = dict(
        outer_frame=dict(
            exterior=dict(level_1=0.8750, level_2=0.3125, level_3=0.2500),
            interior=dict(level_1=1.8125, level_2=0.7500, level_3=0.6250),
        )
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=doubler_plate_thicknesses,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="smrf",
        risk_category="ii",
    )

    return mdl, loadcase


def smrf_3_iv() -> tuple[Model, LoadCase]:
    """
    3 story special moment frame risk category IV archetype
    """

    heights = np.array((15.00, 13.00 + 15.00, 13.00 + 13.00 + 15.00)) * 12.00

    sections = dict(
        gravity_cols=dict(level_1="W14X48", level_2="W14X48", level_3="W14X48"),
        gravity_beams_a=dict(level_1="W16X31", level_2="W16X31", level_3="W16X31"),
        gravity_beams_b=dict(level_1="W16X31", level_2="W16X31", level_3="W16X31"),
        outer_frame=dict(
            lateral_cols=dict(
                exterior=dict(level_1="W24X146", level_2="W24X146", level_3="W24X146"),
                interior=dict(level_1="W24X279", level_2="W24X279", level_3="W24X279"),
            ),
            lateral_beams=dict(level_1="W33X169", level_2="W33X169", level_3="W24X76"),
        ),
        inner_frame=dict(
            lateral_cols=dict(
                exterior=dict(level_1="W24X76", level_2="W24X76", level_3="W24X76"),
                interior=dict(level_1="W24X162", level_2="W24X162", level_3="W24X131"),
            ),
            lateral_beams=dict(level_1="W24X76", level_2="W24X76", level_3="W24X76"),
        ),
    )

    doubler_plate_thicknesses = dict(
        outer_frame=dict(
            exterior=dict(level_1=1.00, level_2=1.00, level_3=0.25),
            interior=dict(level_1=1.75, level_2=1.75, level_3=0.00),
        ),
        inner_frame=dict(
            exterior=dict(level_1=0.3125, level_2=0.3125, level_3=0.3125),
            interior=dict(level_1=0.6875, level_2=0.6875, level_3=0.8750),
        ),
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=doubler_plate_thicknesses,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="smrf",
        risk_category="iv",
    )

    return mdl, loadcase


def smrf_6_ii() -> tuple[Model, LoadCase]:
    """
    6 story special moment frame risk category II archetype
    """

    heights = (
        np.array(
            (
                15.00,
                13.00 + 15.00,
                13.00 * 2.00 + 15.00,
                13.00 * 3.00 + 15.00,
                13.00 * 4.00 + 15.00,
                13.00 * 5.00 + 15.00,
            )
        )
        * 12.00
    )

    sections = dict(
        gravity_cols=dict(
            level_1="W14X61",
            level_2="W14X61",
            level_3="W14X61",
            level_4="W14X48",
            level_5="W14X48",
            level_6="W14X48",
        ),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
        ),
        gravity_beams_b=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
        ),
        outer_frame=dict(
            lateral_cols=dict(
                exterior=dict(
                    level_1="W27X114",
                    level_2="W27X114",
                    level_3="W27X102",
                    level_4="W27X102",
                    level_5="W27X94",
                    level_6="W27X94",
                ),
                interior=dict(
                    level_1="W27X217",
                    level_2="W27X217",
                    level_3="W27X161",
                    level_4="W27X161",
                    level_5="W27X129",
                    level_6="W27X129",
                ),
            ),
            lateral_beams=dict(
                level_1="W33X130",
                level_2="W33X130",
                level_3="W30X108",
                level_4="W30X108",
                level_5="W27X94",
                level_6="W27X94",
            ),
        ),
    )

    doubler_plate_thicknesses = dict(
        outer_frame=dict(
            exterior=dict(
                level_1=0.5625,
                level_2=0.5625,
                level_3=0.4375,
                level_4=0.4375,
                level_5=0.3750,
                level_6=0.3750,
            ),
            interior=dict(
                level_1=1.250,
                level_2=1.250,
                level_3=1.125,
                level_4=1.125,
                level_5=1.000,
                level_6=1.000,
            ),
        )
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 6.0) / (12.0**2),
        4: (38.0 + 15.0 + 6.0) / (12.0**2),
        5: (38.0 + 15.0 + 6.0) / (12.0**2),
        6: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2) + 0.0184524,
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 50.00 / (12.0**2),
        4: 50.00 / (12.0**2),
        5: 50.00 / (12.0**2),
        6: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=doubler_plate_thicknesses,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="smrf",
        risk_category="ii",
    )

    return mdl, loadcase


def smrf_6_iv() -> tuple[Model, LoadCase]:
    """
    6 story special moment frame risk category IV archetype
    """

    heights = (
        np.array(
            (
                15.00,
                13.00 + 15.00,
                13.00 * 2.00 + 15.00,
                13.00 * 3.00 + 15.00,
                13.00 * 4.00 + 15.00,
                13.00 * 5.00 + 15.00,
            )
        )
        * 12.00
    )

    sections = dict(
        gravity_cols=dict(
            level_1="W14X61",
            level_2="W14X61",
            level_3="W14X61",
            level_4="W14X48",
            level_5="W14X48",
            level_6="W14X48",
        ),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
        ),
        gravity_beams_b=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
        ),
        outer_frame=dict(
            lateral_cols=dict(
                exterior=dict(
                    level_1="W24X192",
                    level_2="W24X176",
                    level_3="W24X176",
                    level_4="W24X146",
                    level_5="W24X146",
                    level_6="W24X146",
                ),
                interior=dict(
                    level_1="W24X370",
                    level_2="W24X370",
                    level_3="W24X279",
                    level_4="W24X279",
                    level_5="W24X250",
                    level_6="W24X250",
                ),
            ),
            lateral_beams=dict(
                level_1="W36X182",
                level_2="W36X182",
                level_3="W36X170",
                level_4="W36X160",
                level_5="W33X130",
                level_6="W24X76",
            ),
        ),
        inner_frame=dict(
            lateral_cols=dict(
                exterior=dict(
                    level_1="W24X162",
                    level_2="W24X162",
                    level_3="W24X146",
                    level_4="W24X84",
                    level_5="W24X76",
                    level_6="W24X76",
                ),
                interior=dict(
                    level_1="W24X306",
                    level_2="W24X306",
                    level_3="W24X250",
                    level_4="W24X192",
                    level_5="W24X103",
                    level_6="W24X103",
                ),
            ),
            lateral_beams=dict(
                level_1="W27X194",
                level_2="W27X194",
                level_3="W27X146",
                level_4="W27X94",
                level_5="W24X76",
                level_6="W24X76",
            ),
        ),
    )

    doubler_plate_thicknesses = dict(
        outer_frame=dict(
            exterior=dict(
                level_1=0.8125,
                level_2=0.9375,
                level_3=0.8125,
                level_4=0.8750,
                level_5=0.5625,
                level_6=0.2500,
            ),
            interior=dict(
                level_1=1.3125,
                level_2=1.3125,
                level_3=1.6875,
                level_4=1.5000,
                level_5=1.1250,
                level_6=0.2500,
            ),
        ),
        inner_frame=dict(
            exterior=dict(
                level_1=1.2500,
                level_2=1.2500,
                level_3=0.8125,
                level_4=0.4375,
                level_5=0.3125,
                level_6=0.3125,
            ),
            interior=dict(
                level_1=2.1875,
                level_2=2.1875,
                level_3=1.6250,
                level_4=0.8750,
                level_5=0.9375,
                level_6=0.9375,
            ),
        ),
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 6.0) / (12.0**2),
        4: (38.0 + 15.0 + 6.0) / (12.0**2),
        5: (38.0 + 15.0 + 6.0) / (12.0**2),
        6: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 50.00 / (12.0**2),
        4: 50.00 / (12.0**2),
        5: 50.00 / (12.0**2),
        6: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=doubler_plate_thicknesses,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="smrf",
        risk_category="iv",
    )

    return mdl, loadcase


def smrf_9_ii() -> tuple[Model, LoadCase]:
    """
    9 story special moment frame risk category II archetype
    """

    heights = (
        np.array(
            (
                15.00,
                13.00 + 15.00,
                13.00 * 2.00 + 15.00,
                13.00 * 3.00 + 15.00,
                13.00 * 4.00 + 15.00,
                13.00 * 5.00 + 15.00,
                13.00 * 6.00 + 15.00,
                13.00 * 7.00 + 15.00,
                13.00 * 8.00 + 15.00,
            )
        )
        * 12.00
    )

    sections = dict(
        gravity_cols=dict(
            level_1="W14X82",
            level_2="W14X82",
            level_3="W14X82",
            level_4="W14X61",
            level_5="W14X61",
            level_6="W14X61",
            level_7="W14X48",
            level_8="W14X48",
            level_9="W14X48",
        ),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
            level_7="W16X31",
            level_8="W16X31",
            level_9="W16X31",
        ),
        gravity_beams_b=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
            level_7="W16X31",
            level_8="W16X31",
            level_9="W16X31",
        ),
        outer_frame=dict(
            lateral_cols=dict(
                exterior=dict(
                    level_1="W27X194",
                    level_2="W27X194",
                    level_3="W27X146",
                    level_4="W27X146",
                    level_5="W27X102",
                    level_6="W27X102",
                    level_7="W27X94",
                    level_8="W27X94",
                    level_9="W27X94",
                ),
                interior=dict(
                    level_1="W27X307",
                    level_2="W27X307",
                    level_3="W27X235",
                    level_4="W27X235",
                    level_5="W27X194",
                    level_6="W27X161",
                    level_7="W27X161",
                    level_8="W27X129",
                    level_9="W27X129",
                ),
            ),
            lateral_beams=dict(
                level_1="W33X141",
                level_2="W33X141",
                level_3="W33X130",
                level_4="W33X130",
                level_5="W30X116",
                level_6="W30X116",
                level_7="W27X94",
                level_8="W27X94",
                level_9="W27X94",
            ),
        ),
    )

    doubler_plate_thicknesses = dict(
        outer_frame=dict(
            exterior=dict(
                level_1=0.5000,
                level_2=0.5000,
                level_3=0.5625,
                level_4=0.5625,
                level_5=0.5000,
                level_6=0.5625,
                level_7=0.3750,
                level_8=0.3750,
                level_9=0.3750,
            ),
            interior=dict(
                level_1=1.2500,
                level_2=1.3750,
                level_3=1.3750,
                level_4=1.3750,
                level_5=1.1250,
                level_6=1.3125,
                level_7=0.9375,
                level_8=1.0000,
                level_9=1.0000,
            ),
        )
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 6.0) / (12.0**2),
        4: (38.0 + 15.0 + 6.0) / (12.0**2),
        5: (38.0 + 15.0 + 6.0) / (12.0**2),
        6: (38.0 + 15.0 + 6.0) / (12.0**2),
        7: (38.0 + 15.0 + 6.0) / (12.0**2),
        8: (38.0 + 15.0 + 6.0) / (12.0**2),
        9: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 50.00 / (12.0**2),
        4: 50.00 / (12.0**2),
        5: 50.00 / (12.0**2),
        6: 50.00 / (12.0**2),
        7: 50.00 / (12.0**2),
        8: 50.00 / (12.0**2),
        9: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=doubler_plate_thicknesses,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="smrf",
        risk_category="ii",
    )

    return mdl, loadcase


def smrf_9_iv() -> tuple[Model, LoadCase]:
    """
    9 story special moment frame risk category IV archetype
    """

    heights = (
        np.array(
            (
                15.00,
                13.00 + 15.00,
                13.00 * 2.00 + 15.00,
                13.00 * 3.00 + 15.00,
                13.00 * 4.00 + 15.00,
                13.00 * 5.00 + 15.00,
                13.00 * 6.00 + 15.00,
                13.00 * 7.00 + 15.00,
                13.00 * 8.00 + 15.00,
            )
        )
        * 12.00
    )

    sections = dict(
        gravity_cols=dict(
            level_1="W14X82",
            level_2="W14X82",
            level_3="W14X82",
            level_4="W14X61",
            level_5="W14X61",
            level_6="W14X61",
            level_7="W14X48",
            level_8="W14X48",
            level_9="W14X48",
        ),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
            level_7="W16X31",
            level_8="W16X31",
            level_9="W16X31",
        ),
        gravity_beams_b=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
            level_7="W16X31",
            level_8="W16X31",
            level_9="W16X31",
        ),
        outer_frame=dict(
            lateral_cols=dict(
                exterior=dict(
                    level_1="W27X258",
                    level_2="W27X258",
                    level_3="W27X235",
                    level_4="W27X235",
                    level_5="W27X194",
                    level_6="W27X194",
                    level_7="W27X161",
                    level_8="W27X161",
                    level_9="W27X114",
                ),
                interior=dict(
                    level_1="W27X539",
                    level_2="W27X539",
                    level_3="W27X539",
                    level_4="W27X539",
                    level_5="W27X539",
                    level_6="W27X307",
                    level_7="W27X307",
                    level_8="W27X217",
                    level_9="W27X217",
                ),
            ),
            lateral_beams=dict(
                level_1="W36X256",
                level_2="W36X256",
                level_3="W36X256",
                level_4="W36X256",
                level_5="W36X232",
                level_6="W36X194",
                level_7="W36X160",
                level_8="W33X141",
                level_9="W27X94",
            ),
        ),
        inner_frame=dict(
            lateral_cols=dict(
                exterior=dict(
                    level_1="W27X217",
                    level_2="W27X217",
                    level_3="W27X217",
                    level_4="W27X217",
                    level_5="W27X146",
                    level_6="W27X129",
                    level_7="W27X129",
                    level_8="W27X94",
                    level_9="W27X94",
                ),
                interior=dict(
                    level_1="W27X539",
                    level_2="W27X539",
                    level_3="W27X368",
                    level_4="W27X368",
                    level_5="W27X258",
                    level_6="W27X258",
                    level_7="W27X194",
                    level_8="W27X194",
                    level_9="W27X194",
                ),
            ),
            lateral_beams=dict(
                level_1="W27X281",
                level_2="W27X281",
                level_3="W27X281",
                level_4="W27X235",
                level_5="W27X194",
                level_6="W27X178",
                level_7="W27X129",
                level_8="W27X102",
                level_9="W27X94",
            ),
        ),
    )

    doubler_plate_thicknesses = dict(
        outer_frame=dict(
            exterior=dict(
                level_1=1.0625,
                level_2=1.0625,
                level_3=1.1875,
                level_4=1.1875,
                level_5=1.1875,
                level_6=0.8125,
                level_7=0.6875,
                level_8=0.5625,
                level_9=0.2500,
            ),
            interior=dict(
                level_1=1.3750,
                level_2=1.3750,
                level_3=1.3750,
                level_4=1.3750,
                level_5=1.0000,
                level_6=1.8125,
                level_7=1.2500,
                level_8=1.4375,
                level_9=0.6250,
            ),
        ),
        inner_frame=dict(
            exterior=dict(
                level_1=1.6250,
                level_2=1.6250,
                level_3=1.6250,
                level_4=1.1875,
                level_5=1.1875,
                level_6=1.0000,
                level_7=0.5000,
                level_8=0.4375,
                level_9=0.3750,
            ),
            interior=dict(
                level_1=1.8750,
                level_2=1.8750,
                level_3=3.0625,
                level_4=2.3125,
                level_5=2.3125,
                level_6=2.0000,
                level_7=1.4375,
                level_8=0.9375,
                level_9=0.7500,
            ),
        ),
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 6.0) / (12.0**2),
        4: (38.0 + 15.0 + 6.0) / (12.0**2),
        5: (38.0 + 15.0 + 6.0) / (12.0**2),
        6: (38.0 + 15.0 + 6.0) / (12.0**2),
        7: (38.0 + 15.0 + 6.0) / (12.0**2),
        8: (38.0 + 15.0 + 6.0) / (12.0**2),
        9: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 50.00 / (12.0**2),
        4: 50.00 / (12.0**2),
        5: 50.00 / (12.0**2),
        6: 50.00 / (12.0**2),
        7: 50.00 / (12.0**2),
        8: 50.00 / (12.0**2),
        9: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=doubler_plate_thicknesses,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="smrf",
        risk_category="iv",
    )

    return mdl, loadcase


def scbf_3_ii() -> tuple[Model, LoadCase]:
    """
    3 story special concentrically braced frame risk category II
    archetype
    """

    heights = np.array((15.00, 13.00 + 15.00, 13.00 + 13.00 + 15.00)) * 12.00

    sections = dict(
        gravity_cols=dict(level_1="W14X48", level_2="W14X48", level_3="W14X48"),
        gravity_beams_a=dict(level_1="W16X31", level_2="W16X31", level_3="W16X31"),
        gravity_beams_b=dict(level_1="W16X31", level_2="W16X31", level_3="W16X31"),
        lateral_cols=dict(level_1="W14X132", level_2="W14X132", level_3="W14X132"),
        lateral_beams=dict(level_1="W18X86", level_2="W18X86", level_3="W18X86"),
        braces=dict(
            level_1="HSS9.625X0.500", level_2="HSS8.625X0.625", level_3="HSS8.625X0.625"
        ),
    )

    metadata = dict(
        brace_buckling_length={1: 277.1226, 2: 258.3746, 3: 258.3746},
        brace_l_c={1: 19.0213, 2: 16.7005, 3: 16.7005},
        gusset_t_p={1: 1.0000, 2: 1.1250, 3: 1.1250},
        gusset_avg_buckl_len={1: 17.3715, 2: 20.3601, 3: 20.3601},
        hinge_dist={1: 40.3673, 2: 44.3807, 3: 44.3807},
        plate_a={1: 76.0000, 2: 66.0000, 3: 66.0000},
        plate_b={1: 45.6000, 2: 34.3200, 3: 34.3200},
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=metadata,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="scbf",
        risk_category="ii",
    )

    return mdl, loadcase


def scbf_3_iv() -> tuple[Model, LoadCase]:
    """
    3 story special concentrically braced frame risk category IV
    archetype
    """

    heights = np.array((15.00, 13.00 + 15.00, 13.00 + 13.00 + 15.00)) * 12.00

    sections = dict(
        gravity_cols=dict(level_1="W14X48", level_2="W14X48", level_3="W14X48"),
        gravity_beams_a=dict(level_1="W16X31", level_2="W16X31", level_3="W16X31"),
        gravity_beams_b=dict(level_1="W16X31", level_2="W16X31", level_3="W16X31"),
        lateral_cols=dict(level_1="W14X132", level_2="W14X132", level_3="W14X132"),
        lateral_beams=dict(level_1="W18X86", level_2="W18X60", level_3="W18X35"),
        braces=dict(
            level_1="HSS8.625X0.625", level_2="HSS8.625X0.625", level_3="HSS7.625X0.375"
        ),
    )

    metadata = dict(
        brace_buckling_length={1: 277.0718, 2: 258.4769, 3: 268.1875},
        brace_l_c={1: 16.7005, 2: 16.7005, 3: 15.0926},
        gusset_t_p={1: 1.1250, 2: 1.1250, 3: 0.8125},
        gusset_avg_buckl_len={1: 18.9243, 2: 20.3322, 3: 15.1637},
        hinge_dist={1: 40.8927, 2: 44.2227, 3: 37.9504},
        plate_a={1: 66.0000, 2: 66.0000, 3: 60.0000},
        plate_b={1: 39.6000, 2: 34.2980, 3: 31.1500},
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=metadata,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="scbf",
        risk_category="iv",
    )

    return mdl, loadcase


def scbf_6_ii() -> tuple[Model, LoadCase]:
    """
    6 story special concentrically braced frame risk category II
    archetype
    """

    heights = (
        np.array(
            (
                15.00,
                13.00 + 15.00,
                13.00 * 2.00 + 15.00,
                13.00 * 3.00 + 15.00,
                13.00 * 4.00 + 15.00,
                13.00 * 5.00 + 15.00,
            )
        )
        * 12.00
    )

    sections = dict(
        gravity_cols=dict(
            level_1="W14X48",
            level_2="W14X48",
            level_3="W14X48",
            level_4="W14X48",
            level_5="W14X48",
            level_6="W14X48",
        ),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
        ),
        gravity_beams_b=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
        ),
        lateral_cols=dict(
            level_1="W14X211",
            level_2="W14X211",
            level_3="W14X132",
            level_4="W14X132",
            level_5="W14X74",
            level_6="W14X74",
        ),
        lateral_beams=dict(
            level_1="W18X97",
            level_2="W18X97",
            level_3="W18X97",
            level_4="W18X86",
            level_5="W18X86",
            level_6="W18X35",
        ),
        braces=dict(
            level_1="HSS12.750X0.500",
            level_2="HSS10.000X0.625",
            level_3="HSS10.000X0.625",
            level_4="HSS10.000X0.625",
            level_5="HSS8.625X0.625",
            level_6="HSS8.625X0.625",
        ),
    )

    metadata = dict(
        brace_buckling_length={
            1: 268.9417,
            2: 254.8040,
            3: 255.0238,
            4: 255.1230,
            5: 258.4881,
            6: 258.8131,
        },
        brace_l_c={
            1: 25.4090,
            2: 19.5407,
            3: 19.5407,
            4: 19.5407,
            5: 16.7005,
            6: 16.7005,
        },
        gusset_t_p={1: 1.0000, 2: 1.1250, 3: 1.1250, 4: 1.1250, 5: 1.1250, 6: 1.1250},
        gusset_avg_buckl_len={
            1: 18.3244,
            2: 20.4391,
            3: 20.5072,
            4: 20.4851,
            5: 20.4081,
            6: 20.3100,
        },
        hinge_dist={
            1: 44.4577,
            2: 46.1660,
            3: 46.0561,
            4: 45.9021,
            5: 44.3240,
            6: 43.7703,
        },
        plate_a={
            1: 101.0000,
            2: 78.0000,
            3: 78.0000,
            4: 78.0000,
            5: 66.0000,
            6: 66.0000,
        },
        plate_b={
            1: 60.6000,
            2: 40.5600,
            3: 40.5600,
            4: 40.5340,
            5: 34.3200,
            6: 34.2430,
        },
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 6.0) / (12.0**2),
        4: (38.0 + 15.0 + 6.0) / (12.0**2),
        5: (38.0 + 15.0 + 6.0) / (12.0**2),
        6: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 50.00 / (12.0**2),
        4: 50.00 / (12.0**2),
        5: 50.00 / (12.0**2),
        6: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=metadata,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="scbf",
        risk_category="ii",
    )

    return mdl, loadcase


def scbf_6_iv() -> tuple[Model, LoadCase]:
    """
    6 story special concentrically braced frame risk category IV
    archetype
    """

    heights = (
        np.array(
            (
                15.00,
                13.00 + 15.00,
                13.00 * 2.00 + 15.00,
                13.00 * 3.00 + 15.00,
                13.00 * 4.00 + 15.00,
                13.00 * 5.00 + 15.00,
            )
        )
        * 12.00
    )

    sections = dict(
        gravity_cols=dict(
            level_1="W14X61",
            level_2="W14X61",
            level_3="W14X61",
            level_4="W14X48",
            level_5="W14X48",
            level_6="W14X48",
        ),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
        ),
        gravity_beams_b=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
        ),
        lateral_cols=dict(
            level_1="W14X193",
            level_2="W14X193",
            level_3="W14X132",
            level_4="W14X132",
            level_5="W14X68",
            level_6="W14X68",
        ),
        lateral_beams=dict(
            level_1="W18X86",
            level_2="W18X86",
            level_3="W18X86",
            level_4="W18X86",
            level_5="W18X60",
            level_6="W18X35",
        ),
        braces=dict(
            level_1="HSS10.000X0.625",
            level_2="HSS10.000X0.625",
            level_3="HSS8.625X0.625",
            level_4="HSS8.625X0.625",
            level_5="HSS8.625X0.625",
            level_6="HSS7.625X0.375",
        ),
    )

    metadata = dict(
        brace_buckling_length={
            1: 273.6893,
            2: 255.1650,
            3: 258.3746,
            4: 258.3746,
            5: 258.6226,
            6: 268.3380,
        },
        brace_l_c={
            1: 19.5407,
            2: 19.5407,
            3: 16.7005,
            4: 16.7005,
            5: 16.7005,
            6: 15.0926,
        },
        gusset_t_p={1: 1.1250, 2: 1.1250, 3: 1.1250, 4: 1.1250, 5: 1.1250, 6: 0.8125},
        gusset_avg_buckl_len={
            1: 19.1980,
            2: 20.4200,
            3: 20.3601,
            4: 20.3601,
            5: 20.3951,
            6: 15.2223,
        },
        hinge_dist={
            1: 42.5839,
            2: 45.9856,
            3: 44.3807,
            4: 44.3807,
            5: 44.1495,
            6: 37.8742,
        },
        plate_a={
            1: 78.0000,
            2: 78.0000,
            3: 66.0000,
            4: 66.0000,
            5: 66.0000,
            6: 60.0000,
        },
        plate_b={
            1: 46.8000,
            2: 40.5600,
            3: 34.3200,
            4: 34.3200,
            5: 34.2980,
            6: 31.1500,
        },
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 6.0) / (12.0**2),
        4: (38.0 + 15.0 + 6.0) / (12.0**2),
        5: (38.0 + 15.0 + 6.0) / (12.0**2),
        6: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 50.00 / (12.0**2),
        4: 50.00 / (12.0**2),
        5: 50.00 / (12.0**2),
        6: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=metadata,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="scbf",
        risk_category="iv",
    )

    return mdl, loadcase


def scbf_9_ii() -> tuple[Model, LoadCase]:
    """
    9 story special concentrically braced frame risk category II
    archetype
    """

    heights = (
        np.array(
            (
                15.00,
                13.00 + 15.00,
                13.00 * 2.00 + 15.00,
                13.00 * 3.00 + 15.00,
                13.00 * 4.00 + 15.00,
                13.00 * 5.00 + 15.00,
                13.00 * 6.00 + 15.00,
                13.00 * 7.00 + 15.00,
                13.00 * 8.00 + 15.00,
            )
        )
        * 12.00
    )

    sections = dict(
        gravity_cols=dict(
            level_1="W14X48",
            level_2="W14X48",
            level_3="W14X48",
            level_4="W14X48",
            level_5="W14X48",
            level_6="W14X48",
            level_7="W14X48",
            level_8="W14X48",
            level_9="W14X48",
        ),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
            level_7="W16X31",
            level_8="W16X31",
            level_9="W16X31",
        ),
        gravity_beams_b=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
            level_7="W16X31",
            level_8="W16X31",
            level_9="W16X31",
        ),
        lateral_cols=dict(
            level_1="W14X311",
            level_2="W14X311",
            level_3="W14X233",
            level_4="W14X233",
            level_5="W14X159",
            level_6="W14X159",
            level_7="W14X132",
            level_8="W14X132",
            level_9="W14X132",
        ),
        lateral_beams=dict(
            level_1="W18X106",
            level_2="W18X106",
            level_3="W18X97",
            level_4="W18X97",
            level_5="W18X97",
            level_6="W18X97",
            level_7="W18X86",
            level_8="W18X86",
            level_9="W18X35",
        ),
        braces=dict(
            level_1="HSS14.000X0.625",
            level_2="HSS12.750X0.500",
            level_3="HSS12.750X0.500",
            level_4="HSS12.750X0.500",
            level_5="HSS12.750X0.500",
            level_6="HSS10.000X0.625",
            level_7="HSS10.000X0.625",
            level_8="HSS8.625X0.625",
            level_9="HSS8.625X0.625",
        ),
    )

    metadata = dict(
        brace_buckling_length={
            1: 262.8945,
            2: 250.2764,
            3: 250.6551,
            4: 250.7603,
            5: 251.0654,
            6: 254.9689,
            7: 255.1230,
            8: 258.3746,
            9: 258.7150,
        },
        brace_l_c={
            1: 27.8341,
            2: 25.4090,
            3: 25.4090,
            4: 25.4090,
            5: 25.4090,
            6: 19.5407,
            7: 19.5407,
            8: 16.7005,
            9: 16.7005,
        },
        gusset_t_p={
            1: 1.1250,
            2: 1.0000,
            3: 1.0000,
            4: 1.0000,
            5: 1.0000,
            6: 1.1250,
            7: 1.1250,
            8: 1.1250,
            9: 1.1250,
        },
        gusset_avg_buckl_len={
            1: 20.4257,
            2: 18.9745,
            3: 19.0089,
            4: 19.0058,
            5: 19.0454,
            6: 20.4888,
            7: 20.4851,
            8: 20.3601,
            9: 20.2641,
        },
        hinge_dist={
            1: 47.9813,
            2: 47.9298,
            3: 47.6909,
            4: 47.6879,
            5: 47.5353,
            6: 46.0836,
            7: 45.9021,
            8: 44.3807,
            9: 43.8285,
        },
        plate_a={
            1: 111.0000,
            2: 101.0000,
            3: 101.0000,
            4: 101.0000,
            5: 101.0000,
            6: 78.0000,
            7: 78.0000,
            8: 66.0000,
            9: 66.0000,
        },
        plate_b={
            1: 66.6000,
            2: 52.5200,
            3: 52.5032,
            4: 52.5200,
            5: 52.5200,
            6: 40.5600,
            7: 40.5340,
            8: 34.3200,
            9: 34.2430,
        },
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 6.0) / (12.0**2),
        4: (38.0 + 15.0 + 6.0) / (12.0**2),
        5: (38.0 + 15.0 + 6.0) / (12.0**2),
        6: (38.0 + 15.0 + 6.0) / (12.0**2),
        7: (38.0 + 15.0 + 6.0) / (12.0**2),
        8: (38.0 + 15.0 + 6.0) / (12.0**2),
        9: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 50.00 / (12.0**2),
        4: 50.00 / (12.0**2),
        5: 50.00 / (12.0**2),
        6: 50.00 / (12.0**2),
        7: 50.00 / (12.0**2),
        8: 50.00 / (12.0**2),
        9: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=metadata,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="scbf",
        risk_category="ii",
    )

    return mdl, loadcase


def scbf_9_iv() -> tuple[Model, LoadCase]:
    """
    9 story special concentrically braced frame risk category IV
    archetype
    """

    heights = (
        np.array(
            (
                15.00,
                13.00 + 15.00,
                13.00 * 2.00 + 15.00,
                13.00 * 3.00 + 15.00,
                13.00 * 4.00 + 15.00,
                13.00 * 5.00 + 15.00,
                13.00 * 6.00 + 15.00,
                13.00 * 7.00 + 15.00,
                13.00 * 8.00 + 15.00,
            )
        )
        * 12.00
    )

    sections = dict(
        gravity_cols=dict(
            level_1="W14X82",
            level_2="W14X82",
            level_3="W14X82",
            level_4="W14X61",
            level_5="W14X61",
            level_6="W14X61",
            level_7="W14X48",
            level_8="W14X48",
            level_9="W14X48",
        ),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
            level_7="W16X31",
            level_8="W16X31",
            level_9="W16X31",
        ),
        gravity_beams_b=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
            level_7="W16X31",
            level_8="W16X31",
            level_9="W16X31",
        ),
        lateral_cols=dict(
            level_1="W14X311",
            level_2="W14X311",
            level_3="W14X257",
            level_4="W14X257",
            level_5="W14X176",
            level_6="W14X176",
            level_7="W14X132",
            level_8="W14X132",
            level_9="W14X132",
        ),
        lateral_beams=dict(
            level_1="W18X130",
            level_2="W18X130",
            level_3="W18X106",
            level_4="W18X106",
            level_5="W18X97",
            level_6="W18X97",
            level_7="W18X86",
            level_8="W18X86",
            level_9="W18X35",
        ),
        braces=dict(
            level_1="HSS14.000X0.625",
            level_2="HSS14.000X0.625",
            level_3="HSS12.750X0.500",
            level_4="HSS12.750X0.500",
            level_5="HSS12.750X0.500",
            level_6="HSS10.750X0.500",
            level_7="HSS10.000X0.625",
            level_8="HSS9.625X0.500",
            level_9="HSS8.625X0.625",
        ),
    )

    metadata = dict(
        brace_buckling_length={
            1: 262.1255,
            2: 243.6128,
            3: 249.7887,
            4: 250.4779,
            5: 250.9142,
            6: 255.8393,
            7: 255.1230,
            8: 258.9488,
            9: 258.7150,
        },
        brace_l_c={
            1: 27.8341,
            2: 27.8341,
            3: 25.4090,
            4: 25.4090,
            5: 25.4090,
            6: 21.2925,
            7: 19.5407,
            8: 19.0213,
            9: 16.7005,
        },
        gusset_t_p={
            1: 1.1250,
            2: 1.1250,
            3: 1.0000,
            4: 1.0000,
            5: 1.0000,
            6: 1.0000,
            7: 1.1250,
            8: 1.0000,
            9: 1.1250,
        },
        gusset_avg_buckl_len={
            1: 20.4390,
            2: 21.1966,
            3: 19.0106,
            4: 18.9977,
            5: 19.0427,
            6: 18.6312,
            7: 20.4851,
            8: 18.4479,
            9: 20.2641,
        },
        hinge_dist={
            1: 48.3658,
            2: 51.7616,
            3: 47.8665,
            4: 47.8291,
            5: 47.5613,
            6: 45.1484,
            7: 45.9021,
            8: 43.5936,
            9: 43.8285,
        },
        plate_a={
            1: 111.0000,
            2: 111.0000,
            3: 101.0000,
            4: 101.0000,
            5: 101.0000,
            6: 85.0000,
            7: 78.0000,
            8: 76.0000,
            9: 66.0000,
        },
        plate_b={
            1: 66.6000,
            2: 57.7200,
            3: 52.4190,
            4: 52.5200,
            5: 52.5032,
            6: 44.2000,
            7: 40.5340,
            8: 39.5200,
            9: 34.2430,
        },
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 6.0) / (12.0**2),
        4: (38.0 + 15.0 + 6.0) / (12.0**2),
        5: (38.0 + 15.0 + 6.0) / (12.0**2),
        6: (38.0 + 15.0 + 6.0) / (12.0**2),
        7: (38.0 + 15.0 + 6.0) / (12.0**2),
        8: (38.0 + 15.0 + 6.0) / (12.0**2),
        9: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 50.00 / (12.0**2),
        4: 50.00 / (12.0**2),
        5: 50.00 / (12.0**2),
        6: 50.00 / (12.0**2),
        7: 50.00 / (12.0**2),
        8: 50.00 / (12.0**2),
        9: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=metadata,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="scbf",
        risk_category="iv",
    )

    return mdl, loadcase


def brbf_3_ii() -> tuple[Model, LoadCase]:
    """
    3 story special buckling restrained braced frame risk category II
    archetype
    """

    heights = np.array((15.00, 13.00 + 15.00, 13.00 + 13.00 + 15.00)) * 12.00

    sections = dict(
        gravity_cols=dict(level_1="W14X48", level_2="W14X48", level_3="W14X48"),
        gravity_beams_a=dict(level_1="W16X31", level_2="W16X31", level_3="W16X31"),
        gravity_beams_b=dict(level_1="W16X31", level_2="W16X31", level_3="W16X31"),
        lateral_cols=dict(level_1="W14X68", level_2="W14X68", level_3="W14X53"),
        lateral_beams=dict(level_1="W18X86", level_2="W18X86", level_3="W18X35"),
        braces=dict(level_1="7.00", level_2="5.50", level_3="4.00"),
    )

    metadata = dict(
        plate_a={1: 40.0000, 2: 40.0000, 3: 40.0000},
        plate_b={1: 20.0000, 2: 20.0000, 3: 20.0000},
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=metadata,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="brbf",
        risk_category="ii",
    )

    return mdl, loadcase


def brbf_3_iv() -> tuple[Model, LoadCase]:
    """
    3 story special buckling restrained braced frame risk category IV
    archetype
    """

    heights = np.array((15.00, 13.00 + 15.00, 13.00 + 13.00 + 15.00)) * 12.00

    sections = dict(
        gravity_cols=dict(level_1="W14X48", level_2="W14X48", level_3="W14X48"),
        gravity_beams_a=dict(level_1="W16X31", level_2="W16X31", level_3="W16X31"),
        gravity_beams_b=dict(level_1="W16X31", level_2="W16X31", level_3="W16X31"),
        lateral_cols=dict(level_1="W14X48", level_2="W14X48", level_3="W14X48"),
        lateral_beams=dict(level_1="W18X86", level_2="W18X86", level_3="W18X35"),
        braces=dict(level_1="5.75", level_2="4.25", level_3="3.00"),
    )

    metadata = dict(
        plate_a={1: 40.0000, 2: 40.0000, 3: 40.0000},
        plate_b={1: 20.0000, 2: 20.0000, 3: 20.0000},
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=metadata,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="brbf",
        risk_category="iv",
    )

    return mdl, loadcase


def brbf_6_ii() -> tuple[Model, LoadCase]:
    """
    6 story special buckling restrained braced frame risk category II
    archetype
    """

    heights = (
        np.array(
            (
                15.00,
                13.00 + 15.00,
                13.00 * 2.00 + 15.00,
                13.00 * 3.00 + 15.00,
                13.00 * 4.00 + 15.00,
                13.00 * 5.00 + 15.00,
            )
        )
        * 12.00
    )

    sections = dict(
        gravity_cols=dict(
            level_1="W14X61",
            level_2="W14X61",
            level_3="W14X61",
            level_4="W14X48",
            level_5="W14X48",
            level_6="W14X48",
        ),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
        ),
        gravity_beams_b=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
        ),
        lateral_cols=dict(
            level_1="W14X145",
            level_2="W14X145",
            level_3="W14X68",
            level_4="W14X68",
            level_5="W14X38",
            level_6="W14X38",
        ),
        lateral_beams=dict(
            level_1="W18X119",
            level_2="W18X119",
            level_3="W18X97",
            level_4="W18X86",
            level_5="W18X86",
            level_6="W18X35",
        ),
        braces=dict(
            level_1="11.00",
            level_2="9.50",
            level_3="9.50",
            level_4="7.00",
            level_5="6.50",
            level_6="3.00",
        ),
    )

    metadata = dict(
        plate_a={
            1: 40.0000,
            2: 40.0000,
            3: 40.0000,
            4: 40.0000,
            5: 40.0000,
            6: 40.0000,
        },
        plate_b={
            1: 20.0000,
            2: 20.0000,
            3: 20.0000,
            4: 20.0000,
            5: 20.0000,
            6: 20.0000,
        },
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 6.0) / (12.0**2),
        4: (38.0 + 15.0 + 6.0) / (12.0**2),
        5: (38.0 + 15.0 + 6.0) / (12.0**2),
        6: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 50.00 / (12.0**2),
        4: 50.00 / (12.0**2),
        5: 50.00 / (12.0**2),
        6: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=metadata,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="brbf",
        risk_category="ii",
    )

    return mdl, loadcase


def brbf_6_iv() -> tuple[Model, LoadCase]:
    """
    6 story special buckling restrained braced frame risk category IV
    archetype
    """

    heights = (
        np.array(
            (
                15.00,
                13.00 + 15.00,
                13.00 * 2.00 + 15.00,
                13.00 * 3.00 + 15.00,
                13.00 * 4.00 + 15.00,
                13.00 * 5.00 + 15.00,
            )
        )
        * 12.00
    )

    sections = dict(
        gravity_cols=dict(
            level_1="W14X61",
            level_2="W14X61",
            level_3="W14X61",
            level_4="W14X48",
            level_5="W14X48",
            level_6="W14X48",
        ),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
        ),
        gravity_beams_b=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
        ),
        lateral_cols=dict(
            level_1="W14X132",
            level_2="W14X132",
            level_3="W14X74",
            level_4="W14X74",
            level_5="W14X53",
            level_6="W14X53",
        ),
        lateral_beams=dict(
            level_1="W18X119",
            level_2="W18X106",
            level_3="W18X97",
            level_4="W18X97",
            level_5="W18X86",
            level_6="W18X35",
        ),
        braces=dict(
            level_1="10.00",
            level_2="9.00",
            level_3="8.00",
            level_4="6.50",
            level_5="5.25",
            level_6="2.75",
        ),
    )

    metadata = dict(
        plate_a={
            1: 40.0000,
            2: 40.0000,
            3: 40.0000,
            4: 40.0000,
            5: 40.0000,
            6: 40.0000,
        },
        plate_b={
            1: 20.0000,
            2: 20.0000,
            3: 20.0000,
            4: 20.0000,
            5: 20.0000,
            6: 20.0000,
        },
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 6.0) / (12.0**2),
        4: (38.0 + 15.0 + 6.0) / (12.0**2),
        5: (38.0 + 15.0 + 6.0) / (12.0**2),
        6: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 50.00 / (12.0**2),
        4: 50.00 / (12.0**2),
        5: 50.00 / (12.0**2),
        6: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=metadata,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="brbf",
        risk_category="iv",
    )

    return mdl, loadcase


def brbf_9_ii() -> tuple[Model, LoadCase]:
    """
    9 story special buckling restrained braced frame risk category II
    archetype
    """

    heights = (
        np.array(
            (
                15.00,
                13.00 + 15.00,
                13.00 * 2.00 + 15.00,
                13.00 * 3.00 + 15.00,
                13.00 * 4.00 + 15.00,
                13.00 * 5.00 + 15.00,
                13.00 * 6.00 + 15.00,
                13.00 * 7.00 + 15.00,
                13.00 * 8.00 + 15.00,
            )
        )
        * 12.00
    )

    sections = dict(
        gravity_cols=dict(
            level_1="W14X82",
            level_2="W14X82",
            level_3="W14X82",
            level_4="W14X61",
            level_5="W14X61",
            level_6="W14X61",
            level_7="W14X48",
            level_8="W14X48",
            level_9="W14X48",
        ),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
            level_7="W16X31",
            level_8="W16X31",
            level_9="W16X31",
        ),
        gravity_beams_b=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
            level_7="W16X31",
            level_8="W16X31",
            level_9="W16X31",
        ),
        lateral_cols=dict(
            level_1="W14X233",
            level_2="W14X233",
            level_3="W14X145",
            level_4="W14X132",
            level_5="W14X132",
            level_6="W14X68",
            level_7="W14X68",
            level_8="W14X38",
            level_9="W14X38",
        ),
        lateral_beams=dict(
            level_1="W18X130",
            level_2="W18X130",
            level_3="W18X119",
            level_4="W18X119",
            level_5="W18X106",
            level_6="W18X97",
            level_7="W18X86",
            level_8="W18X86",
            level_9="W18X35",
        ),
        braces=dict(
            level_1="12.75",
            level_2="10.75",
            level_3="10.50",
            level_4="9.00",
            level_5="9.00",
            level_6="8.00",
            level_7="7.00",
            level_8="4.50",
            level_9="3.50",
        ),
    )

    metadata = dict(
        plate_a={
            1: 40.0000,
            2: 40.0000,
            3: 40.0000,
            4: 40.0000,
            5: 40.0000,
            6: 40.0000,
            7: 40.0000,
            8: 40.0000,
            9: 40.0000,
        },
        plate_b={
            1: 20.0000,
            2: 20.0000,
            3: 20.0000,
            4: 20.0000,
            5: 20.0000,
            6: 20.0000,
            7: 20.0000,
            8: 20.0000,
            9: 20.0000,
        },
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 6.0) / (12.0**2),
        4: (38.0 + 15.0 + 6.0) / (12.0**2),
        5: (38.0 + 15.0 + 6.0) / (12.0**2),
        6: (38.0 + 15.0 + 6.0) / (12.0**2),
        7: (38.0 + 15.0 + 6.0) / (12.0**2),
        8: (38.0 + 15.0 + 6.0) / (12.0**2),
        9: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }

    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 50.00 / (12.0**2),
        4: 50.00 / (12.0**2),
        5: 50.00 / (12.0**2),
        6: 50.00 / (12.0**2),
        7: 50.00 / (12.0**2),
        8: 50.00 / (12.0**2),
        9: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=metadata,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="brbf",
        risk_category="ii",
    )

    return mdl, loadcase


def brbf_9_iv() -> tuple[Model, LoadCase]:
    """
    9 story special buckling restrained braced frame risk category IV
    archetype
    """

    heights = (
        np.array(
            (
                15.00,
                13.00 + 15.00,
                13.00 * 2.00 + 15.00,
                13.00 * 3.00 + 15.00,
                13.00 * 4.00 + 15.00,
                13.00 * 5.00 + 15.00,
                13.00 * 6.00 + 15.00,
                13.00 * 7.00 + 15.00,
                13.00 * 8.00 + 15.00,
            )
        )
        * 12.00
    )

    sections = dict(
        gravity_cols=dict(
            level_1="W14X82",
            level_2="W14X82",
            level_3="W14X82",
            level_4="W14X61",
            level_5="W14X61",
            level_6="W14X61",
            level_7="W14X48",
            level_8="W14X48",
            level_9="W14X48",
        ),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
            level_7="W16X31",
            level_8="W16X31",
            level_9="W16X31",
        ),
        gravity_beams_b=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31",
            level_4="W16X31",
            level_5="W16X31",
            level_6="W16X31",
            level_7="W16X31",
            level_8="W16X31",
            level_9="W16X31",
        ),
        lateral_cols=dict(
            level_1="W14X257",
            level_2="W14X257",
            level_3="W14X193",
            level_4="W14X193",
            level_5="W14X132",
            level_6="W14X132",
            level_7="W14X132",
            level_8="W14X132",
            level_9="W14X132",
        ),
        lateral_beams=dict(
            level_1="W18X143",
            level_2="W18X143",
            level_3="W18X130",
            level_4="W18X130",
            level_5="W18X106",
            level_6="W18X106",
            level_7="W18X97",
            level_8="W18X97",
            level_9="W18X60",
        ),
        braces=dict(
            level_1="11.25",
            level_2="10.50",
            level_3="10.00",
            level_4="9.50",
            level_5="9.00",
            level_6="7.50",
            level_7="7.00",
            level_8="4.50",
            level_9="3.00",
        ),
    )

    metadata = dict(
        plate_a={
            1: 40.0000,
            2: 40.0000,
            3: 40.0000,
            4: 40.0000,
            5: 40.0000,
            6: 40.0000,
            7: 40.0000,
            8: 40.0000,
            9: 40.0000,
        },
        plate_b={
            1: 20.0000,
            2: 20.0000,
            3: 20.0000,
            4: 20.0000,
            5: 20.0000,
            6: 20.0000,
            7: 20.0000,
            8: 20.0000,
            9: 20.0000,
        },
    )

    surf_loads = {
        1: (38.0 + 15.0 + 6.0) / (12.0**2),
        2: (38.0 + 15.0 + 6.0) / (12.0**2),
        3: (38.0 + 15.0 + 6.0) / (12.0**2),
        4: (38.0 + 15.0 + 6.0) / (12.0**2),
        5: (38.0 + 15.0 + 6.0) / (12.0**2),
        6: (38.0 + 15.0 + 6.0) / (12.0**2),
        7: (38.0 + 15.0 + 6.0) / (12.0**2),
        8: (38.0 + 15.0 + 6.0) / (12.0**2),
        9: (38.0 + 15.0 + 80.0 * 0.1666) / (12.0**2),
    }
    surf_loads_massless = {
        1: 50.00 / (12.0**2),
        2: 50.00 / (12.0**2),
        3: 50.00 / (12.0**2),
        4: 50.00 / (12.0**2),
        5: 50.00 / (12.0**2),
        6: 50.00 / (12.0**2),
        7: 50.00 / (12.0**2),
        8: 50.00 / (12.0**2),
        9: 20.00 / (12.0**2),
    }

    mdl, loadcase = generate_archetype(
        level_elevs=heights,
        sections=sections,
        metadata=metadata,
        surf_loads=surf_loads,
        surf_loads_massless=surf_loads_massless,
        lateral_system="brbf",
        risk_category="iv",
    )

    return mdl, loadcase
