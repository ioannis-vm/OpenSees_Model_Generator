"""
Basic Tests
"""

# import pytest
import numpy as np
import numpy.typing as npt
from osmg.model import Model
from osmg.graphics.preprocessing_3d import show
from osmg import defaults
from osmg.gen.section_gen import SectionGenerator
from osmg.ops.section import ElasticSection
from osmg.gen.beamcolumn_gen import BeamColumnGenerator
from osmg.ops.element import ElasticBeamColumn
from osmg.load_case import LoadCase
from osmg.gen.querry import ElmQuerry
from osmg.preprocessing.self_weight_mass import self_weight
from osmg.preprocessing.self_weight_mass import self_mass
from osmg.solver import PushoverAnalysis
from osmg.graphics.postprocessing_3d import show_deformed_shape
from osmg.graphics.postprocessing_3d import show_basic_forces
from osmg.gen.zerolength_gen import gravity_shear_tab
from osmg.model import Model
from osmg.gen.beamcolumn_gen import BeamColumnGenerator
from osmg.gen.section_gen import SectionGenerator
from osmg import defaults
from osmg.preprocessing.self_weight_mass import self_weight
from osmg.preprocessing.self_weight_mass import self_mass
from osmg.ops.section import ElasticSection
from osmg.ops.element import ElasticBeamColumn
from osmg.gen.querry import ElmQuerry
from osmg.gen.zerolength_gen import release_56
from osmg.gen.zerolength_gen import release_5_imk_6
from osmg.gen.zerolength_gen import imk_6
from osmg.gen.zerolength_gen import gravity_shear_tab
from osmg.load_case import LoadCase
from osmg.preprocessing.tributary_area_analysis import PolygonLoad
import numpy as np
from osmg.model import Model
from osmg.gen.beamcolumn_gen import BeamColumnGenerator
from osmg.gen.section_gen import SectionGenerator
from osmg.gen.querry import ElmQuerry
from osmg import defaults
from osmg.ops.element import ElasticBeamColumn
from osmg.graphics.preprocessing_3d import show
from osmg.gen.zerolength_gen import release_6
from osmg.load_case import LoadCase
from osmg import solver
from osmg.ops.section import ElasticSection
from osmg import common
from osmg.postprocessing.steel_design_checks import smrf_scwb
from osmg.postprocessing.steel_design_checks import smrf_pz_doubler_plate_requirement
from osmg.preprocessing.self_weight_mass import self_weight
from osmg.preprocessing.self_weight_mass import self_mass
from osmg.graphics.postprocessing_3d import show_deformed_shape
from osmg.graphics.postprocessing_3d import show_basic_forces
import numpy as np
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import pandas as pd
import timeit


nparr = npt.NDArray[np.float64]


def test_basic_functionality():
    """
    Basic functionality tests
    Simple frame model
    Imperial units
    """

    mdl = Model('test_model')
    mdl.settings.imperial_units = True

    mcg = BeamColumnGenerator(mdl)
    secg = SectionGenerator(mdl)
    querry = ElmQuerry(mdl)

    mdl.add_level(0, 0.00)
    mdl.add_level(1, 15.00 * 12.00)

    defaults.load_default_steel(mdl)
    defaults.load_default_fix_release(mdl)
    steel_phys_mat = mdl.physical_materials.retrieve_by_attr(
        'name', 'default steel')

    section_type = ElasticSection
    element_type = ElasticBeamColumn
    sec_collection = mdl.elastic_sections

    mdl.levels.set_active([1])

    secg.load_aisc_from_database(
        'W',
        ['W24X131'],
        'default steel',
        'default steel',
        section_type
    )

    pt0: nparr = np.array((0.00, 0.00))
    pt1: nparr = np.array((0.00, 25.00 * 12.00))

    sec = sec_collection.retrieve_by_attr('name', 'W24X131')

    mcg.add_vertical_active(
        pt0[0], pt0[1],
        np.zeros(3), np.zeros(3),
        'Linear',
        1,
        sec,
        element_type,
        'centroid',
        2.00 * np.pi / 2.00
    )

    mcg.add_vertical_active(
        pt1[0], pt1[1],
        np.zeros(3), np.zeros(3),
        'Linear',
        1,
        sec,
        element_type,
        'centroid',
        2.00 * np.pi / 2.00
    )

    mcg.add_horizontal_active(
        pt0[0], pt0[1],
        pt1[0], pt1[1],
        np.array((0., 0., 0.)),
        np.array((0., 0., 0.)),
        'bottom_center',
        'top_center',
        'Linear',
        1,
        sec,
        element_type,
        'top_center',
        method='generate_hinged_component_assembly',
        additional_args={
            'zerolength_gen_i': gravity_shear_tab,
            'zerolength_gen_args_i': {
                'consider_composite': True,
                'section': sec,
                'physical_material': steel_phys_mat,
                'distance': 10.00,
                'n_sub': 1
            },
            'zerolength_gen_j': gravity_shear_tab,
            'zerolength_gen_args_j': {
                'consider_composite': True,
                'section': sec,
                'physical_material': steel_phys_mat,
                'distance': 10.00,
                'n_sub': 1
            }
        }
    )

    # fix base
    for node in mdl.levels[0].nodes.values():
        node.restraint = [True]*6

    testcase = LoadCase('test', mdl)
    self_weight(mdl, testcase)
    self_mass(mdl, testcase)

    show(mdl, testcase)

    control_node = querry.search_node_lvl(0.00, 0.00, 1)

    anl = PushoverAnalysis(mdl, {testcase.name: testcase})

    anl.run('y', [+50.00], control_node, 0.1, loaded_node=control_node)

    show_deformed_shape(
        anl, testcase.name,
        anl.results[testcase.name].n_steps_success,
        0.00, True)

    show_basic_forces(anl, testcase.name, 0, 1.00, 0.00, 0.00, 0.00, 0.00, 3)

    # zelms = mdl.list_of_zerolength_elements()
    # zelm = zelms[0].uid
    # res_a = anl.retrieve_release_force_defo(zelm, testcase.name)

    anl.run('y', [-50.00], control_node, 0.1, loaded_node=control_node)

    # deformed_shape(anl, anl.n_steps_success, 0.00, True)
    # res_b = anl.retrieve_release_force_defo(zelm, testcase.name)















    
    heights = np.array(
        (15.00,
         13.00+15.00,
         13.00+13.00+15.00)) * 12.00

    sections = dict(
        gravity_cols=dict(
            level_1="W14X90",
            level_2="W14X90",
            level_3="W14X90"),
        gravity_beams_a=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31"),
        gravity_beams_b=dict(
            level_1="W21X44",
            level_2="W21X44",
            level_3="W21X44"),
        gravity_beams_c=dict(
            level_1="W24X62",
            level_2="W24X62",
            level_3="W24X62"),
        gravity_beams_d=dict(
            level_1="W21X44",
            level_2="W21X44",
            level_3="W21X48"),
        gravity_beams_e=dict(
            level_1="W16X31",
            level_2="W16X31",
            level_3="W16X31"),
        lateral_cols=dict(
            exterior=dict(
                level_1="W18X119",
                level_2="W18X119",
                level_3="W18X106"),
            interior=dict(
                level_1="W24X176",
                level_2="W24X176",
                level_3="W24X162")),
        lateral_beams=dict(
            level_1="W30X132",
            level_2="W30X124",
            level_3="W21X50")
        )

    doubler_plate_thicknesses = dict(
        exterior=dict(
            level_1=0.719992,
            level_2=0.628111,
            level_3=0.000000
        ),
        interior=dict(
            level_1=1.332163,
            level_2=1.193641,
            level_3=0.007416
        )
    )

    surf_loads = {
        1: (63.+15.+15.)/(12.**2),
        2: (63.+15.+15.)/(12.**2),
        3: (63.+15.+80.*0.26786)/(12.**2)
    }
    surf_loads_massless = {
        1: 50.00/(12.**2),
        2: 50.00/(12.**2),
        3: 20.00/(12.**2)
    }

    level_elevs = heights

    mdl = Model('test_model')
    mdl.settings.imperial_units = True
    mcg = BeamColumnGenerator(mdl)
    secg = SectionGenerator(mdl)
    querry = ElmQuerry(mdl)

    num_levels = len(level_elevs)

    mdl.add_level(0, 0.00)
    for i, h in enumerate(level_elevs):
        mdl.add_level(i+1, h)

    defaults.load_default_steel(mdl)
    defaults.load_default_fix_release(mdl)
    steel_phys_mat = mdl.physical_materials.retrieve_by_attr('name', 'default steel')
    # define sections
    wsections = set()
    for lvl_tag in [f'level_{i+1}' for i in range(num_levels)]:
        wsections.add(sections['gravity_beams_a'][lvl_tag])
        wsections.add(sections['gravity_beams_b'][lvl_tag])
        wsections.add(sections['gravity_beams_c'][lvl_tag])
        wsections.add(sections['gravity_beams_d'][lvl_tag])
        wsections.add(sections['gravity_beams_e'][lvl_tag])
        wsections.add(sections['lateral_beams'][lvl_tag])
        wsections.add(sections['gravity_cols'][lvl_tag])
    for function in ['exterior', 'interior']:
        for lvl_tag in [f'level_{i+1}' for i in range(num_levels)]:
            wsections.add(sections['lateral_cols'][function][lvl_tag])

    section_type = ElasticSection
    element_type = ElasticBeamColumn
    sec_collection = mdl.elastic_sections

    for sec in wsections:
        secg.load_aisc_from_database(
            'W',
            [sec],
            'default steel',
            'default steel',
            section_type
        )

    # generate a dictionary containing coordinates given gridline tag names
    point = {}
    x_grd_tags = ['A', 'B', 'C', 'D', 'E', 'F']
    y_grd_tags = ['5', '4', '3', '2', '1']
    x_grd_locs = np.array(
        [0.00, 32.5, 57.5, 82.5, 107.5, 140.00]) * 12.00 + 10.00  # (in)
    y_grd_locs = np.array(
        [0.00, 25.00, 50.00, 75.00, 100.00]) * 12.00 + 10.00  # (in)

    n_sub = 1

    for i in range(len(x_grd_tags)):
        point[x_grd_tags[i]] = {}
        for j in range(len(y_grd_tags)):
            point[x_grd_tags[i]][y_grd_tags[j]] = \
                np.array([x_grd_locs[i], y_grd_locs[j]])

    lat_col_n_sub = 2
    col_gtransf = 'Corotational'

    for level_counter in range(num_levels):
        level_tag = 'level_'+str(level_counter+1)
        mdl.levels.set_active([level_counter+1])

        # define gravity columns
        sec = sec_collection.retrieve_by_attr(
            'name', sections['gravity_cols'][level_tag])
        for tag in ['A', 'F']:
            pt = point[tag]['1']
            mcg.add_vertical_active(
                pt[0], pt[1],
                np.zeros(3), np.zeros(3),
                col_gtransf,
                n_sub,
                sec,
                element_type,
                'centroid',
                0.00,
                method='generate_hinged_component_assembly',
                additional_args={
                    'zerolength_gen_i': None,
                    'zerolength_gen_args_i': {},
                    'zerolength_gen_j': release_56,
                    'zerolength_gen_args_j': {
                        'distance': 10.00,
                        'n_sub': 1
                    },
                }
            )
        for tag1 in ['B', 'C', 'D', 'E']:
            for tag2 in ['2', '3', '4']:
                pt = point[tag1][tag2]
                mcg.add_vertical_active(
                    pt[0], pt[1],
                    np.zeros(3), np.zeros(3),
                    col_gtransf,
                    n_sub,
                    sec,
                    element_type,
                    'centroid',
                    0.00,
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': None,
                        'zerolength_gen_args_i': {},
                        'zerolength_gen_j': release_56,
                        'zerolength_gen_args_j': {
                            'distance': 10.00,
                            'n_sub': 1
                        },
                    }
                )

        # define X-dir frame columns
        sec = sec_collection.retrieve_by_attr(
            'name', sections['lateral_cols']['exterior'][level_tag])
        column_depth = sec.properties['d']
        beam_depth = sec_collection.retrieve_by_attr(
            'name', sections['lateral_beams'][level_tag]).properties['d']
        for tag1 in ['B', 'E']:
            for tag2 in ['1', '5']:
                pt = point[tag1][tag2]
                mcg.add_pz_active(
                    pt[0], pt[1],
                    sec,
                    steel_phys_mat,
                    np.pi/2.00,
                    column_depth,
                    beam_depth,
                    doubler_plate_thicknesses['exterior'][level_tag],
                    0.02
                )
                mcg.add_vertical_active(
                    pt[0], pt[1],
                    np.zeros(3), np.zeros(3),
                    col_gtransf,
                    lat_col_n_sub,
                    sec,
                    element_type,
                    'centroid',
                    np.pi/2.00,
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': imk_6,
                        'zerolength_gen_args_i': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': release_5_imk_6,
                        'zerolength_gen_args_j': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )
        sec = sec_collection.retrieve_by_attr(
            'name', sections['lateral_cols']['interior'][level_tag])
        column_depth = sec.properties['d']
        for tag1 in ['C', 'D']:
            for tag2 in ['1', '5']:
                pt = point[tag1][tag2]
                mcg.add_pz_active(
                    pt[0], pt[1],
                    sec,
                    steel_phys_mat,
                    np.pi/2.00,
                    column_depth,
                    beam_depth,
                    doubler_plate_thicknesses['interior'][level_tag],
                    0.02
                )
                mcg.add_vertical_active(
                    pt[0], pt[1],
                    np.zeros(3), np.zeros(3),
                    col_gtransf,
                    lat_col_n_sub,
                    sec,
                    element_type,
                    'centroid',
                    np.pi/2.00,
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': imk_6,
                        'zerolength_gen_args_i': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': release_5_imk_6,
                        'zerolength_gen_args_j': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )

        # deffine Y-dir frame columns
        sec = sec_collection.retrieve_by_attr(
            'name', sections['lateral_cols']['exterior'][level_tag])
        column_depth = sec.properties['d']
        for tag1 in ['A', 'F']:
            for tag2 in ['5', '2']:
                pt = point[tag1][tag2]
                mcg.add_pz_active(
                    pt[0], pt[1],
                    sec,
                    steel_phys_mat,
                    0.00,
                    column_depth,
                    beam_depth,
                    doubler_plate_thicknesses['exterior'][level_tag],
                    0.02
                )
                mcg.add_vertical_active(
                    pt[0], pt[1],
                    np.zeros(3), np.zeros(3),
                    col_gtransf,
                    lat_col_n_sub,
                    sec,
                    element_type,
                    'centroid',
                    0.00,
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': imk_6,
                        'zerolength_gen_args_i': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': release_5_imk_6,
                        'zerolength_gen_args_j': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )
        sec = sec_collection.retrieve_by_attr(
            'name', sections['lateral_cols']['interior'][level_tag])
        column_depth = sec.properties['d']
        for tag1 in ['A', 'F']:
            for tag2 in ['4', '3']:
                pt = point[tag1][tag2]
                mcg.add_pz_active(
                    pt[0], pt[1],
                    sec,
                    steel_phys_mat,
                    0.00,
                    column_depth,
                    beam_depth,
                    doubler_plate_thicknesses['interior'][level_tag],
                    0.02
                )
                mcg.add_vertical_active(
                    pt[0], pt[1],
                    np.zeros(3), np.zeros(3),
                    col_gtransf,
                    lat_col_n_sub,
                    sec,
                    element_type,
                    'centroid',
                    0.00,
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': imk_6,
                        'zerolength_gen_args_i': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': release_5_imk_6,
                        'zerolength_gen_args_j': {
                            'lbry': 60.00,
                            'loverh': 1.00,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )
        # define X-dir frame beams
        sec = sec_collection.retrieve_by_attr(
            'name', sections['lateral_beams'][level_tag])
        for tag1 in ['1', '5']:
            tag2_start = ['B', 'C', 'D']
            tag2_end = ['C', 'D', 'E']
            for j in range(len(tag2_start)):
                mcg.add_horizontal_active(
                    point[tag2_start[j]][tag1][0], point[tag2_start[j]][tag1][1],
                    point[tag2_end[j]][tag1][0], point[tag2_end[j]][tag1][1],
                    np.array((0., 0., 0.)),
                    np.array((0., 0., 0.)),
                    'middle_back',
                    'middle_front',
                    # 'centroid',
                    # 'centroid',
                    'Linear',
                    n_sub,
                    sec,
                    element_type,
                    'top_center',
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': imk_6,
                        'zerolength_gen_args_i': {
                            'lbry': 60.00,
                            'loverh': 0.50,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': imk_6,
                        'zerolength_gen_args_j': {
                            'lbry': 60.00,
                            'loverh': 0.50,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )
        # define Y-dir frame beams
        for tag1 in ['A', 'F']:
            tag2_start = ['2', '3', '4']
            tag2_end = ['3', '4', '5']
            for j in range(len(tag2_start)):
                mcg.add_horizontal_active(
                    point[tag1][tag2_start[j]][0], point[tag1][tag2_start[j]][1],
                    point[tag1][tag2_end[j]][0], point[tag1][tag2_end[j]][1],
                    np.array((0., 0., 0.)),
                    np.array((0., 0., 0.)),
                    'middle_back',
                    'middle_front',
                    # 'centroid',
                    # 'centroid',
                    'Linear',
                    n_sub,
                    sec,
                    element_type,
                    'top_center',
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': imk_6,
                        'zerolength_gen_args_i': {
                            'lbry': 60.00,
                            'loverh': 0.50,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': imk_6,
                        'zerolength_gen_args_j': {
                            'lbry': 60.00,
                            'loverh': 0.50,
                            'rbs_factor': 0.60,
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )

        # define gravity beams of designation A
        sec = sec_collection.retrieve_by_attr(
            'name', sections['gravity_beams_a'][level_tag])
        for tag1 in ['A', 'F']:
            tag2_start = ['1']
            tag2_end = ['2']
            for j in range(len(tag2_start)):
                mcg.add_horizontal_active(
                    point[tag1][tag2_start[j]][0], point[tag1][tag2_start[j]][1],
                    point[tag1][tag2_end[j]][0], point[tag1][tag2_end[j]][1],
                    np.array((0., 0., 0.)),
                    np.array((0., 0., 0.)),
                    'bottom_center',
                    'top_center',
                    'Linear',
                    n_sub,
                    sec,
                    element_type,
                    'top_center',
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': gravity_shear_tab,
                        'zerolength_gen_args_i': {
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': gravity_shear_tab,
                        'zerolength_gen_args_j': {
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )
        for tag1 in ['B', 'C', 'D', 'E']:
            tag2_start = ['1', '2', '3', '4']
            tag2_end = ['2', '3', '4', '5']
            for j in range(len(tag2_start)):
                if tag2_start[j] == '1':
                    si = 'center_left'
                    sj = 'top_center'
                elif tag2_end[j] == '5':
                    si = 'bottom_center'
                    sj = 'center_right'
                else:
                    si = 'bottom_center'
                    sj = 'top_center'
                mcg.add_horizontal_active(
                    point[tag1][tag2_start[j]][0], point[tag1][tag2_start[j]][1],
                    point[tag1][tag2_end[j]][0], point[tag1][tag2_end[j]][1],
                    np.array((0., 0., 0.)),
                    np.array((0., 0., 0.)),
                    si,
                    sj,
                    'Linear',
                    n_sub,
                    sec,
                    element_type,
                    'top_center',
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': gravity_shear_tab,
                        'zerolength_gen_args_i': {
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': gravity_shear_tab,
                        'zerolength_gen_args_j': {
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )

        # define gravity beams of designation B
        sec = sec_collection.retrieve_by_attr(
            'name', sections['gravity_beams_b'][level_tag])
        mcg.add_horizontal_active(
            point['A']['1'][0], point['A']['1'][1],
            point['B']['1'][0], point['B']['1'][1],
            np.array((0., 0., 0.)),
            np.array((0., 0., 0.)),
            'center_right',
            'middle_front',
            'Linear',
            n_sub,
            sec,
            element_type,
            'top_center',
            method='generate_hinged_component_assembly',
            additional_args={
                'zerolength_gen_i': gravity_shear_tab,
                'zerolength_gen_args_i': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                },
                'zerolength_gen_j': gravity_shear_tab,
                'zerolength_gen_args_j': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                }
            }
        )
        mcg.add_horizontal_active(
            point['E']['1'][0], point['E']['1'][1],
            point['F']['1'][0], point['F']['1'][1],
            np.array((0., 0., 0.)),
            np.array((0., 0., 0.)),
            'middle_back',
            'center_left',
            'Linear',
            n_sub,
            sec,
            element_type,
            'top_center',
            method='generate_hinged_component_assembly',
            additional_args={
                'zerolength_gen_i': gravity_shear_tab,
                'zerolength_gen_args_i': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                },
                'zerolength_gen_j': gravity_shear_tab,
                'zerolength_gen_args_j': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                }
            }
        )
        mcg.add_horizontal_active(
            point['A']['5'][0], point['A']['5'][1],
            point['B']['5'][0], point['B']['5'][1],
            np.array((0., 0., 0.)),
            np.array((0., 0., 0.)),
            'center_right',
            'middle_front',
            'Linear',
            n_sub,
            sec,
            element_type,
            'top_center',
            method='generate_hinged_component_assembly',
            additional_args={
                'zerolength_gen_i': gravity_shear_tab,
                'zerolength_gen_args_i': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                },
                'zerolength_gen_j': gravity_shear_tab,
                'zerolength_gen_args_j': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                }
            }
        )
        mcg.add_horizontal_active(
            point['E']['5'][0], point['E']['5'][1],
            point['F']['5'][0], point['F']['5'][1],
            np.array((0., 0., 0.)),
            np.array((0., 0., 0.)),
            'middle_back',
            'center_left',
            'Linear',
            n_sub,
            sec,
            element_type,
            'top_center',
            method='generate_hinged_component_assembly',
            additional_args={
                'zerolength_gen_i': gravity_shear_tab,
                'zerolength_gen_args_i': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                },
                'zerolength_gen_j': gravity_shear_tab,
                'zerolength_gen_args_j': {
                    'consider_composite': True,
                    'section': sec,
                    'physical_material': steel_phys_mat,
                    'distance': 10.00,
                    'n_sub': 1
                }
            }
        )

        # define gravity beams of designation C
        sec = sec_collection.retrieve_by_attr(
            'name', sections['gravity_beams_c'][level_tag])
        for tag1 in ['2', '3', '4']:
            mcg.add_horizontal_active(
                point['A'][tag1][0], point['A'][tag1][1],
                point['B'][tag1][0], point['B'][tag1][1],
                np.array((0., 0., 0.)),
                np.array((0., 0., 0.)),
                'center_right',
                'center_left',
                'Linear',
                n_sub,
                sec,
                element_type,
                'top_center',
                method='generate_hinged_component_assembly',
                additional_args={
                    'zerolength_gen_i': gravity_shear_tab,
                    'zerolength_gen_args_i': {
                        'consider_composite': True,
                        'section': sec,
                        'physical_material': steel_phys_mat,
                        'distance': 10.00,
                        'n_sub': 1
                    },
                    'zerolength_gen_j': gravity_shear_tab,
                    'zerolength_gen_args_j': {
                        'consider_composite': True,
                        'section': sec,
                        'physical_material': steel_phys_mat,
                        'distance': 10.00,
                        'n_sub': 1
                    }
                }
            )
            mcg.add_horizontal_active(
                point['E'][tag1][0], point['E'][tag1][1],
                point['F'][tag1][0], point['F'][tag1][1],
                np.array((0., 0., 0.)),
                np.array((0., 0., 0.)),
                'center_right',
                'center_left',
                'Linear',
                n_sub,
                sec,
                element_type,
                'top_center',
                method='generate_hinged_component_assembly',
                additional_args={
                    'zerolength_gen_i': gravity_shear_tab,
                    'zerolength_gen_args_i': {
                        'consider_composite': True,
                        'section': sec,
                        'physical_material': steel_phys_mat,
                        'distance': 10.00,
                        'n_sub': 1
                    },
                    'zerolength_gen_j': gravity_shear_tab,
                    'zerolength_gen_args_j': {
                        'consider_composite': True,
                        'section': sec,
                        'physical_material': steel_phys_mat,
                        'distance': 10.00,
                        'n_sub': 1
                    }
                }
            )

        # define gravity beams of designation D
        sec = sec_collection.retrieve_by_attr(
            'name', sections['gravity_beams_d'][level_tag])
        for tag1 in ['2', '3', '4']:
            tag2_start = ['B', 'C', 'D']
            tag2_end = ['C', 'D', 'E']
            for j in range(len(tag2_start)):
                mcg.add_horizontal_active(
                    point[tag2_start[j]][tag1][0], point[tag2_start[j]][tag1][1],
                    point[tag2_end[j]][tag1][0], point[tag2_end[j]][tag1][1],
                    np.array((0., 0., 0.)),
                    np.array((0., 0., 0.)),
                    'center_right',
                    'center_left',
                    'Linear',
                    n_sub,
                    sec,
                    element_type,
                    'top_center',
                    method='generate_hinged_component_assembly',
                    additional_args={
                        'zerolength_gen_i': gravity_shear_tab,
                        'zerolength_gen_args_i': {
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        },
                        'zerolength_gen_j': gravity_shear_tab,
                        'zerolength_gen_args_j': {
                            'consider_composite': True,
                            'section': sec,
                            'physical_material': steel_phys_mat,
                            'distance': 10.00,
                            'n_sub': 1
                        }
                    }
                )
        # define gravity beams of designation e
        sec = sec_collection.retrieve_by_attr(
            'name', sections['gravity_beams_e'][level_tag])
        for tag1 in ['A', 'B', 'C', 'D', 'E']:
            tag2_start = ['1', '2', '3', '4']
            tag2_end = ['2', '3', '4', '5']

            if tag1 in ['A', 'E']:
                shifts = 32.5/4. * 12.  # in
                num = 3
            else:
                shifts = 25.0/3. * 12  # in
                num = 2
            shift = 0.00
            for i in range(num):
                shift += shifts
                for j in range(len(tag2_start)):
                    pt_i = point[tag1][tag2_start[j]] + np.array([shift, 0.00])
                    pt_j = point[tag1][tag2_end[j]] + np.array([shift, 0.00])
                    sup_elm_i = querry.retrieve_component(
                        pt_i[0], pt_i[1], level_counter+1)
                    sup_elm_j = querry.retrieve_component(
                        pt_j[0], pt_j[1], level_counter+1)
                    mcg.add_horizontal_active(
                        pt_i[0], pt_i[1],
                        pt_j[0], pt_j[1],
                        np.array((0., -3., 0.)),
                        np.array((0., 3., 0.)),
                        'centroid',
                        'centroid',
                        'Linear',
                        n_sub,
                        sec,
                        element_type,
                        'top_center',
                        0.00,
                        sup_elm_i,
                        sup_elm_j)

    # fix base
    for node in mdl.levels[0].nodes.values():
        node.restraint = [True]*6

    # ~~~~~~~~~~~~ #
    # assign loads #
    # ~~~~~~~~~~~~ #

    loadcase = LoadCase('1.2D+0.25L+-E', mdl)
    self_weight(mdl, loadcase, factor=1.20)
    self_mass(mdl, loadcase)

    # surface loads
    for key in range(1, 1+num_levels):
        loadcase.tributary_area_analysis[key].polygon_loads.append(
            PolygonLoad('dead', surf_loads[key], None, None, False))
        loadcase.tributary_area_analysis[key].polygon_loads.append(
            PolygonLoad('dead', surf_loads_massless[key], None, None, True))
        loadcase.tributary_area_analysis[key].run(
            load_factor=1.20,
            massless_load_factor=0.25)

    # cladding loads
    def apply_cladding_load(
            coords, surf_load, surf_area, factor, massless=False):
        subset_model = mdl.initialize_empty_copy('subset_1')
        mdl.transfer_by_polygon_selection(subset_model, coords)
        # show(subset_model)
        elms = {}
        elm_lens = {}
        for comp in subset_model.list_of_components():
            if comp.component_purpose != 'steel_W_panel_zone':
                for elm in (comp.list_of_elastic_beamcolumn_elements()
                            + comp.list_of_disp_beamcolumn_elements()):
                    elms[elm.uid] = elm
                    elm_lens[elm.uid] = elm.clear_length()
        len_tot = sum(elm_lens.values())
        load = surf_load * surf_area
        line_load = load/len_tot
        from osmg.common import G_CONST_IMPERIAL
        for key, elm in elms.items():
            half_mass = line_load * elm_lens[key] / G_CONST_IMPERIAL
            loadcase.line_element_udl[key].add_glob(
                np.array((0.00, 0.00, - line_load*factor)))
            loadcase.node_mass[
                elm.nodes[0].uid].add([half_mass]*3+[0.00]*3)

    apply_cladding_load(
        np.array(
            [
                [-10.00, -50.00],
                [+1690.00, -50.00],
                [+1690.00, +50.00],
                [-10.00, +50.00]
            ]
        ),
        15.00/12.00**2,
        140.00*(15.00+13.00+13.00)*12.00**2,
        1.2
    )
    apply_cladding_load(
        np.array(
            [
                [-10.00, +1200.00-50.00],
                [+1690.00, +1200.00-50.00],
                [+1690.00, +1200.00+50.00],
                [-10.00, +1200.00+50.00]
            ]
        ),
        15.00/12.00**2,
        140.00*(15.00+13.00+13.00)*12.00**2,
        1.2
    )
    apply_cladding_load(
        np.array(
            [
                [-50.00, -50.00],
                [-50.00, 1250.00],
                [+50.00, 1250.00],
                [+50.00, -50.00]
            ]
        ),
        15.00/12.00**2,
        100.00*(15.00+13.00+13.00)*12.00**2,
        1.2
    )
    apply_cladding_load(
        np.array(
            [
                [+1680.00-50.00, -50.00],
                [+1680.00-50.00, 1250.00],
                [+1680.00+50.00, 1250.00],
                [+1680.00+50.00, -50.00]
            ]
        ),
        15.00/12.00**2,
        100.00*(15.00+13.00+13.00)*12.00**2,
        1.2
    )
    loadcase.rigid_diaphragms(
        [i for i in range(1, num_levels+1)], gather_mass=True)











    # define section groups and interpolated section objects
    family_14 = [
        'W14X26', 'W14X38', 'W14X48', 'W14X53', 'W14X68', 'W14X74',
        'W14X82', 'W14X132', 'W14X145', 'W14X159', 'W14X176', 'W14X193',
        'W14X211', 'W14X233', 'W14X257', 'W14X283', 'W14X311', 'W14X342',
        'W14X370', 'W14X398', 'W14X426', 'W14X455', 'W14X500', 'W14X550',
        'W14X605', 'W14X665', 'W14X730']
    family_16 = [
        'W16X31', 'W16X40', 'W16X45', 'W16X50', 'W16X57', 'W16X77',
        'W16X89', 'W16X100']
    family_18 = [
        'W18X35', 'W18X40', 'W18X46', 'W18X50', 'W18X55', 'W18X60',
        'W18X65', 'W18X71', 'W18X86', 'W18X97', 'W18X106', 'W18X119',
        'W18X130', 'W18X143', 'W18X158', 'W18X175', 'W18X192', 'W18X211',
        'W18X234', 'W18X258', 'W18X283', 'W18X311']
    family_21 = [
        'W21X44', 'W21X50', 'W21X57', 'W21X62', 'W21X68', 'W21X73',
        'W21X83', 'W21X93', 'W21X101', 'W21X111', 'W21X122', 'W21X132',
        'W21X147', 'W21X166', 'W21X182', 'W21X201']
    family_24 = [
        'W24X76', 'W24X84', 'W24X94', 'W24X103', 'W24X131', 'W24X146',
        'W24X162', 'W24X176', 'W24X192', 'W24X229', 'W24X207', 'W24X229',
        'W24X250', 'W24X279', 'W24X306', 'W24X335', 'W24X370']
    family_27 = [
        'W27X94', 'W27X102', 'W27X114', 'W27X129', 'W27X146', 'W27X161',
        'W27X178', 'W27X194', 'W27X217', 'W27X235', 'W27X258', 'W27X281',
        'W27X307', 'W27X336', 'W27X368', 'W27X539']
    family_30 = [
        'W30X108', 'W30X116', 'W30X124', 'W30X132', 'W30X148', 'W30X173',
        'W30X191', 'W30X211', 'W30X235', 'W30X261', 'W30X292', 'W30X326',
        'W30X357', 'W30X391']
    family_33 = [
        'W33X130', 'W33X141', 'W33X152', 'W33X169', 'W33X201', 'W33X221',
        'W33X241', 'W33X263', 'W33X291', 'W33X318', 'W33X354']





    # # previously used design:
    # #   weight:  54,441 lb


    # this solution:
    beams_1 = family_30
    beams_2 = family_30
    beams_3 = family_21
    cols_int = family_24
    cols_ext = family_18
    #   weight:  53,404 lb
    #   lvl: 1    2    3
    coeff = [3,   2,   1,  # beams
             7,   7,   6,  # interior
             11,  11,  10] # exterior








    columns_int = cols_int
    columns_ext = cols_ext


    # output flags
    get_doubler_plates = True
    get_beam_checks = True











    # selecting sections

    beam_coeff_lvl1 = coeff[0]
    beam_coeff_lvl2 = coeff[1]
    beam_coeff_lvl3 = coeff[2]

    col_int_coeff_lvl1 = coeff[3]
    col_int_coeff_lvl2 = coeff[4]
    col_int_coeff_lvl3 = coeff[5]

    col_ext_coeff_lvl1 = coeff[6]
    col_ext_coeff_lvl2 = coeff[7]
    col_ext_coeff_lvl3 = coeff[8]


    # initializing model

    mdl = Model('office_3_design')
    mdl.settings.imperial_units = True
    mcg = BeamColumnGenerator(mdl)
    secg = SectionGenerator(mdl)
    querry = ElmQuerry(mdl)

    rigidsec = secg.generate_generic_elastic(
        'rigidsec', 1.0e12, 1.0e12, 1.0e12)

    hi = np.array((15.00, 13.00+15.00, 13.00+13.00+15.00)) * 12.00  # in

    mdl.add_level(0, 0.00)
    for i, h in enumerate(hi):
        mdl.add_level(i+1, h)

    defaults.load_default_steel(mdl)
    defaults.load_default_fix_release(mdl)

    def section_from_coeff(coeff, list_of_section_names):
        secg.load_aisc_from_database(
            'W',
            [list_of_section_names[coeff]],
            'default steel', 'default steel',
            ElasticSection)
        res_sec = mdl.elastic_sections.retrieve_by_attr(
            'name', list_of_section_names[coeff])
        return res_sec

    beam_sec_lvl1 = section_from_coeff(beam_coeff_lvl1, beams_1)
    beam_sec_lvl2 = section_from_coeff(beam_coeff_lvl2, beams_2)
    beam_sec_lvl3 = section_from_coeff(beam_coeff_lvl3, beams_3)
    col_int_sec_lvl1 = section_from_coeff(col_int_coeff_lvl1, columns_int)
    col_int_sec_lvl2 = section_from_coeff(col_int_coeff_lvl2, columns_int)
    col_int_sec_lvl3 = section_from_coeff(col_int_coeff_lvl3, columns_int)
    col_ext_sec_lvl1 = section_from_coeff(col_ext_coeff_lvl1, columns_ext)
    col_ext_sec_lvl2 = section_from_coeff(col_ext_coeff_lvl2, columns_ext)
    col_ext_sec_lvl3 = section_from_coeff(col_ext_coeff_lvl3, columns_ext)

    beam_secs = {
        'level_1': beam_sec_lvl1,
        'level_2': beam_sec_lvl2,
        'level_3': beam_sec_lvl3
    }
    col_int_secs = {
        'level_1': col_int_sec_lvl1,
        'level_2': col_int_sec_lvl2,
        'level_3': col_int_sec_lvl3
    }
    col_ext_secs = {
        'level_1': col_ext_sec_lvl1,
        'level_2': col_ext_sec_lvl2,
        'level_3': col_ext_sec_lvl3
    }

    # define structural elements
    x_locs = np.array([0.00, 25.00, 50.00, 75.00]) * 12.00  # (in)

    for level_counter in range(3):
        level_tag = 'level_'+str(level_counter+1)
        mdl.levels.set_active([level_counter+1])
        for xpt in x_locs:
            if xpt in [0.00, 75.00*12.00]:
                sec = col_ext_secs[level_tag]
            else:
                sec = col_int_secs[level_tag]
            pt = np.array((xpt, 0.00))
            mcg.add_vertical_active(
                pt[0], pt[1],
                np.zeros(3), np.zeros(3),
                'Linear',
                1,
                sec,
                ElasticBeamColumn,
                'centroid',
                np.pi/2.00
            )
        for ipt_idx in range(len(x_locs)-1):
            pt_i = np.array((x_locs[ipt_idx], 0.00))
            pt_j = np.array((x_locs[ipt_idx+1], 0.00))
            sec = beam_secs[level_tag]
            mcg.add_horizontal_active(
                pt_i[0], pt_i[1],
                pt_j[0], pt_j[1],
                np.array((0., 0., 0.)),
                np.array((0., 0., 0.)),
                'centroid',
                'centroid',
                'Linear',
                1,
                sec,
                ElasticBeamColumn,
                'centroid'
            )

    # leaning column
    for level_counter in range(3):
        level_tag = 'level_'+str(level_counter+1)
        mdl.levels.set_active([level_counter+1])
        pt = np.array((100.00*12.00, 0.00))
        mcg.add_vertical_active(
            pt[0], pt[1],
            np.zeros(3), np.zeros(3),
            'Corotational',
            1,
            rigidsec,
            ElasticBeamColumn,
            'centroid',
            np.pi/2.00,
            method='generate_hinged_component_assembly',
            additional_args={
                'zerolength_gen_i': None,
                'zerolength_gen_args_i': {},
                'zerolength_gen_j': release_6,
                'zerolength_gen_args_j': {
                    'distance': 1.00,
                    'n_sub': 1
                },
            }
        )
        pt_i = np.array((x_locs[-1], 0.00))
        pt_j = pt
        mcg.add_horizontal_active(
            pt_i[0], pt_i[1],
            pt_j[0], pt_j[1],
            np.array((0., 0., 0.)),
            np.array((0., 0., 0.)),
            'centroid',
            'centroid',
            'Linear',
            1,
            rigidsec,
            ElasticBeamColumn,
            'centroid',
            method='generate_hinged_component_assembly',
            additional_args={
                'zerolength_gen_i': release_6,
                'zerolength_gen_args_i': {
                    'distance': 1.00,
                    'n_sub': 1
                },
                'zerolength_gen_j': release_6,
                'zerolength_gen_args_j': {
                    'distance': 1.00,
                    'n_sub': 1
                }
            }
        )


    p_nodes = [
        querry.search_node_lvl(0.00, 0.00, 1),
        querry.search_node_lvl(0.00, 0.00, 2),
        querry.search_node_lvl(0.00, 0.00, 3)
    ]


    # restrict motion in XZ plane
    for node in mdl.list_of_all_nodes():
        node.restraint = [False, True, False, True, False, True]
    # fix base
    for node in mdl.levels[0].nodes.values():
        node.restraint = [True]*6


    # ~~~~~~~~~~~~ #
    # assign loads #
    # ~~~~~~~~~~~~ #

    lc_dead = LoadCase('dead', mdl)
    self_mass(mdl, lc_dead)
    self_weight(mdl, lc_dead)
    lc_live = LoadCase('live', mdl)

    beam_udls_dead = {
        'level_1': 72.30,
        'level_2': 72.30,
        'level_3': 76.10
    }
    beam_udls_live = {
        'level_1': 29.90,
        'level_2': 29.90,
        'level_3': 12.00
    }
    lvl_weight = {
        'level_1': 793500.00,
        'level_2': 785000.00,
        'level_3': 817400.00
    }

    for level_counter in range(1, 3+1):
        level_tag = 'level_'+str(level_counter)
        for xpt in x_locs[:-1]:
            xpt += 30.00
            comp = querry.retrieve_component(xpt, 0.00, level_counter)
            for elm in comp.elastic_beamcolumn_elements.values():
                lc_dead.line_element_udl[elm.uid].add_glob(np.array(
                    (0.00, 0.00, -beam_udls_dead[level_tag])))
                lc_live.line_element_udl[elm.uid].add_glob(np.array(
                    (0.00, 0.00, -beam_udls_live[level_tag])))
        nd = querry.search_node_lvl(1200.00, 0.00, level_counter)
        lc_dead.node_loads[nd.uid].val += np.array(
            (0.00, 0.00, -lvl_weight[level_tag],
             0.00, 0.00, 0.00))
        mass = lvl_weight[level_tag] / common.G_CONST_IMPERIAL
        lc_dead.node_mass[nd.uid].val += np.array(
            (mass, mass, mass, 0.00, 0.00, 0.00))


    # ~~~~~~~~~~~~ #
    # run analyses #
    # ~~~~~~~~~~~~ #

    # dead and live static analysis

    static_anl = solver.StaticAnalysis(
        mdl,
        {
            'dead': lc_dead,
            'live': lc_live
        }
    )
    static_anl.run()

    # from osmg.graphics.postprocessing_3D import show_deformed_shape
    # from osmg.graphics.postprocessing_3D import show_basic_forces
    # show_deformed_shape(static_anl, 'dead', 0, 0.00, False)
    # show_basic_forces(static_anl, 'dead', 0, 1.00, 0.00, 0.00, 1.00e-1, 0.00, 10, global_axes=True)


    # earthquake - ELF

    # design parameters
    Cd = 5.5
    R = 8.0
    Ie = 1.0
    Sds = 1.58
    Sd1 = 1.38
    Tshort = Sd1/Sds
    max_drift = 0.02

    def k(T):
        if T <= 0.5:
            res = 1.0
        elif T >= 2.5:
            res = 2.0
        else:
            x = np.array([0.5, 2.5])
            y = np.array([1., 2.])
            f = interp1d(x, y)
            res = f(np.array([T]))[0]
        return res

    def Tmax(ct, exponent, height, Sd1):
        def cu(Sd1):
            if Sd1 <= 0.1:
                cu = 1.7
            elif Sd1 >= 0.4:
                cu = 1.4
            else:
                x = np.array([0.1, 0.15, 0.2, 0.3, 0.4])
                y = np.array([1.7, 1.6, 1.5, 1.4, 1.4])
                f = interp1d(x, y)
                cu = f(np.array([Sd1]))[0]
            return cu

        Ta = ct * height**exponent
        return cu(Sd1) * Ta

    def cs(T, Sds, Sd1, R, Ie):
        Tshort = Sd1/Sds
        if T < Tshort:
            res = Sds / R * Ie
        else:
            res = Sd1 / R * Ie / T
        return res

    # period estimation (Table 12.8-2)

    ct = 0.028
    exponent = 0.8
    T_max = Tmax(ct, exponent, hi[-1]/12.00, Sd1)

    # print('T_max = %.2f s\n' % (T_max))

    # modal period
    lc_modal = LoadCase('modal', mdl)
    lc_modal.node_mass = lc_dead.node_mass
    num_modes = 3

    modal_analysis = solver.ModalAnalysis(mdl, {'modal': lc_modal}, num_modes=num_modes)
    modal_analysis.run()
    ts = modal_analysis.results['modal'].periods

    # # mode shape
    # disps = np.zeros(len(p_nodes))
    # for i, p_node in enumerate(p_nodes):
    #     disps[i] = modal_analysis.results['modal'].node_displacements[p_node.uid][0][0]
    # disps /= disps[-1]
    # print(disps)

    # print(f'T_modal = {ts[0]:.2f} s\n')

    t_use = min(ts[0], T_max)
    wi = np.array(list(lvl_weight.values()))
    vb_elf = np.sum(wi) * cs(t_use, Sds, Sd1, R, Ie)
    print(f'Seismic weight: {np.sum(wi):.0f} kips')
    # print('V_b_elf = %.2f kips \n' % (vb_elf/1000))
    # print(f'Cs = {cs(t_use, Sds, Sd1, R, Ie)}')
    cvx = wi * hi**k(ts[1]) / np.sum(wi * hi**k(ts[1]))
    fx = vb_elf * cvx

    lc_elf = LoadCase('elf', mdl)
    for i, nd in enumerate(p_nodes):
        lc_elf.node_loads[nd.uid].add(np.array((fx[i], 0.00, 0.00, 0.00, 0.00, 0.00)))


    elf_anl = solver.StaticAnalysis(
        mdl,
        {
            'elf': lc_elf
        }
    )
    elf_anl.run()

    from osmg.postprocessing.design import LoadCombination
    elf_combo = LoadCombination(
        mdl,
        {
            '+-E': [(1.00, elf_anl, 'elf'), (-1.00, elf_anl, 'elf')]

    })
    # show_basic_forces_combo(
    #     elf_combo, 1.00, .00, .0, .0, .0, 50, global_axes=True,
    #     # force_conversion=1.00/1000.00,
    #     # moment_conversion=1.00/12.00/1000.00
    # )





    # Global stability (P-Delta Effects). ASCE 7-22 12.8.7
    # units used here: lb, in
    thetas = np.zeros(len(p_nodes))
    theta_lim = 0.10
    lvlw = np.array(list(lvl_weight.values()))
    for lvl_idx in range(len(p_nodes)):
        if lvl_idx == 0:
            deltax = np.max(np.abs([r[0] for r in elf_combo.envelope_node_displacement(p_nodes[lvl_idx])]))
        else:
            deltax = np.max(np.abs([r[0] for r in elf_combo.envelope_node_displacement_diff(p_nodes[lvl_idx], p_nodes[lvl_idx-1])]))
        px = np.sum(lvlw[lvl_idx:])
        vx = np.sum(fx[lvl_idx:])
        hsx = hi[lvl_idx]
        thetas[lvl_idx] = (px / hsx) / (vx/deltax)
    thetas/theta_lim  # should be < 1



    # multi-period design spectrum
    mlp_periods = np.array(
        (0.00, 0.01, 0.02, 0.03, 0.05,
         0.075, 0.1, 0.15, 0.2, 0.25,
         0.3, 0.4, 0.5, 0.75, 1.,
         1.5, 2., 3., 4., 5., 7.5, 10.))
    mlp_des_spc = np.array(
        (0.66, 0.66, 0.66, 0.67, 0.74,
         0.90, 1.03, 1.22, 1.36, 1.48,
         1.62, 1.75, 1.73, 1.51, 1.32,
         0.98, 0.77, 0.51, 0.35, 0.26,
         0.14, 0.083))


    design_spectrum_ifun = interp1d(mlp_periods, mlp_des_spc, kind='linear')

    rsa = solver.ModalResponseSpectrumAnalysis(
        mdl, lc_modal, num_modes,
        mlp_periods, mlp_des_spc, 'x')
    rsa.run()
    ts = rsa.anl.results['modal'].periods




    from osmg.postprocessing.design import LoadCombination
    drift_combo = LoadCombination(
        mdl,
        {
            'D+L': [(1.00, static_anl, 'dead'), (0.50*0.4, static_anl, 'live')],
            '+E': [(1.00/(R/Ie), rsa, 'modal')],
            '-E': [(-1.00/(R/Ie), rsa, 'modal')],
    })  # coeffs from ASCE 7-22 12.8.6.1

    # show_basic_forces_combo(
    #     drift_combo, 1.00, .00, .0, .0, .0, 50, global_axes=True,
    #     # force_conversion=1.00/1000.00,
    #     # moment_conversion=1.00/12.00/1000.00
    # )

    dr1 = np.max(np.abs([r[0] for r in drift_combo.envelope_node_displacement(p_nodes[0])])) / (15.*12.) * Cd / Ie
    dr2 = np.max(np.abs([r[0] for r in drift_combo.envelope_node_displacement_diff(p_nodes[1], p_nodes[0])])) / (13.*12.) * Cd / Ie
    dr3 = np.max(np.abs([r[0] for r in drift_combo.envelope_node_displacement_diff(p_nodes[2], p_nodes[1])])) / (13.*12.) * Cd / Ie


    strength_combo = LoadCombination(
        mdl,
        {
            'D+L': [(1.20, static_anl, 'dead'), (0.50*0.4, static_anl, 'live')],
            '+Ev': [(0.20*Sds, static_anl, 'dead')],
            '-Ev': [(-0.20*Sds, static_anl, 'dead')],
            '+Eh': [(1.00/(R/Ie), rsa, 'modal')],
            '-Eh': [(-1.00/(R/Ie), rsa, 'modal')],
    })
    # show_basic_forces_combo(
    #     strength_combo, 1.00, .00, .0, .0, .0, 50, global_axes=True,
    #     # force_conversion=1.00/1000.00,
    #     # moment_conversion=1.00/12.00/1000.00
    # )



    # strong column-weak beam check

    level_tags = [f'level_{i+1}' for i in range(3)]

    col_puc = {
        'exterior': {},
        'interior': {}}
    for i, level in enumerate(level_tags):
        comp = querry.retrieve_component(0.00, 0.00, i+1)
        elm = list(comp.elastic_beamcolumn_elements.values())[0]
        axial = np.abs(drift_combo.envelope_basic_forces(elm, 2)[0]['nx'].min())
        col_puc['exterior'][level] = axial
        comp = querry.retrieve_component(300.00, 0.00, i+1)
        elm = list(comp.elastic_beamcolumn_elements.values())[0]
        axial = np.abs(drift_combo.envelope_basic_forces(elm, 2)[0]['nx'].min())
        col_puc['interior'][level] = axial

    # strong column - weak beam
    # in the future : beam udls from combination, checks should be from module
    sh = 26.25  # in
    ext_res = []
    int_res = []
    for place in ['exterior', 'interior']:
        for level_num in range(len(level_tags)-1):
            this_level = level_tags[level_num]
            level_above = level_tags[level_num + 1]
            beam_sec = beam_secs[this_level].properties
            if place == 'exterior':
                col_sec = col_ext_secs[this_level].properties
                col_sec_above = col_ext_secs[level_above].properties
            else:
                col_sec = col_int_secs[this_level].properties
                col_sec_above = col_int_secs[level_above].properties
            if place == 'exterior':
                capacity = smrf_scwb(
                    col_sec, col_sec_above, beam_sec,
                    col_puc[place][this_level],
                    1.2 * beam_udls_dead[this_level] + 0.50*0.40*beam_udls_live[this_level],
                    0.60, hi[level_num], 25.00*12.00,
                    None, None, None, sh, 50000.00
                )
                ext_res.append(capacity)
            if place == 'interior':
                capacity = smrf_scwb(
                    col_sec, col_sec_above, beam_sec,
                    col_puc[place][this_level],
                    beam_udls_dead[this_level] + 0.50*0.40*beam_udls_live[this_level],
                    0.60, hi[level_num], 25.00*12.00,
                    beam_sec, 1.2 * beam_udls_dead[this_level] + 0.50*0.40*beam_udls_live[this_level]
                    , 0.60, sh, 50000.00
                )
                int_res.append(capacity)
    scwb_check = pd.DataFrame(
        {'exterior': ext_res, 'interior': int_res}, index=level_tags[:-1])
    print(scwb_check)

    if get_doubler_plates:
        # calculate doubler plate requirements
        ext_doubler_thickness = []
        int_doubler_thickness = []
        for place in ['exterior', 'interior']:
            for level_num in range(len(level_tags)):
                this_level = level_tags[level_num]
                beam_sec = beam_secs[this_level].properties
                if place == 'exterior':
                    col_sec = col_ext_secs[this_level].properties
                else:
                    col_sec = col_int_secs[this_level].properties
                if place == 'interior':
                    tdoub = smrf_pz_doubler_plate_requirement(
                        col_sec, beam_sec, 0.60,
                        25.00*12.00, 'interior', sh, 50000.00
                    )
                    int_doubler_thickness.append(tdoub)
                else:
                    tdoub = smrf_pz_doubler_plate_requirement(
                        col_sec, beam_sec, 0.60,
                        25.00*12.00, 'exterior', sh, 50000.00
                    )
                    ext_doubler_thickness.append(tdoub)
        pz_check = pd.DataFrame(
            {'exterior': ext_doubler_thickness,
             'interior': int_doubler_thickness},
            index=level_tags)
        print('Doubler Plate Requirement')
        print(pz_check)



    # check beam strength
    if get_beam_checks:
        for level_counter in range(1, 3+1):
            level_tag = 'level_'+str(level_counter)
            for xpt in x_locs[:-1]:
                xpt += 30.00
                comp = querry.retrieve_component(xpt, 0.00, level_counter)
                for elm in comp.elastic_beamcolumn_elements.values():
                    sec = elm.section.properties
                    strength_combo.envelope_basic_forces(elm, 2)[0]['mz']
                    strength_combo.envelope_basic_forces(elm, 2)[1]['mz']
                    rbs_proportion = 0.60
                    c_rbs_j = sec['bf'] * (1. - rbs_proportion) / 2.
                    z_rbs_j = (sec['Zx'] - 2. * c_rbs_j * sec['tf']
                               * (sec['d'] - sec['tf']))
                    fy = 50000.00
                    m_pr = fy * z_rbs_j
                    factor_i = abs(strength_combo.envelope_basic_forces(elm, 2)[0]['mz'])/m_pr
                    factor_j = abs(strength_combo.envelope_basic_forces(elm, 2)[1]['mz'])/m_pr
                    factor = pd.concat([factor_i, factor_j]).max()
                    print(f'{factor:.2f}')



    














    





























    

if __name__ == '__main__':
    pass
