import sys
sys.path.append("../OpenSees_Model_Builder/src")

import numpy as np
import model
import solver
import pickle
import matplotlib.pyplot as plt
from components import LineElementSequence

# Initialize a container
analysis_objects = []

def get_response(lat_bm_ends, lat_bm_modeling_type,
                 lat_col_ends, lat_col_modeling_type,
                 grav_bm_ends):

    b = model.Model()

    hi = np.array([15.00, 13.00, 13.00]) * 12.00  # in

    b.add_level("base", 0.00, "fixed")
    b.add_level("1", hi[0])
    b.add_level("2", hi[0]+hi[1])
    b.add_level("3", hi[0]+hi[1]+hi[2])

    sections = dict(
        gravity_cols=dict(
            level_1="W14X90",
            level_2="W14X90",
            level_3="W14X90"),
        gravity_beams_perimeter=dict(
            level_1="W21X55",
            level_2="W21X55",
            level_3="W21X55"),
        gravity_beams_interior_32=dict(
            level_1="W12X152",
            level_2="W12X152",
            level_3="W12X152"),
        gravity_beams_interior_25=dict(
            level_1="W10X100",
            level_2="W10X100",
            level_3="W10X100"),
        secondary_beams="W14X30",
        lateral_cols=dict(
            level_1="W14X426",
            level_2="W14X426",
            level_3="W14X342"),
        lateral_beams=dict(
            level_1="W24X192",
            level_2="W24X192",
            level_3="W24X94")
        )

    # define materials
    b.set_active_material('steel02-fy50')

    # define sections
    wsections = set()
    for lvl_tag in ['level_1', 'level_2', 'level_3']:
        wsections.add(sections['gravity_beams_perimeter'][lvl_tag])
        wsections.add(sections['gravity_beams_interior_32'][lvl_tag])
        wsections.add(sections['gravity_beams_interior_25'][lvl_tag])
        wsections.add(sections['lateral_cols'][lvl_tag])
        wsections.add(sections['lateral_beams'][lvl_tag])
        wsections.add(sections['gravity_cols'][lvl_tag])
    wsections.add(sections['secondary_beams'])

    for sec in wsections:
        b.add_sections_from_json(
            "../OpenSees_Model_Builder/section_data/sections.json",
            'W',
            [sec])

    #
    # define structural members
    #
    pinned_ends = {'type': 'pinned', 'end_dist': 0.005}
    fixedpinned_ends = {'type': 'fixed-pinned', 'end_dist': 0.005,
                        'doubler plate thickness': 0.00}
    elastic_modeling_type = {'type': 'elastic'}
    col_gtransf = 'Corotational'
    nsub = 8  # element subdivision
    grav_col_ends = fixedpinned_ends

    # generate a dictionary containing coordinates given gridline tag names
    # (here we won't use the native gridline objects,
    #  since the geometry is very simple)
    point = {}
    x_grd_tags = ['A', 'B', 'C', 'D', 'E', 'F']
    y_grd_tags = ['5', '4', '3', '2', '1']
    x_grd_locs = np.array([0.00, 32.5, 57.5, 82.5, 107.5, 140.00]) * 12.00  # (in)
    y_grd_locs = np.array([0.00, 25.00, 50.00, 75.00, 100.00]) * 12.00  # (in)

    for i in range(len(x_grd_tags)):
        point[x_grd_tags[i]] = {}
        for j in range(len(y_grd_tags)):
            point[x_grd_tags[i]][y_grd_tags[j]] = \
                np.array([x_grd_locs[i], y_grd_locs[j]])

    for level_counter in range(3):
        level_tag = 'level_'+str(level_counter+1)
        # define gravity columns
        b.set_active_angle(0.00)
        b.set_active_placement('centroid')
        b.set_active_levels([str(level_counter+1)])
        b.set_active_section(sections['gravity_cols'][level_tag])
        for tag in ['A', 'F']:
            pt = point[tag]['1']
            col = b.add_column_at_point(
                pt[0], pt[1], n_sub=1, ends=grav_col_ends,
                model_as=elastic_modeling_type, geom_transf=col_gtransf)
        for tag1 in ['B', 'C', 'D', 'E']:
            for tag2 in ['2', '3', '4']:
                pt = point[tag1][tag2]
                col = b.add_column_at_point(
                    pt[0], pt[1], n_sub=1, ends=grav_col_ends,
                    model_as=elastic_modeling_type, geom_transf=col_gtransf)

        # define X-dir frame columns
        b.set_active_section(sections['lateral_cols'][level_tag])
        b.set_active_angle(np.pi/2.00)
        for tag1 in ['B', 'C', 'D', 'E']:
            for tag2 in ['1', '5']:
                pt = point[tag1][tag2]
                b.add_column_at_point(
                    pt[0], pt[1], n_sub=nsub, ends=lat_col_ends,
                    model_as=lat_col_modeling_type, geom_transf=col_gtransf)
        # deffine Y-dir frame columns
        b.set_active_angle(0.00)
        for tag1 in ['A', 'F']:
            for tag2 in ['5', '4', '3', '2']:
                pt = point[tag1][tag2]
                b.add_column_at_point(
                    pt[0], pt[1], n_sub=nsub, ends=lat_col_ends,
                    model_as=lat_col_modeling_type, geom_transf=col_gtransf)
        # define X-dir frame beams
        b.set_active_section(sections['lateral_beams'][level_tag])
        b.set_active_placement('top_center')
        for tag1 in ['1', '5']:
            tag2_start = ['B', 'C', 'D']
            tag2_end = ['C', 'D', 'E']
            for j in range(len(tag2_start)):
                b.add_beam_at_points(
                    point[tag2_start[j]][tag1],
                    point[tag2_end[j]][tag1],
                    ends=lat_bm_ends,
                    model_as=lat_bm_modeling_type, n_sub=nsub,
                    snap_i='bottom_center',
                    snap_j='top_center')
        # define Y-dir frame beams
        for tag1 in ['A', 'F']:
            tag2_start = ['2', '3', '4']
            tag2_end = ['3', '4', '5']
            for j in range(len(tag2_start)):
                b.add_beam_at_points(
                    point[tag1][tag2_start[j]],
                    point[tag1][tag2_end[j]],
                    ends=lat_bm_ends,
                    model_as=lat_bm_modeling_type, n_sub=nsub,
                    snap_i='bottom_center',
                    snap_j='top_center')
        # define perimeter gravity beams
        b.set_active_section(sections['gravity_beams_perimeter'][level_tag])
        for tag1 in ['A', 'F']:
            tag2_start = ['1']
            tag2_end = ['2']
            for j in range(len(tag2_start)):
                b.add_beam_at_points(
                    point[tag1][tag2_start[j]],
                    point[tag1][tag2_end[j]],
                    ends=grav_bm_ends,
                    snap_i='bottom_center',
                    snap_j='top_center')
        b.add_beam_at_points(
            point['A']['1'],
            point['B']['1'],
            snap_j='top_center',
            ends=grav_bm_ends)
        b.add_beam_at_points(
            point['E']['1'],
            point['F']['1'],
            snap_i='bottom_center',
            ends=grav_bm_ends)
        b.add_beam_at_points(
            point['A']['5'],
            point['B']['5'],
            snap_j='top_center',
            ends=grav_bm_ends)
        b.add_beam_at_points(
            point['E']['5'],
            point['F']['5'],
            snap_i='bottom_center',
            ends=grav_bm_ends)
        # define interior gravity beams
        for tag1 in ['B', 'C', 'D', 'E']:
            b.set_active_section(
                sections['gravity_beams_interior_25'][level_tag])
            tag2_start = ['2', '3']
            tag2_end = ['3', '4']
            for j in range(len(tag2_start)):
                b.add_beam_at_points(
                    point[tag1][tag2_start[j]],
                    point[tag1][tag2_end[j]],
                    snap_i='bottom_center',
                    snap_j='top_center',
                    ends=grav_bm_ends)
            tag2_start = ['1']
            tag2_end = ['2']
            for j in range(len(tag2_start)):
                b.add_beam_at_points(
                    point[tag1][tag2_start[j]],
                    point[tag1][tag2_end[j]],
                    snap_j='top_center',
                    ends=grav_bm_ends)
            tag2_start = ['4']
            tag2_end = ['5']
            for j in range(len(tag2_start)):
                b.add_beam_at_points(
                    point[tag1][tag2_start[j]],
                    point[tag1][tag2_end[j]],
                    snap_i='bottom_center',
                    ends=grav_bm_ends)
        for tag1 in ['2', '3', '4']:
            tag2_start = ['A', 'B', 'C', 'D', 'E']
            tag2_end = ['B', 'C', 'D', 'E', 'F']
            for j in range(len(tag2_start)):
                if tag2_start[j] in ['B', 'E']:
                    b.set_active_section(
                        sections['gravity_beams_interior_32'][level_tag])
                else:
                    b.set_active_section(
                        sections['gravity_beams_interior_25'][level_tag])
                b.add_beam_at_points(
                    point[tag2_start[j]][tag1],
                    point[tag2_end[j]][tag1],
                    ends=grav_bm_ends)
        # define secondary beams
        b.set_active_section(sections['secondary_beams'])
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
                    b.add_beam_at_points(
                        point[tag1][tag2_start[j]] + np.array([shift, 0.00]),
                        point[tag1][tag2_end[j]] + np.array([shift, 0.00]),
                        offset_i=np.array([0., 0., -10.]),
                        offset_j=np.array([0., 0., -10.]),
                        ends=pinned_ends)

    #
    # define surface loads
    #


    b.set_active_levels(['1', '2'])
    b.assign_surface_DL((75.+15.+20.+0.25*80.)/(12.**2))

    b.set_active_levels(['3'])
    b.assign_surface_DL((75.+15.+80.+0.25*20)/(12.**2))


    # cladding - 1st story
    b.select_perimeter_beams_story('1')
    # 10 is the load in lb/ft2, we multiply it by the height
    # the tributary area of the 1st story cladding support is
    # half the height of the 1st story and half the height of the second
    # we get lb/ft, so we divide by 12 to convert this to lb/in
    # which is what OpenSees_Model_Builder uses.
    b.selection.add_UDL(np.array((0.00, 0.00,
                                  -((10./12.**2) * (hi[0] + hi[1]) / 2.00))))

    # cladding - 2nd story
    b.selection.clear()
    b.select_perimeter_beams_story('2')
    b.selection.add_UDL(np.array((0.00, 0.00,
                                  -((10./12.**2) * (hi[1] + hi[2]) / 2.00))))

    # cladding - roof
    b.selection.clear()
    b.select_perimeter_beams_story('3')
    b.selection.add_UDL(np.array((0.00, 0.00,
                                  -((10./12.**2) * hi[2] / 2.00))))
    b.selection.clear()

    # b.plot_building_geometry(extrude_frames=False,
    #                          offsets=True,
    #                          gridlines=True,
    #                          global_axes=False,
    #                          diaphragm_lines=True,
    #                          tributary_areas=True,
    #                          just_selection=False,
    #                          parent_nodes=True,
    #                          frame_axes=False)

    b.preprocess(assume_floor_slabs=True, self_weight=True,
                 steel_panel_zones=True, elevate_column_splices=0.25)

    b.plot_building_geometry(extrude_frames=False,
                             offsets=True,
                             gridlines=True,
                             global_axes=False,
                             diaphragm_lines=True,
                             tributary_areas=True,
                             just_selection=False,
                             parent_nodes=True,
                             frame_axes=False)

    
    # ~~~~~~~~~~~~~~~~ #
    #  modal analysis  #
    # ~~~~~~~~~~~~~~~~ #

    # # performing a linear modal analysis
    # modal_analysis = solver.ModalAnalysis(b, num_modes=3)
    # modal_analysis.run()

    # modal_analysis.deformed_shape(step=0, scaling=0.00, extrude_frames=True)
    # modal_analysis.deformed_shape(step=1, scaling=0.00, extrude_frames=True)
    
    # # retrieving textual results
    # print(modal_analysis.periods)
    # print(modal_analysis.table_shape(1))
    # print(modal_analysis.table_shape(2))
    # print(modal_analysis.table_shape(3))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    #  nonlinear pushover analysis  #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # performing a nonlinear pushover analysis
    pushover_analysis = solver.PushoverAnalysis(b)
    control_node = b.list_of_parent_nodes()[-1]  # top floor
    analysis_metadata = pushover_analysis.run(
        "x",
        np.array([50.0]),
        control_node,
        1./1., modeshape=np.array([0., 0.35, 0.68, 1.0]))

    # plot the deformed shape for any of the steps
    n_plot_steps = analysis_metadata['successful steps']
    plot_metadata = pushover_analysis.deformed_shape(
        step=n_plot_steps-1, scaling=0.00, extrude_frames=True)
    print(plot_metadata)

    # plot pushover curve
    pushover_analysis.plot_pushover_curve("x", control_node)

    deltas, vbs = pushover_analysis.table_pushover_curve('x', control_node)
    seismic_weight = np.sum(b.level_masses() * 386.22 / 1.e3)  # (kips)
    vbs /= seismic_weight * 1000.00
    total_height = (15. + 13.*2) * 12.00
    deltas /= total_height

    analysis_objects.append(pushover_analysis)

    return deltas, vbs, pushover_analysis


lat_bm_ends = {'type': 'steel_W_IMK', 'end_dist': 0.05,
               'Lb/ry': 60., 'L/H': 0.50, 'RBS_factor': 0.60,
               'composite action': False,
               'doubler plate thickness': 0.00}
lat_bm_modeling_type = {'type': 'elastic'}
lat_col_ends = {'type': 'steel_W_PZ_IMK', 'end_dist': 0.05,
                'Lb/ry': 60., 'L/H': 1.0, 'pgpye': 0.005,
                'doubler plate thickness': 0.00}
lat_col_modeling_type = {'type': 'elastic'}
grav_bm_ends = {'type': 'steel W shear tab', 'end_dist': 0.005,
                'composite action': False}
deltas_imk, vbs_imk, _ = get_response(lat_bm_ends, lat_bm_modeling_type,
                                      lat_col_ends, lat_col_modeling_type,
                                      grav_bm_ends)


# lat_bm_ends = {'type': 'steel_W_IMK', 'end_dist': 0.05,
#                'Lb/ry': 60., 'L/H': 0.50, 'RBS_factor': 0.60,
#                'composite action': True,
#                'doubler plate thickness': 0.00}
# lat_bm_modeling = {'type': 'elastic'}
# lat_col_ends = {'type': 'steel_W_PZ_IMK', 'end_dist': 0.05,
#                 'Lb/ry': 60., 'L/H': 1.0, 'pgpye': 0.05,
#                 'doubler plate thickness': 0.00}
# lat_col_modeling_type = {'type': 'elastic'}
# lat_bm_modeling_type = {'type': 'elastic'}
# grav_bm_ends = {'type': 'steel W shear tab', 'end_dist': 0.005,
#                 'composite action': False}
# deltas_imk_c, vbs_imk_c, analysis = get_response(
#     lat_bm_ends, lat_bm_modeling_type, lat_col_ends,
#     lat_col_modeling_type, grav_bm_ends)

# lat_bm_ends = {'type': 'steel_W_IMK', 'end_dist': 0.05,
#                'Lb/ry': 60., 'L/H': 0.50, 'RBS_factor': 0.60,
#                'composite action': True,
#                'doubler plate thickness': 0.00}
# lat_bm_modeling = {'type': 'elastic'}
# lat_col_ends = {'type': 'steel_W_PZ_IMK', 'end_dist': 0.05,
#                 'Lb/ry': 60., 'L/H': 1.0, 'pgpye': 0.05,
#                 'doubler plate thickness': 0.00}
# lat_col_modeling_type = {'type': 'elastic'}
# lat_bm_modeling_type = {'type': 'elastic'}
# grav_bm_ends = {'type': 'steel W shear tab', 'end_dist': 0.005,
#                 'composite action': True}
# deltas_imk_c2, vbs_imk_c2, analysis = get_response(
#     lat_bm_ends, lat_bm_modeling_type, lat_col_ends,
#     lat_col_modeling_type, grav_bm_ends)


# lat_bm_ends = {'type': 'RBS', 'end_dist': (17.50+17.5)/(25.*12.),
#                'rbs_length': 17.5, 'rbs_reduction': 0.60, 'rbs_n_sub': 15}
# lat_bm_modeling = {'type': 'fiber', 'n_x': 10, 'n_y': 25}
# lat_col_ends = {'type': 'steel_W_PZ', 'doubler plate thickness': 0.00, 'end_dist': 0.01}
# lat_col_modeling_type = {'type': 'fiber', 'n_x': 10, 'n_y': 25}
# grav_bm_ends = {'type': 'pinned', 'end_dist': 0.005}
# deltas_fib, vbs_fib, _ = get_response(lat_bm_ends, lat_bm_modeling,
#                                       lat_col_ends, lat_col_modeling_type,
#                                       grav_bm_ends)


# cs = 0.15115
# omEga = 3.00


# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='medium')
# plt.rc('ytick', labelsize='medium')
# plt.rc('text', usetex=False)

# plt.figure()
# plt.grid()
# plt.axhline(y=cs, color='0.50', ls='dashed')
# plt.axhline(y=cs*omEga, color='0.50', ls='dashed')
# plt.plot(deltas_fib, vbs_fib, color='k', ls='dotted', label='fiber')
# plt.plot(deltas_imk, vbs_imk, color='k', ls='solid', label='IMK')
# plt.plot(deltas_imk_c, vbs_imk_c, color='k',
#          ls='dashed', label='IMK, composite lat beams')
# plt.plot(deltas_imk_c2, vbs_imk_c2, color='k',
#          ls='dashed', label='IMK, composite all beams')
# plt.ylabel('Vb / W')
# plt.xlabel('Roof Drift Ratio $\\Delta$/H')
# plt.legend()
# plt.show()


# analysis = analysis_objects[0]

# b = analysis.building

# my_col = b.levels.level_list[1].columns.element_list[4]

# analysis.plot_moment_rot(my_col)




# # ~~~~~~~~~~~~~~~~~~ #
# # Panel zone springs #
# # ~~~~~~~~~~~~~~~~~~ #


# panel_zone_springs = []
# seqs = b.list_of_line_element_sequences()
# for seq in seqs:
#     if isinstance(seq, LineElementSequence_Steel_W_PanelZone):
#         panel_zone_springs.append(seq)
#     if isinstance(seq, LineElementSequence_Steel_W_PanelZone_IMK):
#         panel_zone_springs.append(seq)

# len(panel_zone_springs)
# springs = [spring.end_segment_i.internal_elems[8]
#            for spring in panel_zone_springs]

# spring = springs[14]

# data = []
# for i in range(analysis.n_steps_success):
#     data.append(
#         analysis.release_force_defo[spring.uid][i]
#     )

# data = np.array(data)


# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='medium')
# plt.rc('ytick', labelsize='medium')
# # plt.rc('text', usetex=True)
# plt.figure()
# plt.grid()
# plt.plot(data[:, 1], data[:, 0], color='k', ls='solid')
# plt.ylabel('Moment (lb-in)')
# plt.xlabel('Rotation (rad)')
# plt.show()

# # ~~~~~~~~~~~~~~~~~~~ #
# # IMK spring response #
# # ~~~~~~~~~~~~~~~~~~~ #

# # columns

# imk_springs = []
# seqs = b.list_of_line_element_sequences()
# for seq in seqs:
#     if isinstance(seq, LineElementSequence_Steel_W_PanelZone_IMK):
#         imk_springs.append(seq)

# len(imk_springs)
# springs = []
# for seq in imk_springs:
#     springs.append(seq.end_segment_i.internal_elems[12])
#     springs.append(seq.end_segment_j.internal_elems[0])

# jmax = 0
# ult_max = 0.00

# for j, spring in enumerate(springs):
#     data = []
#     for i in range(analysis.n_steps_success):
#         data.append(
#             analysis.release_force_defo[spring.uid][i]
#         )

#     data = np.array(data)
#     curr_max = np.max(np.abs(data[:, 1]))
#     if curr_max > ult_max:
#         ult_max = curr_max
#         jmax = j

# spring = springs[jmax]
# data = []
# for i in range(analysis.n_steps_success):
#     data.append(
#         analysis.release_force_defo[spring.uid][i]
#     )

# data = np.array(data)

# curr_max = np.max(np.abs(data[:, 1]))

# mat_properties = spring.materials[-1].parameters

# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='medium')
# plt.rc('ytick', labelsize='medium')
# # plt.rc('text', usetex=True)
# plt.figure()
# plt.grid()
# plt.axvline(x=mat_properties['theta_p+'], color='red', label='theta_p')
# plt.axvline(x=-mat_properties['theta_p-'], color='red')
# plt.axvline(x=mat_properties['theta_p+']+mat_properties['theta_pc+'],
#             color='green', label='theta_pc + theta_p')
# plt.axvline(x=-mat_properties['theta_p-']-mat_properties['theta_pc-'],
#             color='green')
# plt.axvline(x=mat_properties['theta_u'], color='purple', label='theta_u')
# plt.axvline(x=-mat_properties['theta_u'], color='purple')
# plt.axhline(y=mat_properties['my+']/1e3/12., color='cyan', label='my+')
# plt.axhline(y=mat_properties['my-']/1e3/12., color='cyan')
# plt.axhline(y=mat_properties['residual_plus'] / 1e3 / 12. *
#             mat_properties['my+'], color='purple', label='residual moment')
# plt.axhline(y=mat_properties['residual_minus'] / 1e3 / 12. *
#             mat_properties['my-'], color='purple')
# plt.plot(data[:, 1], data[:, 0]/(1.e3 * 12.), color='k', ls='solid')
# plt.ylabel('Moment (kip-ft)')
# plt.xlabel('Rotation (rad)')
# plt.show()

# # beams

# imk_springs = []
# for seq in seqs:
#     if isinstance(seq, LineElementSequence_IMK):
#         imk_springs.append(seq)

# len(imk_springs)
# springs = []
# for seq in imk_springs:
#     springs.append(seq.end_segment_i.internal_elems[1])
#     springs.append(seq.end_segment_j.internal_elems[0])

# jmax = 0
# ult_max = 0.00

# for j, spring in enumerate(springs):
#     data = []
#     for i in range(analysis.n_steps_success):
#         data.append(
#             analysis.release_force_defo[spring.uid][i]
#         )

#     data = np.array(data)
#     curr_max = np.max(np.abs(data[:, 1]))
#     if curr_max > ult_max:
#         ult_max = curr_max
#         jmax = j

# spring = springs[jmax]
# data = []
# for i in range(analysis.n_steps_success):
#     data.append(
#         analysis.release_force_defo[spring.uid][i]
#     )

# data = np.array(data)

# curr_max = np.max(np.abs(data[:, 1]))

# mat_properties = spring.materials[-1].parameters

# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='medium')
# plt.rc('ytick', labelsize='medium')
# # plt.rc('text', usetex=True)
# plt.figure()
# plt.grid()
# plt.axvline(x=mat_properties['theta_p+'], color='red', label='theta_p')
# plt.axvline(x=-mat_properties['theta_p-'], color='red')
# plt.axvline(x=mat_properties['theta_p+']+mat_properties['theta_pc+'],
#             color='green', label='theta_pc + theta_p')
# plt.axvline(x=-mat_properties['theta_p-']-mat_properties['theta_pc-'],
#             color='green')
# plt.axvline(x=mat_properties['theta_u'], color='purple', label='theta_u')
# plt.axvline(x=-mat_properties['theta_u'], color='purple')
# plt.axhline(y=mat_properties['my+']/1e3/12., color='cyan', label='my+')
# plt.axhline(y=mat_properties['my-']/1e3/12., color='cyan')
# plt.axhline(y=mat_properties['residual_plus'] / 1e3 / 12. *
#             mat_properties['my+'], color='purple', label='residual moment')
# plt.axhline(y=mat_properties['residual_minus'] / 1e3 / 12. *
#             mat_properties['my-'], color='purple')
# plt.plot(data[:, 1], data[:, 0]/(1.e3 * 12.), color='k', ls='solid')
# plt.ylabel('Moment (kip-ft)')
# plt.xlabel('Rotation (rad)')
# plt.show()


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# # gravity shear tab connection #
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# grav_springs = []
# seqs = analysis.building.list_of_line_element_sequences()
# for seq in seqs:
#     if isinstance(seq, LineElementSequence_W_grav_sear_tab):
#         grav_springs.append(seq)

# len(grav_springs)
# springs = []
# for seq in grav_springs:
#     springs.append(seq.end_segment_i.internal_elems[1])
#     springs.append(seq.end_segment_j.internal_elems[0])

# jmax = 0
# ult_max = 0.00

# for j, spring in enumerate(springs):
#     data = []
#     for i in range(analysis.n_steps_success):
#         data.append(
#             analysis.release_force_defo[spring.uid][i]
#         )

#     data = np.array(data)
#     curr_max = np.max(np.abs(data[:, 0]))
#     if curr_max > ult_max:
#         ult_max = curr_max
#         jmax = j

# spring = springs[jmax]
# data = []
# for i in range(analysis.n_steps_success):
#     data.append(
#         analysis.release_force_defo[spring.uid][i]
#     )

# data = np.array(data)

# curr_max = np.max(np.abs(data[:, 1]))

# mat_properties = spring.materials[-1].parameters

# backbone_x = [2*mat_properties['th_4_n'],
#               mat_properties['th_4_n'],
#               mat_properties['th_3_n'],
#               mat_properties['th_2_n'],
#               mat_properties['th_1_n'],
#               0.00,
#               mat_properties['th_1_p'],
#               mat_properties['th_2_p'],
#               mat_properties['th_3_p'],
#               mat_properties['th_4_p'],
#               2. * mat_properties['th_4_p']]
# backbone_y = [mat_properties['m4_n'],
#               mat_properties['m4_n'],
#               mat_properties['m3_n'],
#               mat_properties['m2_n'],
#               mat_properties['m1_n'],
#               0.00,
#               mat_properties['m1_p'],
#               mat_properties['m2_p'],
#               mat_properties['m3_p'],
#               mat_properties['m4_p'],
#               mat_properties['m4_p']]

# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='medium')
# plt.rc('ytick', labelsize='medium')
# plt.figure(figsize=(8, 4))
# plt.grid()
# plt.plot(data[:, 1], data[:, 0], color='blue',
#          ls='solid', linewidth=0.5)
# plt.scatter(data[23, 1], data[23, 0])
# # plt.plot(backbone_x, backbone_y, 'red', ls='dashed', label='backbone')
# # plt.axhline(y=sec_mp, label='Mult')
# # plt.axhline(y=-sec_mp)
# plt.ylabel('Moment $M$ (kip-in)')
# plt.xlabel('Rotation $\phi$ (rad)')
# plt.legend()
# plt.show()
# # plt.savefig('figure.pdf')
# plt.close()

# nid3 = analysis.building.list_of_parent_nodes()[-1].uid
# nid2 = analysis.building.list_of_parent_nodes()[-2].uid
# u3 = analysis.node_displacements[nid3][23][0]
# u2 = analysis.node_displacements[nid2][23][0]
# (u3-u2)/(13*12)
