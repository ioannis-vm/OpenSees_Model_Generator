import sys
sys.path.append("../OpenSees_Model_Builder/src")

import numpy as np
import matplotlib.pyplot as plt
import model
import solver

hi = np.array([15.00, 13.00, 13.00]) * 12.00  # in
weight = dict(
    level_1=1957.64 * 1e3 / 2.,
    level_2=1929.68 * 1e3 / 2.,
    level_3=2677.99 * 1e3 / 2.)
seismic_weight = np.sum(list(weight.values()))/1.e3 * 2.00  # (kips)
cs = 0.1478
omEga = 3.00
total_height = (15. + 13.*2) * 12.00


def define_building(lat_bm_ends, lat_bm_modeling,
                    lat_col_ends, lat_col_modeling_type):

    # Define a building
    b = model.Model()

    # Add levels
    b.add_level("base", 0.00, "fixed")
    b.add_level("1", hi[0])
    b.add_level("2", hi[0]+hi[1])
    b.add_level("3", hi[0]+hi[1]+hi[2])

    sections = dict(
        lateral_cols=dict(
            level_1="W14X455",
            level_2="W14X455",
            level_3="W14X370"),
        lateral_beams=dict(
            level_1="W24X229",
            level_2="W24X207",
            level_3="W21X122"),
        )

    pinned_ends = {'type': 'pinned', 'dist': 0.005}
    fixedpinned_ends = {'type': 'fixed-pinned', 'dist': 0.005}
    elastic_modeling_type = {'type': 'elastic'}
    col_gtransf = 'Corotational'
    nsub = 8  # element subdivision

    # define materials
    b.set_active_material('steel02-fy50')

    # define sections
    wsections = set()
    for lvl_tag in ['level_1', 'level_2', 'level_3']:
        wsections.add(sections['lateral_cols'][lvl_tag])
        wsections.add(sections['lateral_beams'][lvl_tag])

    for sec in wsections:
        b.add_sections_from_json(
            "../OpenSees_Model_Builder/section_data/sections.json",
            'W',
            [sec])

    #
    # define structural members
    #

    # generate a dictionary containing coordinates given gridline tag names
    # (here we won't use the native gridline objects,
    #  since the geometry is very simple)
    point = {}
    x_grd_tags = ['A', 'B', 'C', 'D', 'E', 'F']
    y_grd_tags = ['5', '4', '3', '2', '1']
    x_grd_locs = np.array([0.00, 32.5, 57.5, 82.5, 107.5, 140.00]) * 12.00
    y_grd_locs = np.array([0.00, 25.00, 50.00, 75.00, 100.00]) * 12.00
    for i in range(len(x_grd_tags)):
        point[x_grd_tags[i]] = {}
        for j in range(len(y_grd_tags)):
            point[x_grd_tags[i]][y_grd_tags[j]] = \
                np.array([x_grd_locs[i], y_grd_locs[j]])

    nis = []
    for level_counter in range(3):
        level_tag = 'level_'+str(level_counter+1)
        b.set_active_placement('centroid')
        b.set_active_levels([str(level_counter+1)])
        # define X-dir frame columns
        b.set_active_section(sections['lateral_cols'][level_tag])
        b.set_active_angle(np.pi/2.00)

        for tag1 in ['B', 'C', 'D', 'E']:
            for tag2 in ['1']:
                pt = point[tag1][tag2]
                b.add_column_at_point(
                    pt[0], pt[1], n_sub=nsub,
                    ends=lat_col_ends,
                    model_as=lat_col_modeling_type, geom_transf=col_gtransf)
        # define X-dir frame beams
        b.set_active_angle(0.00)
        b.set_active_section(sections['lateral_beams'][level_tag])
        b.set_active_placement('top_center')
        for tag1 in ['1']:
            tag2_start = ['B', 'C', 'D']
            tag2_end = ['C', 'D', 'E']
            for j in range(len(tag2_start)):
                b.add_beam_at_points(
                    point[tag2_start[j]][tag1],
                    point[tag2_end[j]][tag1],
                    snap_i='bottom_center',
                    snap_j='top_center',
                    ends=lat_bm_ends,
                    model_as=lat_bm_modeling, n_sub=nsub)
        # define leaning column
        b.set_active_section('rigid')
        for tag1 in ['1']:
            tag2_start = ['E']
            tag2_end = ['F']
            for j in range(len(tag2_start)):
                b.add_beam_at_points(
                    point[tag2_start[j]][tag1],
                    point[tag2_end[j]][tag1],
                    ends=pinned_ends,
                    model_as=elastic_modeling_type, n_sub=1)
        b.set_active_placement('centroid')
        for tag1 in ['F']:
            for tag2 in ['1']:
                pt = point[tag1][tag2]
                col = b.add_column_at_point(
                    pt[0], pt[1], n_sub=1, ends=fixedpinned_ends,
                    model_as=elastic_modeling_type, geom_transf=col_gtransf)
                ni = col[0].node_i
                nis.append(ni)

    # b.plot_building_geometry(extrude_frames=True)
    # b.plot_building_geometry(extrude_frames=False, frame_axes=False)

    b.preprocess(assume_floor_slabs=False, self_weight=True,
                 steel_panel_zones=True, elevate_column_splices=0.25)

    for level_counter, ni in enumerate(nis):
        level_tag = 'level_'+str(level_counter+1)
        ww = -weight[level_tag]
        mm = -ww / 386.22
        ni.load += np.array([0.00, 0.00, ww,
                             0.00, 0.00, 0.00])
        ni.mass += np.array([mm, mm, mm])

    # b.plot_building_geometry(extrude_frames=True)
    # b.plot_building_geometry(extrude_frames=False, frame_axes=False)

    pushover_analysis = solver.PushoverAnalysis(b)
    control_node = nis[-1]  # top floor
    analysis_metadata = pushover_analysis.run(
        "x",
        np.array([50.]),
        control_node,
        1./1.)

    deltas, vbs = pushover_analysis.table_pushover_curve('x', control_node)
    vbs *= 2.00 / (seismic_weight * 1.e3)
    deltas /= total_height

    # n_plot_steps = analysis_metadata['successful steps']
    # plot_metadata = pushover_analysis.deformed_shape(
    #     step=n_plot_steps-1, scaling=0.00, extrude_frames=False)
    # print(plot_metadata)

    return deltas, vbs


fiber_modeling_type = {'type': 'fiber', 'n_x': 10, 'n_y': 25}

deltas_imk, vbs_imk = define_building(
    lat_bm_ends={'type': 'steel_W_IMK', 'dist': 0.05,
                 'Lb/ry': 60., 'L/H': 0.50, 'RBS_factor': 0.60,
                 'composite action': False,
                 'doubler plate thickness': 0.00},
    lat_bm_modeling={'type': 'elastic'},
    lat_col_ends={'type': 'steel_W_PZ_IMK', 'dist': 0.05,
                  'Lb/ry': 60., 'L/H': 1.0, 'pgpye': 0.20,
                  'doubler plate thickness': 0.00},
    lat_col_modeling_type={'type': 'elastic'})

deltas_imk_comp, vbs_imk_comp = define_building(
    lat_bm_ends={'type': 'steel_W_IMK', 'dist': 0.05,
                 'Lb/ry': 60., 'L/H': 0.50, 'RBS_factor': 0.60,
                 'composite action': True,
                 'doubler plate thickness': 0.00},
    lat_bm_modeling={'type': 'elastic'},
    lat_col_ends={'type': 'steel_W_PZ_IMK', 'dist': 0.05,
                  'Lb/ry': 60., 'L/H': 1.0, 'pgpye': 0.20,
                  'doubler plate thickness': 0.00},
    lat_col_modeling_type={'type': 'elastic'})

deltas_fiber, vbs_fiber = define_building(
    lat_bm_ends={'type': 'RBS', 'dist': (17.50+17.5)/(25.*12.),
                 'length': 17.5, 'factor': 0.60, 'n_sub': 15},
    lat_bm_modeling={'type': 'fiber', 'n_x': 10, 'n_y': 25},
    lat_col_ends={'type': 'steel_W_PZ', 'doubler plate thickness': 0.00},
    lat_col_modeling_type={'type': 'fiber', 'n_x': 10, 'n_y': 25})


# # strength check
# mplW14x455 = 936. * 55.
# mplW24x229 = 675. * 55.
# mplW24x207 = 606. * 55.
# mplW21x122 = 307. * 55.
# cvx1 = 0.134
# cvx2 = 0.270
# cvx3 = 0.596
# h1 = 15. * 12.
# h2 = 28. * 12.
# h3 = 41. * 12.
# VyExp = 2.00 * ((4.*mplW14x455 +
#                  6.*(mplW24x229+mplW24x207+mplW21x122)) /
#                 (cvx1*h1 + cvx2 * h2 + cvx3 * h3)
#                 )


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
# plt.rc('text', usetex=True)

plt.figure()
plt.grid()
plt.axhline(y=cs, color='0.50', ls='dashed')
plt.axhline(y=cs*omEga, color='0.50', ls='dashed')
# plt.axhline(y=VyExp/seismic_weight, color='r', linestyle='--')
plt.plot(deltas_fiber, vbs_fiber, color='k', ls='dotted', label='fiber')
plt.plot(deltas_imk, vbs_imk, color='k', ls='solid', label='IMK')
plt.plot(deltas_imk_comp, vbs_imk_comp, color='k',
         ls='dashed', label='IMK, composite')
plt.ylabel('Vb / W')
plt.xlabel('Roof Drift Ratio $\\Delta$/H')
plt.legend()
plt.show()

# pushover_analysis.basic_forces(step=19)

 
