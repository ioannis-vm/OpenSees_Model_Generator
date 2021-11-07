import sys
sys.path.append("../OpenSeesPy_Building_Modeler")

import numpy as np
import modeler
import solver
import matplotlib.pyplot as plt


hi = np.array([15.00, 13.00, 13.00]) * 12.00  # in
weight = dict(
    level_1=2238.*1e3/2.,
    level_2=2210.*1e3/2.,
    level_3=2748.*1e3/2.)
seismic_weight = np.sum(list(weight.values()))/1.e3 * 2.00  # (kips)
total_height = (15. + 13.*2) * 12.00


def define_building(lat_bm_ends):

    # Define a building
    b = modeler.Building()
    b.set_global_restraints([0, 1, 0, 0, 0, 0])

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

    pinned_ends = {'type': 'pinned', 'dist': 0.001}
    fixedpinned_ends = {'type': 'fixed-pinned', 'dist': 0.05}
    elastic_modeling_type = {'type': 'elastic'}
    fiber_modeling_type = {'type': 'fiber', 'n_x': 10, 'n_y': 25}
    col_gtransf = 'Corotational'
    lat_col_ends = {'type': 'fixed', 'doubler plate thickness': 0.}
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
            "../OpenSeesPy_Building_Modeler/section_data/sections.json",
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
    x_grd_locs = np.array([0.00, 32.5, 57.5,
                           82.5, 107.5, 140.00]) * 12.00  # (in)
    y_grd_locs = np.array([0.00, 25.00, 50.00, 75.00, 100.00]) * 12.00  # (in)
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
                    model_as=fiber_modeling_type,
                    geomTransf=col_gtransf,
                    ends=lat_col_ends)
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
                    model_as=fiber_modeling_type, n_sub=nsub)
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
                    model_as=elastic_modeling_type, geomTransf=col_gtransf)
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

    b.plot_building_geometry(extrude_frames=True)
    b.plot_building_geometry(extrude_frames=False, frame_axes=False)

    control_node = nis[-1]  # top floor

    nlth = solver.NLTHAnalysis(b)

    nlth.run(0.01,
             'examples/groundmotions/1xa.txt',
             None,
             None, 0.005)

    time, displ = nlth.table_node__history(control_node, 0, 'Displacement')
    time, accel = nlth.table_node__history(control_node, 0, 'Acceleration')

    return time, displ, accel


time_fiber, displ_fiber, accel_fiber = define_building(
    lat_bm_ends={'type': 'RBS', 'dist': (17.50+17.5)/(25.*12.),
                 'length': 17.5, 'factor': 0.60, 'n_sub': 15})

time_imk, displ_imk, accel_imk = define_building(
    lat_bm_ends={'type': 'steel_W_IMK', 'dist': 0.05,
                 'Lb/ry': 60., 'L/H': 0.50, 'RBS_factor': 0.60,
                 'composite_action': False})


plt.figure()
plt.grid()
plt.plot(time_fiber, displ_fiber, 'blue', label='fiber')
plt.plot(time_imk, displ_imk, 'red', label='IMK')
plt.ylabel('Displacement (in)')
plt.xlabel('Time (s)')
plt.legend()
plt.show()

plt.figure()
plt.grid()
plt.plot(time_fiber, accel_fiber, 'blue', label='fiber')
plt.plot(time_imk, accel_imk, 'red', label='IMK')
plt.ylabel('Acceleration (g)')
plt.xlabel('Time (s)')
plt.legend()
plt.show()
