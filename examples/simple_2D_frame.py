import sys
sys.path.append("../OpenSeesPy_Building_Modeler")

import numpy as np
import modeler
import solver
import pickle
import matplotlib.pyplot as plt


b = modeler.Building()

b.set_global_restraints([0, 1, 0, 1, 0, 1])

b.add_level("base", 0.00, "fixed")
b.add_level("1", 12. * 12.)

col_sec = 'W14X370'
beam_sec = 'W24X94'


b.set_active_material('steel02-fy50')

b.add_sections_from_json(
    "../OpenSeesPy_Building_Modeler/section_data/sections.json",
    'W',
    [col_sec, beam_sec])

# pinned_ends = {'type': 'pinned', 'dist': 0.005}
elastic_modeling_type = {'type': 'elastic'}
col_gtransf = 'Corotational'
nsub = 8  # element subdivision
col_ends = {'type': 'steel_W_PZ_IMK', 'dist': 0.05,
            'Lb/ry': 60., 'L/H': 1.0, 'pgpye': 0.20,
            'doubler plate thickness': 0.00}
bm_ends = {'type': 'steel_W_IMK', 'dist': 0.05,
           'Lb/ry': 60., 'L/H': 0.50, 'RBS_factor': 0.60,
           'composite action': True,
           'doubler plate thickness': 0.00}
bm_modeling = {'type': 'elastic'}

pt1 = np.array([0., 0.])
pt2 = np.array([25. * 12., 0.])

b.set_active_angle(np.pi/2.00)
b.set_active_placement('centroid')
b.set_active_levels(['1'])
b.set_active_section(col_sec)
col1 = b.add_column_at_point(
    pt1[0], pt1[1], n_sub=nsub, ends=col_ends,
    model_as=elastic_modeling_type, geomTransf=col_gtransf)[0]
col2 = b.add_column_at_point(
    pt2[0], pt2[1], n_sub=nsub, ends=col_ends,
    model_as=elastic_modeling_type, geomTransf=col_gtransf)[0]
b.set_active_placement('top_center')
b.set_active_section(beam_sec)
b.set_active_angle(0.00)
beam = b.add_beam_at_points(
    pt1, pt2, n_sub=nsub,
    snap_i='bottom_center',
    snap_j='top_center',
    ends=bm_ends,
    model_as=bm_modeling)[0]


b.preprocess(assume_floor_slabs=False, self_weight=True,
             steel_panel_zones=True, elevate_column_splices=0.25)

b.plot_building_geometry(extrude_frames=True)

col = b.levels.level_list[1].columns.element_list[0]
control_node = beam.node_i


pushover_analysis = solver.PushoverAnalysis(b)
analysis_metadata = pushover_analysis.run(
    "x",
    np.array([.5]),
    control_node,
    1./1.)
n_plot_steps = analysis_metadata['successful steps']

plot the deformed shape for any of the steps
plot_metadata = pushover_analysis.deformed_shape(
    step=n_plot_steps-1, scaling=0.00, extrude_frames=False)
print(plot_metadata)

pushover_analysis.basic_forces(n_plot_steps-1)
pushover_analysis.plot_moment_rot(col)
