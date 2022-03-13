import sys
sys.path.append("../OpenSees_Model_Builder/src")

import model
import solver
import numpy as np
import matplotlib.pyplot as plt

# Define a building
b = model.Model()

# Add levels
b.add_level("base", 0.00, "fixed")
b.add_level("1", 6.0 * 12.00)
b.add_level("2", 10.0 * 12.00)

b.set_active_material('steel-bilinear-fy50')

b.add_sections_from_json(
    "section_data/sections.json",
    'W',
    ["W14X120", "W14X132"])

b.set_active_section("W14X120")


p0 = np.array([1., 1.]) * 12
p1 = np.array([2.0, 2.90]) * 12

b.set_active_levels("all_above_base")

b.set_active_angle(-(90-62.241)/180.*np.pi)
col = b.add_column_at_point(
    p0[0], p0[1], n_sub=10,
    geom_transf='Corotational',
    ends={'type': 'steel_W_PZ',
          'doubler plate thickness': 0.0})
b.set_active_angle(0.00)
b.set_active_placement('top_center')
b.set_active_levels(['1'])
b.set_active_section('W14X132')
beams = b.add_beam_at_points(p0, p1, snap_i='top_center',
                             n_sub=10)

# fix the top node too
col.node_i.restraint_type='fixed'

# before preprocessing
# b.plot_building_geometry(extrude_frames=False)
# b.plot_building_geometry(extrude_frames=True)

b.preprocess(assume_floor_slabs=False, self_weight=True,
             steel_panel_zones=True)

# # after preprocessing
b.plot_building_geometry(extrude_frames=True)
b.plot_building_geometry(extrude_frames=False)

# # ~~~~~~~~~~~~~~~~~ #
# #  linear analysis  #
# # ~~~~~~~~~~~~~~~~~ #

# control_node = beams[0].node_i

# control_node.load += np.array([100.00, 30.00, -100000.00, 0.00, 0.00, 0.00])

# linear_gravity_analysis = solver.LinearGravityAnalysis(b)
# linear_gravity_analysis.run()

# linear_gravity_analysis.deformed_shape(extrude_frames=False)
# linear_gravity_analysis.deformed_shape(extrude_frames=True)


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# #  nonlinear pushover analysis  #
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# control_node.load -= np.array([100.00, 30.00, -100000.00, 0.00, 0.00, 0.00])

# control_node = beams[0].node_i

# pushover_analysis = solver.PushoverAnalysis(b)

# vals = []
# for i in range(1, 10):
#     vals.append(i/10)
#     vals.append(-i/10)
# vals = np.array(vals) * 1.

# analysis_metadata = pushover_analysis.run(
#     "z",
#     vals,
#     control_node,
#     1./50.,
#     loaded_node=control_node)
# n_plot_steps = analysis_metadata['successful steps']

# # plot the deformed shape
# plot_metadata = pushover_analysis.deformed_shape(
#     step=n_plot_steps-1, scaling=0.00, extrude_frames=True)
# print(plot_metadata)

# displ, force = pushover_analysis.table_pushover_curve('z', control_node)

# plt.figure()
# plt.plot(displ, force/1000.00)
# plt.show()

