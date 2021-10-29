import modeler
import solver
import numpy as np

# Define a building
b = modeler.Building()

# Add levels
b.add_level("base", 0.00, "fixed")
# b.add_level("1", 144.00)

for i in range(3):
    b.add_level(str(i+1), 144.00*(i+1))


# add girdlines
# getting variables that reference them is optional
g1 = b.add_gridline("1", [0., 0.], [360.00, 0.])
b.add_gridline("2", [0., 288.00], [360.00, 288.00])
gA = b.add_gridline("A", [0., 0.], [0., 288.00])
b.add_gridline("B", [360.00, 0.], [360.00, 288.00])

# add groups
b.add_group("beams")
b.add_group("columns")


# define materials
b.set_active_material('steel-bilinear-fy50')
# solver.plot_stress_strain(b.materials.active, 100, 50000)

# define sections
b.add_sections_from_json(
    "section_data/sections.json",
    'W',
    ["W24X94"])
b.set_active_section("W24X94")
# b.add_sections_from_json(
#     "section_data/sections.json",
#     'HSS',
#     ["HSS20.000X0.375"])


# Modeling procedure:
# - [ ] Set active {section, placement, angle}
# - [ ] Set active levels and groups
# - [ ] Define elements
# - [ ] Repeat
# - [ ] In the end, preprocess building

b.set_active_levels("all_above_base")

b.assign_surface_DL(1.00)

b.active_placement = 'centroid'
b.set_active_section("W24X94")
# b.set_active_section("HSS20.000X0.375")
b.set_active_groups(["columns"])

# modeling_type = {'type': 'elastic'}
modeling_type = {'type': 'fiber', 'n_x': 50, 'n_y': 50}

b.add_columns_from_grids(
    n_sub=10, model_as=modeling_type, geomTransf='Corotational')

b.active_placement = 'top_center'
b.set_active_groups(["beams"])

# b.add_beams_from_grids(n_sub=5, ends={'type': 'pinned', 'dist': 0.001},
#                        model_as=modeling_type)

b.add_beams_from_gridlines(n_sub=10, model_as=modeling_type)


# for more control (offsets etc.), define elements one by one.
#     (see example_offsets.py)
# Primary points can be obtained by intersecting gridlines:
# pt = g1.intersect(gA)

b.select_perimeter_beams_all()
b.selection.add_UDL(np.array((0.00, 0.00, -20.00)))

# b.plot_building_geometry(extrude_frames=False)
# b.plot_building_geometry()

b.preprocess(assume_floor_slabs=False, self_weight=True)

# b.plot_building_geometry(extrude_frames=False,
#                          offsets=True,
#                          gridlines=True,
#                          global_axes=False,
#                          diaphragm_lines=True,
#                          tributary_areas=True,
#                          just_selection=False,
#                          parent_nodes=True,
#                          frame_axes=True)


# b.plot_building_geometry(extrude_frames=True,
#                          offsets=True,
#                          gridlines=True,
#                          global_axes=True,
#                          diaphragm_lines=True,
#                          tributary_areas=True,
#                          just_selection=False,
#                          parent_nodes=True,
#                          frame_axes=True)


# ~~~~~~~~~~~~~~~~~ #
#  linear analysis  #
# ~~~~~~~~~~~~~~~~~ #

for node in b.list_of_parent_nodes():
    node.load += np.array([0.00, 100000.00, 0.00, 0.00, 0.00, 0.00])

# performing a linear gravity analysis.
linear_gravity_analysis = solver.LinearGravityAnalysis(b)
linear_gravity_analysis.run()

# retrieving aggregated textual results
reactions = linear_gravity_analysis.global_reactions(0)
print(reactions[0:3] / 1000)  # kip
print(reactions[3:6] / 1000 / 12)  # kip-ft

# visualizing results
linear_gravity_analysis.deformed_shape(extrude_frames=False)
linear_gravity_analysis.deformed_shape(extrude_frames=True)
linear_gravity_analysis.basic_forces()

# # ~~~~~~~~~~~~~~~~ #
# #  modal analysis  #
# # ~~~~~~~~~~~~~~~~ #

# # performing a linear modal analysis
# modal_analysis = solver.ModalAnalysis(b, num_modes=6)
# modal_analysis.run()

# # retrieving textual results
# print(modal_analysis.periods)

# # visualizing results
# modal_analysis.deformed_shape(step=0, scaling=0.00, extrude_frames=True)
# modal_analysis.deformed_shape(step=1, scaling=0.00, extrude_frames=True)
# modal_analysis.deformed_shape(step=2, scaling=0.00, extrude_frames=True)
# modal_analysis.deformed_shape(step=3, scaling=0.00, extrude_frames=True)
# modal_analysis.deformed_shape(step=4, scaling=0.00, extrude_frames=True)
# modal_analysis.deformed_shape(step=5, scaling=0.00, extrude_frames=True)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#  nonlinear pushover analysis  #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# # performing a nonlinear pushover analysis
# pushover_analysis = solver.PushoverAnalysis(b)
# control_node = b.list_of_all_nodes()[-1]  # top floor
# # control_node = b.list_of_nodes()[-1]  # top floor somewhere
# analysis_metadata = pushover_analysis.run(
#     "y",
#     np.array([48.6]),
#     control_node,
#     1./10.)
# n_plot_steps = analysis_metadata['successful steps']

# # plot the deformed shape for any of the steps
# plot_metadata = pushover_analysis.deformed_shape(
#     step=n_plot_steps-1, scaling=0.00, extrude_frames=True)
# print(plot_metadata)

# # plot pushover curve
# pushover_analysis.plot_pushover_curve("y", control_node)
# pushover_analysis.basic_forces(step=n_plot_steps-1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#  nonlinear time-history analysis  #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# nlth = solver.NLTHAnalysis(b)

# nlth.plot_ground_motion('examples/groundmotions/1xa.txt', 0.005)

# nlth.run(10.00, 0.01,
#          'examples/groundmotions/1xa.txt',
#          'examples/groundmotions/1xa.txt',
#          None, 0.005)

# node = b.list_of_parent_nodes()[-1]  # top floor
# nlth.plot_node_displacement_history(node, 1)

# nlth.deformed_shape(53, scaling=0.00, extrude_frames=True)
# nlth.global_reactions(0)[3] / 1000
