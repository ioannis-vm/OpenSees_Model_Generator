"""
This code is used to test out the building modeler
"""

import modeler
import solver
import numpy as np

# Define a building
b = modeler.Building()

# Add levels - single-story building
b.add_level("base", 0.00, "fixed")
# b.add_level('1', 120.00)

for i in range(1, 7):
    b.add_level(str(i), 120*i)


# add girdlines
b.add_gridlines_from_dxf("examples/dxf/gridlines.dxf")


# Add goups
b.add_group('cols')
b.add_group('bms')
b.add_group('braces')

# define materials
b.materials.enable_Steel02()
b.set_active_material('steel')

# define sections
b.add_sections_from_json(
    "section_data/sections.json",
    "W",
    ["W14X90"])

b.add_sections_from_json(
    "section_data/sections.json",
    "HSS",
    ["HSS22X22X3/4", "HSS18X18X3/4", "HSS14X14X3/4",
     "HSS9X9X5/16"])

b.set_active_levels("all_above_base")
b.assign_surface_DL(1.00)
b.set_active_levels(["6"])
b.assign_surface_DL(1.20)

# define columns
b.set_active_groups(['cols'])
b.set_active_levels([str(i) for i in range(1, 2)])
b.set_active_section("HSS22X22X3/4")
b.add_columns_from_grids()
b.set_active_levels([str(i) for i in range(2, 5)])
b.set_active_section("HSS18X18X3/4")
b.add_columns_from_grids()
b.set_active_levels([str(i) for i in range(5, 7)])
b.set_active_section("HSS14X14X3/4")
b.add_columns_from_grids()

# define beams
b.set_active_groups(['bms'])
b.set_active_levels("all_above_base")
b.set_active_section("W14X90")
b.set_active_placement('top_center')
b.add_beams_from_grids()

# define brace elements
b.clear_gridlines()
b.add_gridlines_from_dxf("examples/dxf/gridlines_brace.dxf")
b.set_active_section("HSS9X9X5/16")
b.set_active_groups(['braces'])
b.add_braces_from_grids(btype="single", n_sub=6, camber=0.01)

b.preprocess()

# b.plot_building_geometry(extrude_frames=True)


# ~~~~~~~~~~~~~~~~~ #
#  linear analysis  #
# ~~~~~~~~~~~~~~~~~ #

# for node in b.list_of_parent_nodes():
#     node.load += np.array([0.00, 100000.00, 0.00, 0.00, 0.00, 0.00])

# # performing a linear gravity analysis.
# linear_gravity_analysis = solver.LinearGravityAnalysis(b)
# linear_gravity_analysis.run()

# retrieving aggregated textual results
# reactions = linear_gravity_analysis.global_reactions(0)
# print(reactions[0:3] / 1000)  # kip
# print(reactions[3:6] / 1000 / 12)  # kip-ft

# visualizing results
# linear_gravity_analysis.deformed_shape(extrude_frames=False)
# linear_gravity_analysis.deformed_shape(extrude_frames=True)
# linear_gravity_analysis.basic_forces()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#  nonlinear pushover analysis  #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# performing a nonlinear pushover analysis
pushover_analysis = solver.PushoverAnalysis(b)
control_node = b.list_of_parent_nodes()[-1]  # top floor
# control_node = b.list_of_nodes()[-1]  # top floor somewhere
analysis_metadata = pushover_analysis.run(
    "x",
    40.,
    control_node,
    0.10,
    np.linspace(0., 40., 100), n_x=4, n_y=8, n_p=3)
n_plot_steps = analysis_metadata['successful steps']

# plot the deformed shape for any of the steps
plot_metadata = pushover_analysis.deformed_shape(
    step=n_plot_steps-1, scaling=0.00, extrude_frames=True)
print(plot_metadata)

# plot pushover curve
pushover_analysis.plot_pushover_curve("x", control_node)
