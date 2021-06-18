import modeler
import solver
import numpy as np

# Define a building
b = modeler.Building()

# Add levels - single-story building
b.add_level("base", 0.00, "fixed")
b.add_level("1", 144.00)

# add girdlines
b.add_gridline("1", [0., 0.], [360.00, 0.])
b.add_gridline("2", [0., 288.00], [360.00, 288.00])
b.add_gridline("A", [0., 0.], [0., 288.00])
b.add_gridline("B", [360.00, 0.], [360.00, 288.00])

# define materials
b.materials.enable_Steel02()
b.set_active_material('steel')

# define sections
b.add_sections_from_json(
    "section_data/sections.json",
    'W',
    ["W24X94"])
b.set_active_section("W24X94")

b.set_active_levels("all_above_base")

b.add_columns_from_grids(n_sub=1)
# b.active_placement = 'top_center'
b.add_beams_from_grids(n_sub=1, connection='fix_at_column_edge')

b.preprocess(assume_floor_slabs=False)

# b.plot_building_geometry(extrude_frames=False)
# b.plot_building_geometry(extrude_frames=True)

# ~~~~~~~~~~~~~~~~~ #
#  linear analysis  #
# ~~~~~~~~~~~~~~~~~ #

# for node in b.list_of_master_nodes():
#     node.load += modeler.Load([0.00, 100000.00, 0.00, 0.00, 0.00, 0.00])

# node = b.list_of_primary_nodes()[-1]
# node.load += modeler.Load([1000., 100000.00, 0.00, 0.00, 0.00, 0.00])


# performing a linear gravity analysis.
linear_gravity_analysis = solver.LinearGravityAnalysis(b)
linear_gravity_analysis.run()

# retrieving aggregated textual results
# reactions = linear_gravity_analysis.global_reactions(0)
# print(reactions[0:3] / 1000)  # kip
# print(reactions[3:6] / 1000 / 12)  # kip-ft

# # visualizing results
linear_gravity_analysis.deformed_shape(extrude_frames=False)
linear_gravity_analysis.deformed_shape(extrude_frames=True)
linear_gravity_analysis.basic_forces()
