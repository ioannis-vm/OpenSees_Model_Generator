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

b.plot_building_geometry(extrude_frames=True)
