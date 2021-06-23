"""
This code is used to test out the building modeler
"""

from modeler import Building
import solver
import numpy as np

# Define a building
b = Building()

# Add levels - single-story building
b.add_level("base", 0.00, "fixed")
# b.add_level('1', 120.00)

for i in range(1, 5):
    b.add_level(str(i), 120*i)


# add girdlines
# b.add_gridlines_from_dxf("temp/dxf/dwg_grids.dxf")
b.add_gridlines_from_dxf("temp/dxf/gridlines.dxf")


# Add goups
b.add_group('cols')
b.add_group('bms')

# define materials
b.materials.enable_Steel02()
b.set_active_material('steel')

# define sections
b.add_sections_from_json(
    "section_data/sections.json",
    "W",
    ["W24X94", "W14X90"])

# activate all levels
b.set_active_levels("all_above_base")

# assign surface loads to the active levels
b.assign_surface_DL(1.00)

# define columns
b.set_active_groups(['cols'])
b.set_active_section("W24X94")
b.add_columns_from_grids()

# define beams
b.set_active_groups(['bms'])
b.sections.set_active("W14X90")
b.active_placement = 'top_center'
b.add_beams_from_grids()

b.preprocess()

b.plot_building_geometry(extrude_frames=True)
