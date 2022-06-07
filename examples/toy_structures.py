"""
Toy structures
Predefined models that are used to demonstrate things in the example
notebooks.  They are used to avoid littering the example notebooks
with code that just defines a model, so that they can focus on the
important things.
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

import sys
sys.path.append('../src')
import model
import numpy as np


# Simple 3-story 4-bay moment frame
smrf_3_4 = model.Model()
smrf_3_4.add_level('base', 0.00, 'fixed')
smrf_3_4.add_level('1', 144.00)
smrf_3_4.add_level('2', 144.00*2)
smrf_3_4.add_level('3', 144.00*3)

smrf_3_4.set_active_material('steel-bilinear-fy50')
smrf_3_4.add_sections_from_json(
    "../section_data/sections.json",
    'W',
    ["W24X94"])
smrf_3_4.set_active_section("W24X94")
smrf_3_4.set_active_levels("all_above_base")
smrf_3_4.active_placement = 'centroid'
smrf_3_4.set_active_section("W24X94")
p1 = np.array((0.00, 0.00))
p2 = np.array((360., 0.00))
p3 = np.array((360.*2., 0.00))
p4 = np.array((360.*3, 0.00))
smrf_3_4.active_angle = np.pi/2
for pt in [p1, p2, p3, p4]:
    smrf_3_4.add_column_at_point(
        pt,
        n_sub=4,
        geom_transf='Corotational',
        ends={'type': 'fixed', 'doubler plate thickness': 0.00})
smrf_3_4.active_placement = 'top_center'

smrf_3_4.active_angle = 0.00
for pair in ((p1, p2), (p2, p3), (p3, p4)):
    smrf_3_4.add_beam_at_points(
        pair[0], pair[1],
        n_sub=4,
        snap_i='bottom_center',
        snap_j='top_center')
