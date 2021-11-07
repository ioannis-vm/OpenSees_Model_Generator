import sys
sys.path.append("../OpenSeesPy_Building_Modeler")

import numpy as np
import modeler
import solver
import time
import matplotlib.pyplot as plt

# Define a building
b = modeler.Building()

# Add levels
b.add_level("base", 0.00, "fixed")
b.add_level("1", 13.00 * 12.00)


# define materials
b.set_active_material('steel-bilinear-fy50')

# define sections

b.add_sections_from_json(
    "../OpenSeesPy_Building_Modeler/section_data/sections.json",
    'W',
    ['W14X120'])

b.set_active_levels(['1'])
b.set_active_section('W14X120')

#
# define structural members
#

elastic_modeling_type = {'type': 'elastic'}
fiber_modeling_type = {'type': 'fiber', 'n_x': 10, 'n_y': 25}

pinned_ends = {'type': 'pinned', 'dist': 0.001}
# pinned_ends = {'type': 'fixed'}  # (debug)

gtransf = 'Corotational'

nsub = 1  # element subdivision

b.set_active_angle(np.pi/2.00)
col = b.add_column_at_point(
    0.00, 0.00, n_sub=nsub,
    model_as=fiber_modeling_type, geomTransf=gtransf)


b.preprocess(assume_floor_slabs=False, self_weight=True)


b.plot_building_geometry(extrude_frames=True,
                         offsets=True,
                         gridlines=True,
                         global_axes=False,
                         diaphragm_lines=True,
                         tributary_areas=True,
                         just_selection=False,
                         parent_nodes=True,
                         frame_axes=False)

node = col[0].node_i

node.load += np.array([0.00, 0.00, -635.00*1e3, 0.00, 0.00, 0.00])

linear_gravity_analysis = solver.LinearGravityAnalysis(b)
linear_gravity_analysis.run()

u1_el = linear_gravity_analysis.node_displacements[
    node.uniq_id][0]
print(u1_el)

# analytic solution
# ii = col[0].section.properties['Ix']
aa = col[0].section.mesh.geometric_properties()['area']
ee = col[0].section.material.parameters['E0']
ll = col[0].length_clear
u_analytical = 635.00*1e3 * ll / (ee*aa)
print(u_analytical)

linear_gravity_analysis.deformed_shape(extrude_frames=True)


# performing a nonlinear pushover analysis
pushover_analysis = solver.PushoverAnalysis(b)
control_node = node
# control_node = b.list_of_nodes()[-1]  # top floor somewhere
analysis_metadata = pushover_analysis.run(
    "x",
    np.array([10.]),
    control_node,
    1./10.)
n_plot_steps = analysis_metadata['successful steps']

# plot the deformed shape for any of the steps
plot_metadata = pushover_analysis.deformed_shape(
    step=n_plot_steps-1, scaling=0.00, extrude_frames=True)
print(plot_metadata)

# plot pushover curve
deltas, vbs = pushover_analysis.table_pushover_curve('x', control_node)

plt.figure()
plt.grid()
plt.plot(deltas, vbs/1e3, 'k')
plt.show()



