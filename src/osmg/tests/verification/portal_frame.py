"""
Single-bay single-story portal frame model.

Length units: in
Force units: lb
"""

import numpy as np

from osmg.core.model import Model2D
from osmg.creators.component import BeamColumnCreator
from osmg.creators.section import AISC_Database_Section_Creator
from osmg.elements.node import Node
from osmg.elements.section import ElasticSection
from osmg.graphics.plotly import Figure3D, Figure3DConfiguration

# Instantiate model object
frame = Model2D('Frame model')

# Add grid lines
grids = frame.grid_system
grids.add_level('Base', 0.00)
grids.add_level('Level 1', 15.00 * 12.00)
grids.add_grid('A', 0.00)
grids.add_grid('B', 25.00 * 12.00)

# Add primary nodes
for position in (('A', 'Base'), ('B', 'Base'), ('A', 'Level 1'), ('B', 'Level 1')):
    frame.nodes.add(
        Node(
            uid_generator=frame.uid_generator,
            coordinates=(
                grids.get_grid_location(position[0]),
                grids.get_level_elevation(position[1]),
            ),
        ),
    )

# Find the node at 'A'-'Base'
found_node = frame.nodes.search_by_coordinates_or_raise(
    (
        grids.get_grid_location('A'),
        grids.get_level_elevation('Base'),
    )
)

# Define a common section
simple_section = ElasticSection(
    frame.uid_generator,
    'Test Section',
    e_mod=1e3,
    area=1e3,
    i_y=1.00,
    i_x=1.00,
    g_mod=1.00,
    j_mod=1.00,
    sec_w=0.00,
)
# Define an AISC W section
section_creator = AISC_Database_Section_Creator(frame.uid_generator)
standard_section = section_creator.load_elastic_section('W14X38', 1.00, 1.00)
# Add members
bcg = BeamColumnCreator(frame, 'elastic')
for placement_data in (
    ('A', 'Level 1', 'A', 'Base', simple_section),
    ('B', 'Level 1', 'B', 'Base', simple_section),
    ('A', 'Level 1', 'B', 'Level 1', standard_section),
):
    bcg.generate_plain_component_assembly(
        tags={'column'},
        node_i=frame.nodes.search_by_coordinates_or_raise(
            (
                grids.get_grid_location(placement_data[0]),
                grids.get_level_elevation(placement_data[1]),
            )
        ),
        node_j=frame.nodes.search_by_coordinates_or_raise(
            (
                grids.get_grid_location(placement_data[2]),
                grids.get_level_elevation(placement_data[3]),
            )
        ),
        n_sub=3,
        eo_i=np.array((0.00, 0.0)),
        eo_j=np.array((0.00, 0.0)),
        section=placement_data[4],
        transf_type='Linear',
    )


# Load case registry -> stores load cases.
# Will have separate place for dead, live, seismic, other.
# We'll have "analysis types": static, response spectrum, transient.
# Each load case will run the analysis and store the results.
# Each load case will need to have configuration on what results to keep track of.
# It should define supports. Use support configuration objects, which will also support elastic supports and be dimension-agnostic.
# It should still be possible to define a very simple load case and run an analysis manually
# load case registry -> should be able to run all analyses with one method.
# and then be able to get basic forces and node displacements considering case combinations.
#   Add another load case and a combination.
#   Write convenience code to retrieve basic force data for
#   **assembiles**, including combinations.

# Add a load case, Add supports, update plotting.
# Run a linear elastic analysis.
# Plot results.

# Plot combined basic forces.

# Fix "hinged" component assembly.
# Figure out a better way to manage "recorders" and store results.
# Add a quick nonlinear validation analysis.

# Improve design code.

fig = Figure3D(Figure3DConfiguration(num_space_dimensions=2))
fig.add_nodes(list(frame.nodes.values()), 'primary')
fig.add_components(list(frame.components.values()))
fig.show()
