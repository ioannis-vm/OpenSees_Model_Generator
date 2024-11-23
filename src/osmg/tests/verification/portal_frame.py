"""
Single-bay single-story portal frame model.

Length units: in
Force units: lb
"""

import numpy as np

from osmg.core.model import Model2D
from osmg.creators.component import BeamColumnCreator
from osmg.elements.node import Node
from osmg.elements.section import ElasticSection
from osmg.graphics.plotly import Figure3DConfiguration, Figure3D

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
found_node = frame.nodes.search_by_coordinates(
    (
        grids.get_grid_location('A'),
        grids.get_level_elevation('Base'),
    )
)

# Define a common section
section = ElasticSection(
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
# Add members
bcg = BeamColumnCreator(frame, 'elastic')
for placement in (
    ('A', 'Level 1', 'A', 'Base'),
    ('B', 'Level 1', 'B', 'Base'),
    ('A', 'Level 1', 'B', 'Level 1'),
):
    bcg.generate_plain_component_assembly(
        tags={'column'},
        node_i=frame.nodes.search_by_coordinates(
            (
                grids.get_grid_location(placement[0]),
                grids.get_level_elevation(placement[1]),
            )
        ),
        node_j=frame.nodes.search_by_coordinates(
            (
                grids.get_grid_location(placement[2]),
                grids.get_level_elevation(placement[3]),
            )
        ),
        n_sub=3,
        eo_i=np.array((0.00, 0.0)),
        eo_j=np.array((0.00, 0.0)),
        section=section,
        transf_type='Elastic',
    )


# Remove all code that is currently not used, lint and unit test what
# we have now.
#   We'll use git to recover the old code and add it back as needed.

# Add a load case, Add supports, update plotting.
# Run a linear elastic analysis.
# Plot results.

# Add another load case and a combination.
# Write convenience code to retrieve basic force data for
# **assembiles**, including combinations.

# Plot combined basic forces.

# Fix "hinged" component assembly.
# Figure out a better way to manage "recorders" and store results.
# Add a quick nonlinear validation analysis.

# Improve design code.

fig = Figure3D(Figure3DConfiguration(num_space_dimensions=2))
fig.add_nodes(frame.nodes.values(), 'primary')
fig.add_components(frame.components.values())
fig.show()
