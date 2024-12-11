"""
Braced frame design model.

Length units: in
Force units: kip
"""

import numpy as np

from osmg.analysis.common import UDL, PointLoad, PointMass
from osmg.analysis.load_case import LoadCaseRegistry
from osmg.analysis.supports import FixedSupport
from osmg.core.model import Model2D
from osmg.creators.component import BarGenerator, BeamColumnCreator
from osmg.creators.material import ElasticMaterialCreator
from osmg.creators.section import AISC_Database_Section_Creator
from osmg.graphics.plotly import (
    BasicForceConfiguration,
    DeformationConfiguration,
    Figure3D,
    Figure3DConfiguration,
)
from osmg.model_objects.node import Node
from osmg.model_objects.section import ElasticSection

# Instantiate model object
frame = Model2D(name='Gridline B', dimensionality='2D Frame')

# Add levels and grids
grids = frame.grid_system
grids.add_level('0', 0.00)
grids.add_level('1', 15.00 * 12.00)
grids.add_level('2', (15.00 + 13.00) * 12.00)
grids.add_level('3', (15.00 + 2.0 * 13.00) * 12.00)
grids.add_level('4', (15.00 + 3.0 * 13.00) * 12.00)
grids.add_grid('A', 0.00)
grids.add_grid('B', 25.00 * 12.00)
grids.add_grid('C', 25.00 * 2.0 * 12.00)
grids.add_grid('D', 25.00 * 3.0 * 12.00)
grids.add_grid('E', 25.00 * 4.0 * 12.00)
grids.add_grid('F', 25.00 * 5.0 * 12.00)
grids.add_grid('LC', 25.00 * 6.0 * 12.00)

# Add primary nodes
for level in ('0', '1', '2', '3', '4'):
    for grid in ('A', 'B', 'C', 'D', 'E', 'F', 'LC'):
        if (grid == 'A' and level == '4') or (grid == 'F' and level == '4'):
            continue
        frame.nodes.add(
            Node(
                uid_generator=frame.uid_generator,
                coordinates=(
                    grids.get_grid_location(grid),
                    grids.get_level(level).elevation(),
                ),
            ),
        )

# Example: find the node at 'Base'-'A' (not used)
found_node = frame.nodes.search_by_coordinates_or_raise(
    (
        grids.get_grid('A').data(),
        grids.get_level('0').elevation(),
    )
)

# Define a common section (not used)
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

# Define Steel Young's modulus and shear modulus.
e_modulus = 29000.00  # kip/in2
g_modulus = 11500.00  # kip/in2

# Define an AISC W section (length unit is `in`)
section_creator = AISC_Database_Section_Creator(frame.uid_generator)
column_section = section_creator.load_elastic_section(
    section_label='W14X120', e_modulus=e_modulus, g_modulus=g_modulus
)
beam_section = section_creator.load_elastic_section(
    section_label='W18X119', e_modulus=e_modulus, g_modulus=g_modulus
)


# Add columns
bcg = BeamColumnCreator(frame, 'elastic')
added_columns = []
for level in ('1', '2', '3', '4'):
    for grid in ('A', 'B', 'C', 'D', 'E', 'F'):
        if level == '4' and (grid in {'A', 'F'}):
            continue
        col = bcg.generate_plain_component_assembly(
            tags={'Column'},
            node_i=frame.nodes.search_by_coordinates_or_raise(
                (
                    grids.get_grid(grid).data(),
                    grids.get_level(level).elevation(),
                )
            ),
            node_j=frame.nodes.search_by_coordinates_or_raise(
                (
                    grids.get_grid(grid).data(),
                    grids.get_level(level).previous().elevation(),
                )
            ),
            n_sub=1,
            eo_i=np.array((0.00, 0.0)),
            eo_j=np.array((0.00, 0.0)),
            section=column_section,
            transf_type='Linear',
        )
        added_columns.append(col)

# Add beams
added_beams = []
for level in ('1', '2', '3', '4'):
    for grid in ('A', 'B', 'C', 'D', 'E'):
        if level == '4' and grid in {'A', 'E'}:
            continue
        beam = bcg.generate_plain_component_assembly(
            tags={'Beam'},
            node_i=frame.nodes.search_by_coordinates_or_raise(
                (
                    grids.get_grid(grid).data(),
                    grids.get_level(level).elevation(),
                )
            ),
            node_j=frame.nodes.search_by_coordinates_or_raise(
                (
                    grids.get_grid(grid).next().data(),
                    grids.get_level(level).elevation(),
                )
            ),
            n_sub=1,
            eo_i=np.array((0.00, 0.0)),
            eo_j=np.array((0.00, 0.0)),
            section=beam_section,
            transf_type='Linear',
        )
        added_beams.append(beam)

# # Add braces
brg = BarGenerator(frame)
for level_top, level_bottom, grid_top, grid_bottom in zip(
    ('1', '1', '1', '1', '2', '2', '3', '4'),
    ('0', '0', '0', '0', '1', '1', '2', '3'),
    ('C', 'C', 'E', 'E', 'D', 'D', 'C', 'D'),
    ('B', 'D', 'D', 'F', 'C', 'E', 'D', 'C'),
):
    brg.add(
        tags={'Brace'},
        node_i=frame.nodes.search_by_coordinates_or_raise(
            (
                grids.get_grid(grid_top).data(),
                grids.get_level(level_top).elevation(),
            )
        ),
        node_j=frame.nodes.search_by_coordinates_or_raise(
            (
                grids.get_grid(grid_bottom).data(),
                grids.get_level(level_bottom).elevation(),
            )
        ),
        eo_i=np.array((0.00, 0.00)),
        eo_j=np.array((0.00, 0.00)),
        transf_type='Linear',
        area=40.00,
        material=ElasticMaterialCreator(frame, stiffness=e_modulus).generate(),
        outside_shape=None,
        weight_per_length=0.00,
    )


# Add leaning column
leaning_column_area = 1e4  # in2
for level in ('1', '2', '3', '4'):
    brg.add(
        tags={'Truss'},
        node_i=frame.nodes.search_by_coordinates_or_raise(
            (
                grids.get_grid('LC').data(),
                grids.get_level(level).elevation(),
            )
        ),
        node_j=frame.nodes.search_by_coordinates_or_raise(
            (
                grids.get_grid('LC').data(),
                grids.get_level(level).previous().elevation(),
            )
        ),
        eo_i=np.array((0.00, 0.00)),
        eo_j=np.array((0.00, 0.00)),
        transf_type='Linear',
        area=leaning_column_area,
        material=ElasticMaterialCreator(frame, stiffness=e_modulus).generate(),
        outside_shape=None,
        weight_per_length=0.00,
    )


def show_model() -> None:
    """Show the model (only)."""
    fig = Figure3D(Figure3DConfiguration(ndm=2))
    fig.add_nodes(list(frame.nodes.values()), 'primary')
    fig.add_components(list(frame.components.values()))
    fig.show()


# Create a load case registry
load_case_registry = LoadCaseRegistry(frame)
# Define load cases
lc_modal = load_case_registry.modal['modal_1']
lc_dead = load_case_registry.dead['dead_1']
load_cases = (lc_modal, lc_dead)

# Create a load case and add fixed supports
fixed_support = FixedSupport((True, True, True))
for lc in load_cases:
    lc.add_supports_at_level(frame, fixed_support, '0')

# Add rigid diaphragm
for level in ('1', '2', '3', '4'):
    parent_node = frame.nodes.search_by_coordinates_or_raise(
        (
            grids.get_grid('LC').data(),
            grids.get_level(level).elevation(),
        )
    )
    for lc in load_cases:
        lc.define_rigid_diaphragm(frame, parent_node)


# # Example of how to retrieve a primary node:
# # Locate the nodes at 'A'-'Level 1' and 'B'-'Level 1'
# frame.nodes.search_by_coordinates_or_raise(
#     (grids.get_grid('A').data(), grids.get_level(').elevation())
# )

# `lc_dead`: Add UDLs to the beams
for beam in added_beams:
    lc_dead.load_registry.component_udl[beam.uid] = UDL((0.0, -1.67e-3))  # kip/in

# `lc_dead`: Add a concentrated point load at 'B'-'Level 4'
lc_dead.load_registry.nodal_loads[
    frame.nodes.search_by_coordinates_or_raise(
        (
            grids.get_grid('B').data(),
            grids.get_level('4').elevation(),
        )
    ).uid
] = PointLoad(
    (2000.0, 0.00, 0.00)  # lb
)

# `modal`: Add mass on the leaning column.
for level in ('1', '2', '3', '4'):
    lc_modal.mass_registry[
        frame.nodes.search_by_coordinates_or_raise(
            (
                grids.get_grid('LC').data(),
                grids.get_level(level).elevation(),
            )
        ).uid
    ] = PointMass((+16.0e3 / 386.22, 0.00, 0.00))
lc_modal.analysis.settings.num_modes = 2

# # Example: Add an extra recorder
# load_case_registry.dead['dead_1'].analysis.recorders['node_envelope'] = NodeRecorder(
#     uid_generator=frame.uid_generator,
#     recorder_type='EnvelopeNode',
#     nodes=(
#         frame.nodes.search_by_coordinates_or_raise(
#             (
#                 grids.get_grid('A').data(),
#                 grids.get_level('3').elevation(),
#             )
#         ).uid,
#     ),
#     dofs=(1, 2, 3),
#     response_type='disp',
#     file_name='envelope',
#     output_time=True,
#     number_of_significant_digits=6,
# )


# # Example: change num_steps
# load_case_registry.dead['dead_1'].analysis.settings.num_steps = 10

# Run analysis
load_case_registry.run()

# result_dir = load_case_registry.dead['dead_1'].analysis.settings.result_directory
# print(f'Result directory `dead_1`: {result_dir}')
# result_dir = load_case_registry.dead['dead_2'].analysis.settings.result_directory
# print(f'Result directory `dead_2`: {result_dir}')


# lc_modal.analysis.recorders['default_node'].get_data()
# lc_modal.analysis.recorders[
#     'default_basic_force'
# ].get_data()

# combinations happen here.
# combined_displacements = load_case_registry.combine_recorder('default_node')

# forces = (
#     load_case_registry.dead['dead_1']
#     .analysis.recorders['default_basic_force']
#     .get_data()
# )


# data = (
#     load_case_registry.dead['dead_1']
#     .analysis.recorders['default_basic_force']
#     .get_data()
# )


step = 0
deformation_configuration = DeformationConfiguration(
    reference_length=frame.reference_length(),
    ndf=3,
    ndm=2,
    data=lc_dead.analysis.recorders['default_node'].get_data(),
    step=step,
    # amplification_factor=None,  # Figure it out.
    amplification_factor=10.00 / 3.06618,  # Figure it out.
)
basic_force_configuration = BasicForceConfiguration(
    reference_length=frame.reference_length(),
    ndm=2,
    ndf=3,
    data=lc_dead.calculate_basic_forces(
        'default_basic_force',
        frame.components.get_line_element_lengths(),
        ndm=2,
        num_stations=12,
    ),
    step=step,
    force_to_length_factor=12 * 12 / 1883.89,
    moment_to_length_factor=12 * 12 / 10000.0,
)
fig = Figure3D(Figure3DConfiguration(ndm=2))
fig.add_nodes(list(frame.nodes.values()), 'primary', overlay=True)
fig.add_components(list(frame.components.values()), overlay=True)
# fig.add_nodes(list(frame.nodes.values()), 'primary')
# fig.add_components(list(frame.components.values()))
fig.add_nodes(list(frame.nodes.values()), 'primary', deformation_configuration)
fig.add_components(list(frame.components.values()), deformation_configuration)
fig.add_supports(frame.nodes, lc_modal.fixed_supports, symbol_size=12.00)
# fig.add_udl(
#     load_case_registry.dead['dead_2'].load_registry.component_udl,
#     frame.components,
#     force_to_length_factor=2.0,
#     offset=0.00,
# )
# fig.add_loads(
#     lc_modal.mass_registry,
#     frame.nodes,
#     force_to_length_factor=2.00,
#     offset=0.0,
#     head_length=24.0,
#     head_width=24.0,
#     base_width=5.0,
# )
fig.add_basic_forces(
    components=list(frame.components.values()),
    basic_force_configuration=basic_force_configuration,
)
fig.show()
