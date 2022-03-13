import model
import numpy as np

# Note: units are lb, in

# Define a building
b = model.Model()

# Add levels - single-story building
b.add_level("base", 0.00, "fixed")
b.add_level("1", 144.00)

# define materials

# Note: Concrete would be more suitable for this model.
# Concrete hasn't been implemented yet.
b.set_active_material('steel02-fy50')


# <temporary>

# Note: Normally sections are not supposed to be defined
# here. They should be precompiled in json files and imported
# from those. This is temporary, to illustrate offsets.

# define sections
b.sections.generate_rect(
    '30x30',
    b.materials.active,
    {
        'h': 30.00,
        'b': 30.00
    })

b.sections.generate_rect(
    '60x20',
    b.materials.active,
    {
        'h': 60.00,
        'b': 20.00
    })

b.sections.generate_rect(
    '60x40',
    b.materials.active,
    {
        'h': 60.00,
        'b': 40.00
    })

b.sections.generate_rect(
    '160x40',
    b.materials.active,
    {
        'h': 160.00,
        'b': 40.00
    })

b.sections.generate_rect(
    '30x20',
    b.materials.active,
    {
        'h': 30.00,
        'b': 20.00
    })

# </temporary>

b.set_active_levels('all_above_base')

# floor self-weight
b.assign_surface_DL(1.20)  # lb/in2


# Modeling procedure:
# - [ ] Set active {section, placement, angle}
# - [ ] Set active levels
# - [ ] Define elements
# - [ ] Repeat
# - [ ] In the end, preprocess building

b.set_active_section('160x40')
b.active_placement = 'bottom_left'
b.add_column_at_point(0.00, 0.00)

b.set_active_section('60x40')
b.active_placement = 'bottom_right'
b.add_column_at_point(360.00, 0.00)

b.set_active_section('60x20')
b.active_placement = 'top_right'
b.add_column_at_point(360.00, 360.00)

b.set_active_section('30x30')
b.active_placement = 'top_left'
b.add_column_at_point(0.00, 360.00)

b.set_active_section('30x20')
b.active_placement = 'top_right'

bm = b.add_beam_at_points(np.array((0.00, 0.00)), np.array((360.00, 0.00)),
                          snap_i="bottom_right",
                          snap_j="bottom_left")

b.add_beam_at_points(np.array((360.00, 0.00)), np.array((360.00, 160.00)),
                     snap_i="top_right")
b.add_beam_at_points(np.array((360.00, 160.00)), np.array((360.00, 360.00)),
                     snap_j="bottom_right")
b.add_beam_at_points(np.array((360.00, 360.00)), np.array((0.00, 360.00)),
                     snap_i="top_left",
                     snap_j="top_right")
b.add_beam_at_points(np.array((0.00, 360.00)), np.array((0.00, 0.00)),
                     snap_i="bottom_left",
                     snap_j="top_left")
b.add_beam_at_points(np.array((360.00, 160.00)), np.array((0.00, 0.00)),
                     offset_i=np.array((-20.00, 0.00, 0.00)),
                     snap_j="top_right")

b.preprocess(assume_floor_slabs=True, self_weight=True)


b.plot_building_geometry(extrude_frames=True)
b.plot_building_geometry(extrude_frames=False)
