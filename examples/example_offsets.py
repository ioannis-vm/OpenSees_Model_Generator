import modeler
import numpy as np

# Define a building
b = modeler.Building()

# Add levels - single-story building
b.add_level("base", 0.00, "fixed")
b.add_level("1", 144.00)
b.add_level("2", 144.00*2)
b.add_level("3", 144.00*3)


# define materials
b.materials.enable_Steel02()
b.set_active_material('steel')

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
    '120x40',
    b.materials.active,
    {
        'h': 120.00,
        'b': 40.00
    })

b.sections.generate_rect(
    '30x20',
    b.materials.active,
    {
        'h': 30.00,
        'b': 20.00
    })


b.set_active_levels('all_above_base')

b.set_active_section('120x40')
b.active_placement = 'bottom_left'
b.add_column_at_point(0.00, 0.00)

b.set_active_section('60x40')
b.active_placement = 'bottom_right'
b.add_column_at_point(240.00, 0.00)

b.set_active_section('60x20')
b.active_placement = 'top_right'
b.add_column_at_point(240.00, 240.00)

b.set_active_section('30x30')
b.active_placement = 'top_left'
b.add_column_at_point(0.00, 240.00)

b.set_active_section('30x20')
b.active_placement = 'top_right'
b.add_beam_at_points(np.array((0.00, 0.00)), np.array((240.00, 0.00)),
                     offset_i=np.array((40.0, 0.00, 0.00)),
                     offset_j=np.array((-40.00, 0.00, 0.00)))
b.add_beam_at_points(np.array((240.00, 0.00)), np.array((240.00, 120.00)),
                     offset_i=np.array((0.00, 60.00, 0.00)),
                     offset_j=np.array((0.00, 0.00, 0.00)))
b.add_beam_at_points(np.array((240.00, 120.00)), np.array((240.00, 240.00)),
                     offset_i=np.array((0.00, 0.00, 0.00)),
                     offset_j=np.array((0.00, -60.00, 0.00)))
b.add_beam_at_points(np.array((240.00, 240.00)), np.array((0.00, 240.00)),
                     offset_i=np.array((-20.00, 0.00, 0.00)),
                     offset_j=np.array((30.00, 0.00, 0.00)))
b.add_beam_at_points(np.array((0.00, 240.00)), np.array((0.00, 0.00)),
                     offset_i=np.array((0.00, -30.00, 0.00)),
                     offset_j=np.array((0.00, 120.00, 0.00)))
b.add_beam_at_points(np.array((240.00, 120.00)), np.array((0.00, 0.00)),
                     offset_i=np.array((-20.00, 0.00, 0.00)),
                     offset_j=np.array((40.00, 120.00, 0.00)))

b.preprocess(assume_floor_slabs=True, self_weight=True)


b.plot_2D_level_geometry("1", extrude_frames=True)

# b.plot_building_geometry(extrude_frames=True)
# b.plot_building_geometry(extrude_frames=False)
