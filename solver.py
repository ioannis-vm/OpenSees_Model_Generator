import openseespy.opensees as ops
import numpy as np
from modeler import Building

EPSILON = 1.00E-6


def ops_initialize():
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)


def ops_define_materials(building):
    for material in building.materials.material_list:
        if material.ops_material == 'Steel02':
            ops.uniaxialMaterial('Steel02',
                                 material.uniq_id,
                                 material.parameters['Fy'],
                                 material.parameters['E0'],
                                 material.parameters['b'])


def ops_define_elastic_sections(building):
    for sec in building.sections.section_list:
        ops.section('Elastic',
                    sec.uniq_id,
                    sec.material.parameters['E0'],
                    sec.properties['A'],
                    sec.properties['Ix'],
                    sec.properties['Iy'],
                    sec.material.parameters['G'],
                    sec.properties['J'])
        ops.beamIntegration(
            'Lobatto',
            sec.uniq_id,
            sec.uniq_id,
            10)


def ops_define_nodes(building):
    for lvl in building.levels.level_list:
        for node in lvl.nodes.node_list:
            ops.node(node.uniq_id,
                     node.coordinates[0],
                     node.coordinates[1],
                     node.coordinates[2])


def ops_define_node_restraints(building):
    for lvl in building.levels.level_list:
        for node in lvl.nodes.node_list:
            if node.restraint_type == 'fixed':
                ops.fix(node.uniq_id, 1, 1, 1, 1, 1, 1)
            elif node.restraint_type == 'pinned':
                ops.fix(node.uniq_id, 1, 1, 1, 0, 0, 0)


def ops_define_node_mass(building):
    for lvl in building.levels.level_list:
        for node in lvl.nodes.node_list:
            if node.mass:
                if max(node.mass.value) > EPSILON:
                    ops.mass(node.uniq_id,
                             node.mass.value[0],
                             node.mass.value[1],
                             node.mass.value[2])


def ops_define_beamcolumn_elements_linear(building):
    for lvl in building.levels.level_list:
        for elm in lvl.columns.column_list+lvl.beams.beam_list:
            # geometric transformation
            ops.geomTransf('Linear',
                           elm.uniq_id,
                           *elm.local_z_axis_vector())
            ops.element('dispBeamColumn',
                        elm.uniq_id,
                        elm.node_i.uniq_id,
                        elm.node_j.uniq_id,
                        elm.uniq_id,
                        elm.section.uniq_id)


def ops_define_beamcolumn_elements_fiber(building, n_p, n_x, n_y):
    # first define sections
    for sec in building.sections.section_list:
        pieces = sec.subdivide_section(n_x=n_x, n_y=n_y)
        ops.section('Fiber',
                    sec.uniq_id,
                    '-GJ',
                    sec.properties['J']*sec.material.parameters['G'])
        for piece in pieces:
            area = piece.area
            z_loc = piece.centroid.x
            y_loc = piece.centroid.y
            ops.fiber(y_loc,
                      z_loc,
                      area,
                      sec.material.uniq_id)
        ops.beamIntegration('Lobatto', sec.uniq_id, sec.uniq_id, n_p)
    # define elements
    for lvl in building.levels.level_list:
        for elm in lvl.columns.column_list+lvl.beams.beam_list:
            # geometric transformation
            ops.geomTransf('Linear',
                           elm.uniq_id,
                           *elm.local_z_axis_vector())
            ops.element('dispBeamColumn',
                        elm.uniq_id,
                        elm.node_i.uniq_id,
                        elm.node_j.uniq_id,
                        elm.uniq_id,
                        elm.section.uniq_id)


def ops_define_dead_load(building):
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for lvl in building.levels.level_list:
        for elm in lvl.columns.column_list+lvl.beams.beam_list:
            ops.eleLoad('-ele', elm.uniq_id,
                        '-type', '-beamUniform',
                        elm.udl.value[1],
                        elm.udl.value[2],
                        elm.udl.value[0])


def to_OpenSees_domain(building: Building, frame_elem_type='fiber'):
    """
    Defines the building model in OpenSeesPy
    """
    ops_initialize()
    ops_define_materials(building)
    ops_define_nodes(building)
    ops_define_node_restraints(building)
    ops_define_node_mass(building)
    if frame_elem_type == 'linear':
        ops_define_elastic_sections(building)
        ops_define_beamcolumn_elements_linear(building)
    elif frame_elem_type == 'fiber':
        ops_define_beamcolumn_elements_fiber(
            building, n_p=10, n_x=10, n_y=25)
    else:
        raise ValueError(
            "Frame element type can either be `linear` or `fiber`")


def ops_run_gravity_analysis():
    ops.system('BandGeneral')
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.test('NormDispIncr', 1.0e-12, 10, 3)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 0.1)
    ops.analysis('Static')
    ops.analyze(10)


def modal_analysis(building: Building, n_modes=1):
    """
    Runs a modal analysis assuming the building has
    been defined in the OpenSees domain.
    """
    eigValues = np.array(ops.eigen(n_modes))
    periods = np.sqrt(2.00*np.pi / np.sqrt(eigValues))
    return periods


def gravity_analysis(building: Building):
    """
    Runs gravity analysis assuming the building has
    been defined in the OpenSees domain.
    """
    ops_define_dead_load(building)
    ops_run_gravity_analysis()
