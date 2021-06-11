"""
Building Modeler for OpenSeesPy ~ Solver module
"""

#   __                 UC Berkeley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   April 2021
#
# https://github.com/ioannis-vm/OpenSeesPy_Building_Modeler

from typing import List, TypedDict
from dataclasses import dataclass, field
import openseespy.opensees as ops
import numpy as np
from modeler import Building, Node
from utility.graphics import postprocessing_3D

EPSILON = 1.00E-6


class AnalysisResult(TypedDict):
    uniq_id: int
    results = List


def store_result(analysis_result: AnalysisResult, uniq_id: int, result: List):
    if uniq_id in analysis_result.keys():
        analysis_result[uniq_id].append(result)
    else:
        analysis_result[uniq_id] = [result]


@dataclass
class Analysis:
    building: Building
    node_displacements: AnalysisResult = field(
        default_factory=AnalysisResult)
    node_reactions: AnalysisResult = field(default_factory=AnalysisResult)

    #############################################
    # Methods that send information to OpenSees #
    #############################################

    def ops_initialize(self):
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

    def ops_define_materials(self):
        for material in self.building.materials.material_list:
            if material.ops_material == 'Steel02':
                ops.uniaxialMaterial('Steel02',
                                     material.uniq_id,
                                     material.parameters['Fy'],
                                     material.parameters['E0'],
                                     material.parameters['b'])
            else:
                raise ValueError("Unsupported material")

    def ops_define_nodes(self):
        for node in self.building.list_of_nodes() + \
                self.building.list_of_master_nodes():
            ops.node(node.uniq_id,
                     *node.coordinates)

    def ops_define_node_restraints(self):
        for node in self.building.list_of_nodes():
            if node.restraint_type == 'fixed':
                ops.fix(node.uniq_id, 1, 1, 1, 1, 1, 1)
            elif node.restraint_type == 'pinned':
                ops.fix(node.uniq_id, 1, 1, 1, 0, 0, 0)
        for node in self.building.list_of_master_nodes():
            ops.fix(node.uniq_id, 0, 0, 1, 1, 1, 0)

    def ops_define_node_constraints(self):
        for lvl in self.building.levels.level_list:
            if lvl.master_node:
                ops.rigidDiaphragm(
                    3,
                    lvl.master_node.uniq_id,
                    *[node.uniq_id
                      for node in lvl.nodes.node_list])

    def ops_define_node_mass(self):
        for node in self.building.list_of_nodes() + \
                self.building.list_of_master_nodes():
            if node.mass:
                if max(node.mass.value) > EPSILON:
                    ops.mass(node.uniq_id,
                             *node.mass.value)

    def ops_define_dead_load(self):
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)

        for elm in self.building.list_of_frames():
            ops.eleLoad('-ele', elm.uniq_id,
                        '-type', '-beamUniform',
                        elm.udl.value[1],
                        elm.udl.value[2],
                        elm.udl.value[0])
        for node in self.building.list_of_nodes() + \
                self.building.list_of_master_nodes():
            ops.load(node.uniq_id, *node.load.value)

    def ops_define_sections(self):
        pass

    def ops_define_beamcolumn_elements(self):
        pass

    def to_OpenSees_domain(self):
        """
        Defines the building model in OpenSeesPy
        """
        self.ops_initialize()
        self.ops_define_materials()
        self.ops_define_nodes()
        self.ops_define_node_restraints()
        self.ops_define_node_constraints()
        self.ops_define_node_mass()
        # the following two columns use methods that
        # are defined in the inherited classes that follow
        self.ops_define_sections()
        self.ops_define_beamcolumn_elements()
        self.ops_define_dead_load()

    ####################################################
    # Methods that read back information from OpenSees #
    ####################################################

    def read_node_displacements(self):
        for node in self.building.list_of_nodes():
            store_result(self.node_displacements,
                         node.uniq_id,
                         ops.nodeDisp(node.uniq_id))
        for node in self.building.list_of_master_nodes():
            store_result(self.node_displacements,
                         node.uniq_id,
                         ops.nodeDisp(node.uniq_id))

    def read_node_reactions(self):
        ops.reactions()
        for lvl in self.building.levels.level_list:
            for node in lvl.nodes.node_list:
                if node.restraint_type != 'free':
                    uid = node.uniq_id
                    local_reaction = np.array(ops.nodeReaction(uid))
                    store_result(self.node_reactions,
                                 uid,
                                 local_reaction)

    def read_OpenSees_results(self):
        """
        Reads back the results from the OpenSeesPy domain
        """
        self.read_node_displacements()
        self.read_node_reactions()

    #########################
    # Visualization methods #
    #########################
    def deformed_shape(self, step=0, scaling=0.00, extrude_frames=False):
        postprocessing_3D.deformed_shape(self,
                                         step,
                                         scaling,
                                         extrude_frames)
    ##################################
    # Numeric Result Post-processing #
    ##################################

    def global_reactions(self):
        reactions = np.full(6, 0.00)
        for lvl in self.building.levels.level_list:
            for node in lvl.nodes.node_list:
                if node.restraint_type != 'free':
                    uid = node.uniq_id
                    x = node.coordinates[0]
                    y = node.coordinates[1]
                    z = node.coordinates[2]
                    local_reaction = self.node_reactions[uid][0]
                    global_reaction = np.array([
                        local_reaction[0],
                        local_reaction[1],
                        local_reaction[2],
                        local_reaction[3] + local_reaction[2] * y
                        - local_reaction[1] * z,
                        local_reaction[4] + local_reaction[0] * z
                        - local_reaction[2] * x,
                        local_reaction[5] + local_reaction[1] * x
                        - local_reaction[0] * y
                    ])
                    reactions += global_reaction
        return reactions


@dataclass
class LinearAnalysis(Analysis):

    def ops_define_sections(self):
        for sec in self.building.sections.section_list:
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

    def ops_define_beamcolumn_elements(self):
        for elm in self.building.list_of_frames():
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

    def ops_run_gravity_analysis(self):
        ops.system('BandGeneral')
        ops.numberer('RCM')
        ops.constraints('Transformation')
        ops.test('NormDispIncr', 1.0e-12, 10, 3)
        ops.algorithm('Newton')
        ops.integrator('LoadControl', 1.0)
        ops.analysis('Static')
        ops.analyze(1)


@dataclass
class ModalAnalysis(LinearAnalysis):
    """
    Runs a modal analysis assuming the building has
    been defined in the OpenSees domain.
    """
    num_modes: int = field(default=1)
    periods: List[float] = field(default_factory=list)

    ####################################################
    # Methods that read back information from OpenSees #
    ####################################################

    def read_node_displacements(self):
        for lvl in self.building.levels.level_list:
            for node in lvl.nodes.node_list:
                for i in range(self.num_modes):
                    store_result(self.node_displacements,
                                 node.uniq_id,
                                 ops.nodeEigenvector(
                                     node.uniq_id,
                                     i+1)
                                 )

    def run(self):
        self.to_OpenSees_domain()
        eigValues = np.array(ops.eigen(
            '-fullGenLapack',
            self.num_modes))
        self.periods = 2.00*np.pi / np.sqrt(eigValues)
        self.read_node_displacements()


@dataclass
class LinearGravityAnalysis(LinearAnalysis):
    def run(self):
        self.to_OpenSees_domain()
        self.ops_define_dead_load()
        self.ops_run_gravity_analysis()
        self.read_OpenSees_results()


@dataclass
class NonlinearAnalysis(Analysis):

    def ops_define_sections(self, n_x, n_y, n_p):
        for sec in self.building.sections.section_list:
            pieces = sec.subdivide_section(
                n_x=n_x, n_y=n_y)
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
            ops.beamIntegration(
                'Lobatto', sec.uniq_id, sec.uniq_id, n_p)

    def ops_define_beamcolumn_elements(self):
        for lvl in self.building.levels.level_list:
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

    def to_OpenSees_domain(self, n_x, n_y, n_p):
        """
        Defines the building model in OpenSeesPy
        """
        self.ops_initialize()
        self.ops_define_materials()
        self.ops_define_nodes()
        self.ops_define_node_restraints()
        self.ops_define_node_constraints()
        self.ops_define_node_mass()
        # the following two columns use methods that
        # are defined in the inherited classes that follow
        self.ops_define_sections(n_x, n_y, n_p)
        self.ops_define_beamcolumn_elements()

    def ops_run_gravity_analysis(self):
        ops.system('BandGeneral')
        ops.numberer('RCM')
        ops.constraints('Transformation')
        ops.test('NormDispIncr', 1.0e-12, 10, 3)
        ops.algorithm('Newton')
        ops.integrator('LoadControl', 0.1)
        ops.analysis('Static')
        ops.analyze(10)


@dataclass
class PushoverAnalysis(NonlinearAnalysis):

    def ops_apply_lateral_load(self, direction):
        distribution = self.building.level_masses()
        distribution = distribution / np.linalg.norm(distribution)

        # define the load pattern
        ops.timeSeries('Linear', 2)
        ops.pattern("Plain", 2, 2)

        if direction == 'x':
            load_dir = np.array([1., 0., 0., 0., 0., 0.])
        if direction == 'y':
            load_dir = np.array([0., 1., 0., 0., 0., 0.])

        for i, lvl in enumerate(self.building.levels.level_list):
            # if the level is restrained, no load applied
            if lvl.restraint != 'free':
                continue
            # if there is a master node, all load goes there
            if lvl.master_node:
                ops.load(lvl.master_node.uniq_id,
                         *(distribution[i]*load_dir))
            # if there isn't a master node, distribute that story's load
            # in proportion to the mass of the nodes
            else:
                node_list = lvl.nodes.node_list
                masses = np.array([n.mass.value[0] for n in node_list])
                masses = masses/np.linalg.norm(masses)
                for j, node in enumerate(node_list):
                    ops.load(node.uniq_id,
                             *(distribution[i]*masses[j]*load_dir))

    def run(self, direction, target_displacement,
            control_node, displ_incr, n_x=10, n_y=25, n_p=10):
        if direction == 'x':
            control_DOF = 0
        elif direction == 'y':
            control_DOF = 1
        else:
            raise ValueError("Direction can either be 'x' or 'y'")
        self.to_OpenSees_domain(n_x, n_y, n_p)
        # self.ops_define_dead_load()
        # self.ops_run_gravity_analysis()
        # ops.wipeAnalysis()
        self.ops_apply_lateral_load(direction)
        ops.system("BandGeneral")
        ops.numberer('RCM')
        ops.constraints('Transformation')
        ops.test('NormUnbalance', 1e-6, 1000)
        ops.integrator("DisplacementControl",
                       control_node.uniq_id, control_DOF + 1, displ_incr)
        ops.algorithm("Newton")
        ops.analysis("Static")
        curr_displ = 0.00
        while curr_displ < target_displacement:
            check = ops.analyze(1)
            if check != 0:
                print('Unable to converge')
                break
            # self.read_OpenSees_results()
            curr_displ = ops.nodeDisp(control_node.uniq_id, control_DOF+1)
            # curr_displ = self.node_displacements[
            #     control_node.uniq_id][-1][control_DOF]
            print(curr_displ, end='\r')
        self.read_OpenSees_results()
