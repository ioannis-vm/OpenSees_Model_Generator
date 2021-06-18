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
from modeler import Building
from modeler import G_CONST
from utility.graphics import postprocessing_3D
from utility.graphics import general_2D

EPSILON = 1.00E-6


class AnalysisResult(TypedDict):
    uniq_id: int
    results = List


@dataclass
class Analysis:

    building: Building
    node_displacements: AnalysisResult = field(
        default_factory=AnalysisResult)
    node_reactions: AnalysisResult = field(default_factory=AnalysisResult)
    frame_basic_forces: AnalysisResult = field(default_factory=AnalysisResult)

    def _store_result(self, analysis_result: AnalysisResult,
                      uniq_id: int, result: list):
        if uniq_id in analysis_result.keys():
            analysis_result[uniq_id].append(result)
        else:
            analysis_result[uniq_id] = [result]

    #############################################
    # Methods that send information to OpenSees #
    #############################################

    def _initialize(self):
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

    def _define_materials(self):
        for material in self.building.materials.material_list:
            if material.ops_material == 'Steel02':
                ops.uniaxialMaterial('Steel02',
                                     material.uniq_id,
                                     material.parameters['Fy'],
                                     material.parameters['E0'],
                                     material.parameters['b'])
            else:
                raise ValueError("Unsupported material")

    def _define_nodes(self):
        for node in self.building.list_of_primary_nodes() + \
                self.building.list_of_master_nodes() + \
                self.building.list_of_internal_nodes():
            ops.node(node.uniq_id,
                     *node.coordinates)

    def _define_node_restraints(self):
        for node in self.building.list_of_primary_nodes():
            if node.restraint_type == 'fixed':
                ops.fix(node.uniq_id, 1, 1, 1, 1, 1, 1)
            elif node.restraint_type == 'pinned':
                ops.fix(node.uniq_id, 1, 1, 1, 0, 0, 0)
        for node in self.building.list_of_master_nodes():
            ops.fix(node.uniq_id, 0, 0, 1, 1, 1, 0)

    def _define_node_constraints(self):
        # Rigid Diaphragms
        for lvl in self.building.levels.level_list:
            if lvl.master_node:
                ops.rigidDiaphragm(
                    3,
                    lvl.master_node.uniq_id,
                    *[node.uniq_id
                      for node in lvl.list_of_primary_nodes()])
                # for node in lvl.list_of_primary_nodes():
                #     ops.equalDOF(
                #         lvl.master_node.uniq_id,
                #         node.uniq_id,
                #         6
                #     )
                # ops.rigidLink(
                #     "beam",
                #     lvl.master_node.uniq_id,
                #     lvl.list_of_primary_nodes()[-1].uniq_id
                # )
        # connections using mutli-point constraints
        for connection in self.building.list_of_connections():
            if connection.c_type == 'fixed_zerolength':
                ops.equalDOF(
                    connection.primary_node.uniq_id,
                    connection.internal_node.uniq_id,
                    *[1, 2, 3, 4, 5, 6])
            elif connection.c_type == 'rigid_link':
                ops.rigidLink(
                    "beam",
                    connection.primary_node.uniq_id,
                    connection.internal_node.uniq_id
                )
            else:
                raise ValueError("Unsupported connection")

    def _define_node_mass(self):
        for node in self.building.list_of_primary_nodes() + \
                self.building.list_of_master_nodes() + \
                self.building.list_of_internal_nodes():
            if node.mass:
                if max(node.mass.value) > EPSILON:
                    ops.mass(node.uniq_id,
                             *node.mass.value)

    def _define_dead_load(self):
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)

        for elm in \
                self.building.list_of_internal_elems():
            ops.eleLoad('-ele', elm.uniq_id,
                        '-type', '-beamUniform',
                        elm.udl.value[1],
                        elm.udl.value[2],
                        elm.udl.value[0])
        for node in self.building.list_of_primary_nodes() + \
                self.building.list_of_master_nodes() + \
                self.building.list_of_internal_nodes():
            ops.load(node.uniq_id, *node.load.value)

    def _define_sections(self):
        # will be redefined in the child classes
        # but is needed for `_to_OpenSees_domain()`
        pass

    def _define_beamcolumn_elements(self):
        # will be redefined in the child classes
        # but is needed for `_to_OpenSees_domain()`
        pass

    def _to_OpenSees_domain(self):
        """
        Defines the building model in OpenSeesPy
        """
        self._initialize()
        self._define_materials()
        self._define_nodes()
        self._define_node_restraints()
        self._define_node_constraints()
        self._define_node_mass()
        self._define_sections()
        self._define_beamcolumn_elements()

    ####################################################
    # Methods that read back information from OpenSees #
    ####################################################

    def _read_node_displacements(self):
        for node in self.building.list_of_all_nodes():
            self._store_result(self.node_displacements,
                               node.uniq_id,
                               ops.nodeDisp(node.uniq_id))

    def _read_node_reactions(self):
        ops.reactions()
        for node in self.building.list_of_primary_nodes():
            if node.restraint_type != 'free':
                uid = node.uniq_id
                local_reaction = np.array(ops.nodeReaction(uid))
                self._store_result(self.node_reactions,
                                   uid,
                                   local_reaction)

    def _read_frame_element_forces(self):
        for elm in self.building.list_of_internal_elems():
            uid = elm.uniq_id
            forces = np.array(ops.eleForce(uid))
            self._store_result(self.frame_basic_forces,
                               uid,
                               forces)

    def _read_OpenSees_results(self):
        """
        Reads back the results from the OpenSeesPy domain
        """
        self._read_node_displacements()
        self._read_node_reactions()
        self._read_frame_element_forces()

    #########################
    # Visualization methods #
    #########################
    def deformed_shape(self, step=0, scaling=0.00, extrude_frames=False):
        return postprocessing_3D.deformed_shape(self,
                                                step,
                                                scaling,
                                                extrude_frames)

    def basic_forces(self, step=0,
                     scaling_n=0.00,
                     scaling_q=0.00,
                     scaling_m=0.00,
                     scaling_t=0.00,
                     num_points=11):
        return postprocessing_3D.basic_forces(self,
                                              step,
                                              scaling_n,
                                              scaling_q,
                                              scaling_m,
                                              scaling_t,
                                              num_points)

    ##################################
    # Numeric Result Post-processing #
    ##################################

    def global_reactions(self, step):
        reactions = np.full(6, 0.00)
        for lvl in self.building.levels.level_list:
            for node in lvl.list_of_primary_nodes():
                if node.restraint_type != 'free':
                    uid = node.uniq_id
                    x = node.coordinates[0]
                    y = node.coordinates[1]
                    z = node.coordinates[2]
                    local_reaction = self.node_reactions[uid][step]
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

    def _define_sections(self):

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

    def _define_beamcolumn_elements(self):
        for elm in \
                self.building.list_of_internal_elems():
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

    def _run_gravity_analysis(self):
        ops.system('BandGeneral')
        ops.numberer('RCM')
        ops.constraints('Transformation')
        ops.test('NormDispIncr', 1.0e-9, 10, 3)
        ops.algorithm('Newton')
        ops.integrator('LoadControl', 1.0)
        ops.analysis('Static')
        ops.analyze(1)


@dataclass
class LinearGravityAnalysis(LinearAnalysis):
    def run(self):
        self._to_OpenSees_domain()
        self._define_dead_load()
        self._run_gravity_analysis()
        self._read_OpenSees_results()


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

    def _read_node_displacements(self):
        nodes = self.building.list_of_primary_nodes() + \
            self.building.list_of_internal_nodes()
        master_nodes = self.building.list_of_master_nodes()
        if master_nodes:
            all_nodes = nodes + master_nodes
        else:
            all_nodes = nodes
        for node in all_nodes:
            for i in range(self.num_modes):
                self._store_result(self.node_displacements,
                                   node.uniq_id,
                                   ops.nodeEigenvector(
                                       node.uniq_id,
                                       i+1)
                                   )

    def run(self):
        self._to_OpenSees_domain()
        eigValues = np.array(ops.eigen(
            '-fullGenLapack',
            self.num_modes))
        self.periods = 2.00*np.pi / np.sqrt(eigValues)
        self._read_node_displacements()


@dataclass
class NonlinearAnalysis(Analysis):

    n_steps_success: int = field(default=0)

    def _define_sections(self, n_x, n_y, n_p):

        # temporary solution, utnil I
        # manage to get equalDOF to work
        ops.section('Elastic',
                    9900990099009900,
                    1.00,
                    1.00E8,
                    1.00E8,
                    1.00E8,
                    1.00,
                    1.00E8)
        ops.beamIntegration(
            'Lobatto',
            9900990099009900,
            9900990099009900,
            2)

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

    def _beamcolumn_elements(self):
        for elm in \
                self.building.list_of_internal_elems_without_rigid_links():
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
        for elm in self.building.list_of_rigid_links():
            ops.geomTransf('Linear',
                           elm.uniq_id,
                           *elm.local_z_axis_vector())
            ops.element('dispBeamColumn',
                        elm.uniq_id,
                        elm.node_i.uniq_id,
                        elm.node_j.uniq_id,
                        elm.uniq_id,
                        9900990099009900)

            # doesn't work:
            # n_i = elm.node_i.uniq_id
            # n_j = elm.node_j.uniq_id
            # ops.rigidLink("beam", n_i, n_j)

    def _to_OpenSees_domain(self, n_x, n_y, n_p):
        """
        Defines the building model in OpenSeesPy
        """
        self._initialize()
        self._define_materials()
        self._define_nodes()
        self._define_node_restraints()
        self._define_node_constraints()
        self._define_node_mass()
        # the following two columns use methods that
        # are defined in the inherited classes that follow
        self._define_sections(n_x, n_y, n_p)
        self._define_beamcolumn_elements()

    def _run_gravity_analysis(self):
        ops.system('BandGeneral')
        ops.numberer('RCM')
        ops.constraints('Transformation')
        ops.test('NormDispIncr', 1.0e-6, 1000, 3)
        ops.algorithm('Newton')
        ops.integrator('LoadControl', 0.01)
        ops.analysis('Static')
        ops.analyze(100)


@dataclass
class PushoverAnalysis(NonlinearAnalysis):

    def _apply_lateral_load(self, direction):
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
            control_node, displ_incr, displ_output, n_x=10, n_y=25, n_p=10):
        if direction == 'x':
            control_DOF = 0
        elif direction == 'y':
            control_DOF = 1
        else:
            raise ValueError("Direction can either be 'x' or 'y'")
        self._to_OpenSees_domain(n_x, n_y, n_p)
        self._define_dead_load()
        self._run_gravity_analysis()
        ops.wipeAnalysis()
        ops.loadConst('-time', 0.0)
        self._apply_lateral_load(direction)
        ops.system("BandGeneral")
        ops.numberer('RCM')
        ops.constraints('Transformation')
        # TODO add refined steps if fails
        ops.test('NormUnbalance', 1e-6, 1000)
        ops.integrator("DisplacementControl",
                       control_node.uniq_id, control_DOF + 1, displ_incr)
        ops.algorithm("Newton")
        ops.analysis("Static")
        curr_displ = 0.00
        j_out = 0
        n_steps_success = 0
        while curr_displ + EPSILON < target_displacement:
            if curr_displ + EPSILON > displ_output[j_out]:
                self._read_OpenSees_results()
                n_steps_success += 1
                j_out += 1
            check = ops.analyze(1)
            if check != 0:
                print('Analysis failed to converge')
                break
            curr_displ = ops.nodeDisp(control_node.uniq_id, control_DOF+1)
            print('Target displacement: %.2f | Current: %.2f' %
                  (target_displacement, curr_displ), end='\r')

        self._read_OpenSees_results()
        n_steps_success += 1
        print('Number of saved analysis steps:', n_steps_success)
        metadata = {'successful steps': n_steps_success}
        self.n_steps_success = n_steps_success
        return metadata

    def plot_pushover_curve(self, direction, node):
        if not self.node_displacements:
            raise ValueError(
                'No results to plot. Run analysis first.')
        if direction == 'x':
            control_DOF = 0
        elif direction == 'y':
            control_DOF = 1
        else:
            raise ValueError("Direction can either be 'x' or 'y'")
        base_shear = []
        displacement = []
        for step in range(self.n_steps_success):
            base_shear.append(self.global_reactions(step)[control_DOF])
            displacement.append(
                self.node_displacements[node.uniq_id][step][control_DOF])
        base_shear = np.abs(np.array(base_shear))
        displacement = np.abs(np.array(displacement))
        general_2D.line_plot_interactive(
            "Pushover Analysis Results<br>" + "Direction: " + direction,
            displacement, base_shear, 'spline+markers',
            "Displacement", "in", ".0f",
            "Base Shear", "lb", ".0f")


@dataclass
class NLTHAnalysis(NonlinearAnalysis):

    time_vector: List[float] = field(default_factory=list)

    def plot_ground_motion(self, filename, file_time_incr):
        y = np.loadtxt(filename)
        n_points = len(y)
        x = np.arange(0.00, n_points * file_time_incr, file_time_incr)
        general_2D.line_plot_interactive(
            "Ground motion record<br>" +
            filename,
            x, y,
            "line",
            "Time", "s", ".3f",
            "Absolute Acceleration", "g", ".4f")

    def plot_node_displacement_history(self, node, dof):
        if not self.node_displacements:
            raise ValueError(
                'No results to plot. Run analysis first.')
        if dof == 0:
            direction = "x"
        elif dof == 1:
            direction = "y"
        elif dof == 2:
            direction = "z"
        elif dof == 3:
            direction = "rx"
        elif dof == 4:
            direction = "ry"
        elif dof == 5:
            direction = "rz"
        displacement = []
        for step in range(self.n_steps_success):
            displacement.append(
                self.node_displacements[node.uniq_id][step][dof])
        assert(len(self.time_vector) == len(displacement)), \
            'Something went wrong: ' + \
            'time - displacement dimensions do not match'
        general_2D.line_plot_interactive(
            "Displacement time-history<br>" +
            "Node: " + str(node.uniq_id) +
            ", direction: " + direction,
            self.time_vector,
            displacement,
            "line",
            "Time", "s", ".3f",
            "Displacement", "in", ".1f")

    def define_lateral_load_pattern(
            self,
            filename_x,
            filename_y,
            filename_z,
            file_time_incr,
            damping_ratio):

        # define damping
        period = float(1./np.sqrt(ops.eigen('-fullGenLapack', 1)))
        ops.rayleigh(0., 0., 0., 2. * damping_ratio / period)

        error = True
        if filename_x:
            error = False
            # define X-direction TH
            ops.timeSeries('Path', 2, '-dt', file_time_incr,
                           '-filePath', filename_x, '-factor', G_CONST)
            # pattern, direction, timeseries tag
            ops.pattern('UniformExcitation', 2, 1, '-accel', 2)

        if filename_y:
            error = False
            # define Y-direction TH
            ops.timeSeries('Path', 3, '-dt', file_time_incr,
                           '-filePath', filename_y, '-factor', G_CONST)
            # pattern, direction, timeseries tag
            ops.pattern('UniformExcitation', 3, 2, '-accel', 3)

        if filename_z:
            error = False
            # define Z-direction TH
            ops.timeSeries('Path', 4, '-dt', file_time_incr,
                           '-filePath', filename_z, '-factor', G_CONST)
            # pattern, direction, timeseries tag
            ops.pattern('UniformExcitation', 4, 3, '-accel', 4)

        if error:
            raise ValueError(
                "No input files specified.")

    def run(self, target_timestamp, time_increment,
            timestamps_output,
            filename_x,
            filename_y,
            filename_z,
            file_time_incr,
            damping_ratio=0.05,
            n_x=10, n_y=25, n_p=10):
        self._to_OpenSees_domain(n_x, n_y, n_p)
        self._define_dead_load()
        self._run_gravity_analysis()
        ops.wipeAnalysis()
        ops.loadConst('-time', 0.0)

        self.define_lateral_load_pattern(
            filename_x,
            filename_y,
            filename_z,
            file_time_incr,
            damping_ratio
        )

        ops.system("BandGeneral")
        ops.numberer('RCM')
        ops.constraints('Transformation')
        # TODO add refined steps if fails
        ops.test('NormUnbalance', 1e-6, 1000)
        # Create the integration scheme, the Newmark
        # with alpha = 0.5 and beta = .25
        ops.algorithm("Newton")
        ops.integrator('Newmark',  0.5,  0.25)
        ops.analysis("Transient")

        curr_time = 0.00
        j_out = 0
        n_steps_success = 0
        while curr_time + EPSILON < target_timestamp:
            if curr_time + EPSILON > timestamps_output[j_out]:
                self._read_OpenSees_results()
                self.time_vector.append(curr_time)
                n_steps_success += 1
                j_out += 1
            check = ops.analyze(1, time_increment)
            if check != 0:
                print('Analysis failed to converge')
                break
            curr_time = ops.getTime()
            print('Target timestamp: %.2f s | Current: %.2f s' %
                  (target_timestamp, curr_time), end='\r')

        self.time_vector.append(curr_time)
        self._read_OpenSees_results()
        n_steps_success += 1
        print('Number of saved analysis steps:', n_steps_success)
        metadata = {'successful steps': n_steps_success}
        self.n_steps_success = n_steps_success
        return metadata
