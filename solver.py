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
import modeler
from modeler import Building
from utility.common import G_CONST, EPSILON
from utility.graphics import postprocessing_3D
from utility.graphics import general_2D


def plot_stress_strain(material: modeler.Material,
                       num_steps: int,
                       sigma_max: float):
    ops.wipe()
    ops.model('basic', '-ndm', 1, '-ndf', 1)

    if material.ops_material == 'Steel02':
        ops.uniaxialMaterial('Steel02',
                             material.uniq_id,
                             material.parameters['Fy'],
                             material.parameters['E0'],
                             material.parameters['b'],
                             *material.parameters['params'],
                             material.parameters['a1'],
                             material.parameters['a2'],
                             material.parameters['a3'],
                             material.parameters['a4'],
                             material.parameters['sigInit'])
    else:
        raise ValueError("Unsupported material")
    node1_id = 1
    node1_coords = [0.]
    ops.node(node1_id, *node1_coords)
    node2_id = 2
    node2_coords = [1.]
    ops.node(node2_id, *node2_coords)
    ops.fix(node1_id, 1)
    elm1_id = 1
    area = 1.00
    ops.element("Truss", elm1_id, node1_id, node2_id,
                area, material.uniq_id)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(node2_id, sigma_max)
    ops.system("ProfileSPD")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0/num_steps)
    ops.algorithm("Newton")
    ops.test('NormUnbalance', 1e-8, 10)
    ops.analysis("Static")
    data = np.zeros((num_steps+1, 2))
    for j in range(num_steps):
        ops.analyze(1)
        data[j+1, 0] = ops.nodeDisp(node2_id, 1)
        data[j+1, 1] = ops.getLoadFactor(1)*sigma_max
    general_2D.line_plot_interactive(
        "Stress-strain plot",
        data[:, 0], data[:, 1],
        'spline+markers',
        "strain", None, '.3e',
        "stress", None, '.3e')


class AnalysisResult(TypedDict):
    uniq_id: int
    results = List


@dataclass
class Analysis:

    building: Building
    node_displacements: AnalysisResult = field(
        default_factory=AnalysisResult)
    node_reactions: AnalysisResult = field(default_factory=AnalysisResult)
    eleForces: AnalysisResult = field(default_factory=AnalysisResult)
    fiber_stress_strain: AnalysisResult = field(default_factory=AnalysisResult)

    def _define_material(self, material: modeler.Material):
        if material.ops_material == 'Steel02':
            ops.uniaxialMaterial(
                'Steel02',
                material.uniq_id,
                material.parameters['Fy'],
                material.parameters['E0'],
                material.parameters['b'],
                *material.parameters['params'],
                material.parameters['a1'],
                material.parameters['a2'],
                material.parameters['a3'],
                material.parameters['a4'],
                material.parameters['sigInit'])
        elif material.ops_material == 'Elastic':
            ops.uniaxialMaterial(
                'Elastic',
                material.uniq_id,
                material.parameters['E']
            )
        else:
            raise ValueError("Unsupported material")

    def _define_node(self, node: modeler.Node):
        ops.node(node.uniq_id, *node.coords)
        if node.restraint_type == 'fixed':
            ops.fix(node.uniq_id, 1, 1, 1, 1, 1, 1)
        elif node.restraint_type == 'pinned':
            ops.fix(node.uniq_id, 1, 1, 1, 0, 0, 0)
        elif node.restraint_type == 'parent':
            ops.fix(node.uniq_id, 0, 0, 1, 1, 1, 0)
        elif node.restraint_type == 'free':
            pass
        else:
            raise ValueError("Invalid restraint type")
        if max(node.mass) > EPSILON:
            ops.mass(node.uniq_id,
                     *node.mass)

    def _define_elastic_section(self, sec: modeler.Section):

        ops.section('Elastic',
                    sec.uniq_id,
                    sec.material.parameters['E0'],
                    sec.properties['A'],
                    sec.properties['Ix'],
                    sec.properties['Iy'],
                    sec.material.parameters['G'],
                    sec.properties['J'])

    def _define_fiber_section(self, sec: modeler.Section,
                              n_x: int, n_y: int):

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

    def _define_line_element(self, elm: modeler.LineElement):
        ops.geomTransf(elm.geomTransf,
                       elm.uniq_id,
                       *elm.z_axis)
        ops.beamIntegration(
            'Lobatto', elm.uniq_id, elm.section.uniq_id, 5)
        ops.element('dispBeamColumn',
                    elm.uniq_id,
                    elm.node_i.uniq_id,
                    elm.node_j.uniq_id,
                    elm.uniq_id,
                    elm.uniq_id)

    def _define_node_constraints(self):
        for lvl in self.building.levels.level_list:
            if lvl.parent_node:
                ops.rigidDiaphragm(
                    3,
                    lvl.parent_node.uniq_id,
                    *[node.uniq_id
                      for node in lvl.list_of_primary_nodes()])

    def _to_OpenSees_domain(self):
        """
        Defines the building model in OpenSeesPy
        """

        def define_node(node, defined_nodes):
            if node.uniq_id not in defined_nodes:
                self._define_node(node)
                defined_nodes.append(node.uniq_id)

        defined_nodes = []
        defined_sections = []
        defined_materials = []

        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        # define line elements
        for elm in self.building.list_of_line_elements():

            define_node(elm.node_i, defined_nodes)
            define_node(elm.node_j, defined_nodes)

            # define section
            if elm.section.uniq_id not in defined_sections:
                sec = elm.section
                # define material
                if sec.material.uniq_id \
                   not in defined_materials:
                    self._define_material(sec.material)
                    defined_materials.append(sec.material.uniq_id)
                if elm.model_as['type'] == 'elastic':
                    self._define_elastic_section(sec)
                elif elm.model_as['type'] == 'fiber':
                    n_x = elm.model_as['n_x']
                    n_y = elm.model_as['n_y']
                    self._define_fiber_section(sec, n_x, n_y)
                else:
                    raise ValueError("Invalid modeling type")
                defined_sections.append(elm.section.uniq_id)

            self._define_line_element(elm)

        # define zerolength elements representing end releases
        for elm in self.building.list_of_endreleases():

            define_node(elm.node_i, defined_nodes)
            define_node(elm.node_j, defined_nodes)

            # define fix material
            if elm.mat_fix.uniq_id not in defined_materials:
                self._define_material(elm.mat_fix)
                defined_materials.append(elm.mat_fix.uniq_id)
            # define release material
            if elm.mat_release.uniq_id not in defined_materials:
                self._define_material(elm.mat_release)
                defined_materials.append(elm.mat_release.uniq_id)

            # construct a list of material tags
            all_dofs = [1, 2, 3, 4, 5, 6]
            free_dofs = elm.free_dofs
            mat_tags = []
            for dof in all_dofs:
                if dof in free_dofs:
                    mat_tags.append(elm.mat_release.uniq_id)
                else:
                    mat_tags.append(elm.mat_fix.uniq_id)

            # define the ZeroLength element
            ops.element('zeroLength', elm.uniq_id,
                        elm.node_i.uniq_id,
                        elm.node_j.uniq_id,
                        '-mat',
                        *mat_tags,
                        '-dir',
                        *all_dofs,
                        '-orient',
                        *elm.x_vec,
                        *elm.y_vec)

        # define parent nodes
        for node in self.building.list_of_parent_nodes():
            define_node(node, defined_nodes)

        # define constraints
        self._define_node_constraints()

    def _define_dead_load(self):
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)

        for elm in self.building.list_of_line_elements():
            ops.eleLoad('-ele', elm.uniq_id,
                        '-type', '-beamUniform',
                        elm.udl_total()[1],
                        elm.udl_total()[2],
                        elm.udl_total()[0])
        for node in self.building.list_of_primary_nodes() + \
                self.building.list_of_parent_nodes() + \
                self.building.list_of_internal_nodes():
            ops.load(node.uniq_id, *node.load_total())

    ####################################################
    # Methods that read back information from OpenSees #
    ####################################################

    def _store_result(self, analysis_result: AnalysisResult,
                      uniq_id: int, result: list):
        if uniq_id in analysis_result.keys():
            analysis_result[uniq_id].append(result)
        else:
            analysis_result[uniq_id] = [result]

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
        for elm in self.building.list_of_line_elements():
            uid = elm.uniq_id
            forces = np.array(ops.eleForce(uid))
            self._store_result(self.eleForces,
                               uid,
                               forces)

    def _read_frame_fiber_stress_strain(self, n_p):
        for elm in self.building.list_of_line_elements():
            uid = elm.uniq_id
            mat_id = elm.section.material.uniq_id
            result = []
            for fiber in elm.section.subdivide_section(10, 25):
                z_loc = fiber.centroid.x
                y_loc = fiber.centroid.y
                stress_strain = [ops.eleResponse(
                    uid, "section", i, "-fiber", str(y_loc),
                    str(z_loc), str(mat_id), "stressStrain")
                    for i in range(n_p)]
                result.append(stress_strain)
            self._store_result(self.fiber_stress_strain, uid, result)

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
                    x = node.coords[0]
                    y = node.coords[1]
                    z = node.coords[2]
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

    def _run_gravity_analysis(self):
        ops.system('FullGeneral')
        ops.numberer('RCM')
        ops.constraints('Transformation')
        ops.test('NormDispIncr', 1.0e-8, 20, 3)
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
        nodes = []
        nodes.extend(self.building.list_of_primary_nodes())
        nodes.extend(self.building.list_of_internal_nodes())
        parent_nodes = self.building.list_of_parent_nodes()
        if parent_nodes:
            all_nodes = nodes + parent_nodes
        else:
            all_nodes = nodes
        for node in all_nodes:
            for i in range(self.num_modes):
                self._store_result(
                    self.node_displacements,
                    node.uniq_id,
                    ops.nodeEigenvector(
                        node.uniq_id,
                        i+1))

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

    def _define_beamcolumn_elements(self, n_p):
        for elm in \
                self.building.list_of_line_elements():
            n_sub = int(n_p / elm.parent_n_sub)
            if n_sub < 2:
                n_sub = 2
            # beamIntegration('Lobatto', tag, secTag, N)
            ops.beamIntegration(
                'Lobatto', elm.uniq_id, elm.section.uniq_id, n_p)
            # geometric transformation
            ops.geomTransf('Linear',
                           elm.uniq_id,
                           *elm.z_axis)
            # element('dispBeamColumn', eleTag, *eleNodes,
            #         transfTag, integrationTag, '-cMass',
            #         '-mass', mass=0.0)
            ops.element('dispBeamColumn',
                        elm.uniq_id,
                        elm.node_i.uniq_id,
                        elm.node_j.uniq_id,
                        elm.uniq_id,
                        elm.uniq_id)

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
        self._define_sections(n_x, n_y)
        self._define_beamcolumn_elements(n_p)

    def _run_gravity_analysis(self):
        ops.system('SuperLU')
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
            # if there is a parent node, all load goes there
            if lvl.parent_node:
                ops.load(lvl.parent_node.uniq_id,
                         *(distribution[i]*load_dir))
            # if there isn't a parent node, distribute that story's load
            # in proportion to the mass of the nodes
            else:
                node_list = lvl.nodes_primary.node_list
                masses = np.array([n.mass[0] for n in node_list])
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
        ops.system('ProfileSPD')
        ops.numberer('RCM')
        ops.constraints('Transformation')
        # TODO add refined steps if fails
        ops.test('NormUnbalance', 1e-6, 2000)
        ops.integrator("ParallelDisplacementControl",
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
                ops.integrator("DisplacementControl",
                               control_node.uniq_id, control_DOF + 1,
                               displ_incr/10.)
                check = ops.analyze(1)
                if check != 0:
                    print('Analysis failed to converge')
                    break
                ops.integrator("DisplacementControl",
                               control_node.uniq_id, control_DOF + 1,
                               displ_incr)
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

        ops.system("SuperLU")
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
