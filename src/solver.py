"""
Model Builder for OpenSeesPy ~ Solver module
"""

#   __                 UC Berkeley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   April 2021
#
# https://github.com/ioannis-vm/OpenSees_Model_Builder

from typing import List, TypedDict
from dataclasses import dataclass, field
import openseespy.opensees as ops
import numpy as np
import logging
from datetime import datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import model
import components
from model import Model
from node import Node
from components import LineElementSequence
from utility import common
from utility.graphics import postprocessing_3D
from utility.graphics import general_2D
from utility import transformations

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')

class AnalysisResult(TypedDict):
    uniq_id: int
    results = List


@dataclass
class Analysis:

    building: Model = field(repr=False)
    node_displacements: AnalysisResult = field(
        default_factory=AnalysisResult, repr=False)
    node_velocities: AnalysisResult = field(
        default_factory=AnalysisResult, repr=False)
    node_accelerations: AnalysisResult = field(
        default_factory=AnalysisResult, repr=False)
    node_reactions: AnalysisResult = field(
        default_factory=AnalysisResult, repr=False)
    eleForces: AnalysisResult = field(
        default_factory=AnalysisResult, repr=False)
    fiber_stress_strain: AnalysisResult = field(
        default_factory=AnalysisResult, repr=False)
    release_force_defo: AnalysisResult = field(
        default_factory=AnalysisResult, repr=False)
    log_file: str = field(default=None)


    def __post_init__(self):
        logging.basicConfig(
            filename=self.log_file,
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.DEBUG)
        logging.info('Analysis started')


    def _define_material(self, material: components.Material):
        if material.ops_material == 'Steel01':
            ops.uniaxialMaterial(
                'Steel01',
                material.uniq_id,
                material.parameters['Fy'],
                material.parameters['E0'],
                material.parameters['b'])
        elif material.ops_material == 'Steel02':
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
        elif material.ops_material == 'UVCuniaxial':
            ops.uniaxialMaterial(
                'UVCuniaxial',
                material.uniq_id,
                material.parameters['E0'],
                material.parameters['Fy'],
                *material.parameters['params']
            )
        elif material.ops_material == 'Elastic':
            ops.uniaxialMaterial(
                'Elastic',
                material.uniq_id,
                material.parameters['E']
            )
        elif material.ops_material == 'ElasticPP':
            ops.uniaxialMaterial(
                'ElasticPP',
                material.uniq_id,
                material.parameters['E0'],
                material.parameters['ey']
            )
        elif material.ops_material == 'Hysteretic':
            ops.uniaxialMaterial(
                'Hysteretic',
                material.uniq_id,
                material.parameters['M1y'],
                material.parameters['gamma1_y'],
                material.parameters['M2y'],
                material.parameters['gamma2_y'],
                material.parameters['M3y'],
                material.parameters['gamma3_y'],
                - material.parameters['M1y'],
                - material.parameters['gamma1_y'],
                - material.parameters['M2y'],
                - material.parameters['gamma2_y'],
                - material.parameters['M3y'],
                - material.parameters['gamma3_y'],
                material.parameters['pinchX'],
                material.parameters['pinchY'],
                material.parameters['damage1'],
                material.parameters['damage2'],
                material.parameters['beta']
            )
        elif material.ops_material == 'Bilin':

            ops.uniaxialMaterial(
                'Bilin',
                material.uniq_id,
                material.parameters['initial_stiffness'],
                material.parameters['b+'],
                material.parameters['b-'],
                material.parameters['my+'],
                material.parameters['my-'],
                material.parameters['lamda'],
                material.parameters['lamda'],
                material.parameters['lamda'],
                material.parameters['lamda'],
                1.00, 1.00, 1.00, 1.00,
                material.parameters['theta_p+'],
                material.parameters['theta_p-'],
                material.parameters['theta_pc+'],
                material.parameters['theta_pc-'],
                material.parameters['residual_plus'],
                material.parameters['residual_minus'],
                material.parameters['theta_u'],
                material.parameters['theta_u'],
                material.parameters['d+'],
                material.parameters['d-']
            )

        elif material.ops_material == 'Pinching4':

            ops.uniaxialMaterial(
                'Pinching4',
                material.uniq_id,
                *material.parameters.values())

        else:
            raise ValueError("Unsupported material:" + material.ops_material)

    def _define_node(self, node: model.Node):
        ops.node(node.uniq_id, *node.coords)

        def superimpose_restraints(c_1, c_2):
            assert len(c_1) == len(c_2)
            result = []
            for i in range(len(c_1)):
                result.append(max(c_1[i], c_2[i]))
            return result

        # restraints
        if self.building.global_restraints:
            n_g = self.building.global_restraints
        else:
            n_g = [0, 0, 0, 0, 0, 0]
        if node.restraint_type == 'fixed':
            n_r = [1, 1, 1, 1, 1, 1]
            n_res = superimpose_restraints(n_g, n_r)
            ops.fix(node.uniq_id, *n_res)
        elif node.restraint_type == 'pinned':
            n_r = [1, 1, 1, 0, 0, 0]
            n_res = superimpose_restraints(n_g, n_r)
            ops.fix(node.uniq_id, *n_res)
        elif node.restraint_type == 'parent':
            n_r = [0, 0, 1, 1, 1, 0]
            n_res = superimpose_restraints(n_g, n_r)
            ops.fix(node.uniq_id, *n_res)
        elif node.restraint_type == 'free':
            n_r = [0, 0, 0, 0, 0, 0]
            n_res = superimpose_restraints(n_g, n_r)
            if 1 in n_res:
                ops.fix(node.uniq_id, *n_res)
        else:
            raise ValueError("Invalid restraint type")

        # mass
        if max(node.mass) > common.EPSILON:
            ops.mass(node.uniq_id,
                     *node.mass)

    def _define_elastic_section(self, sec: components.Section):

        # # using AISC database properties
        # # RBS sections won't work
        # ops.section('Elastic',
        #             sec.uniq_id,
        #             sec.material.parameters['E0'],
        #             sec.properties['A'],
        #             sec.properties['Ix'],
        #             sec.properties['Iy'],
        #             sec.material.parameters['G'],
        #             sec.properties['J'])

        # using mesh properties
        ops.section('Elastic',
                    sec.uniq_id,
                    sec.material.parameters['E0'],
                    sec.mesh.geometric_properties()['area'],
                    sec.mesh.geometric_properties()['inertia']['ixx'],
                    sec.mesh.geometric_properties()['inertia']['iyy'],
                    sec.material.parameters['G'],
                    sec.properties['J'])

    def _define_fiber_section(self, sec: components.Section,
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

    def _define_line_element(self, elm: model.LineElement):

        if np.linalg.norm(elm.offset_i) + \
           np.linalg.norm(elm.offset_j) > common.EPSILON:
            ops.geomTransf(elm.geom_transf,
                           elm.uniq_id,
                           *elm.z_axis,
                           '-jntOffset', *elm.offset_i, *elm.offset_j)
        else:
            ops.geomTransf(elm.geom_transf,
                           elm.uniq_id,
                           *elm.z_axis)

        if elm.section.sec_type != 'utility':
            if elm.model_as['type'] == 'elastic':
                ops.element('elasticBeamColumn', elm.uniq_id,
                            elm.node_i.uniq_id,
                            elm.node_j.uniq_id,
                            elm.section.properties['A'],
                            elm.section.material.parameters['E0'],
                            elm.section.material.parameters['G'],
                            elm.section.properties['J'],
                            elm.section.properties['Iy'],
                            elm.section.properties['Ix'],
                            elm.uniq_id)
                # or using the mesh properties (difference is negligible)
                # ops.element('elasticBeamColumn', elm.uniq_id,
                #             elm.node_i.uniq_id,
                #             elm.node_j.uniq_id,
                #             elm.section.mesh.geometric_properties()['area'],
                #             elm.section.material.parameters['E0'],
                #             elm.section.material.parameters['G'],
                #             elm.section.properties['J'],
                #             elm.section.mesh.geometric_properties()['inertia']['iyy'],
                #             elm.section.mesh.geometric_properties()['inertia']['ixx'],
                #             elm.uniq_id)
            else:
                ops.beamIntegration(
                    'Lobatto', elm.uniq_id, elm.section.uniq_id, elm.n_p)
                ops.element('forceBeamColumn',
                            elm.uniq_id,
                            elm.node_i.uniq_id,
                            elm.node_j.uniq_id,
                            elm.uniq_id,
                            elm.uniq_id)
        else:
            ops.element('elasticBeamColumn', elm.uniq_id,
                        elm.node_i.uniq_id,
                        elm.node_j.uniq_id,
                        1.00, common.STIFF, common.STIFF,
                        1.00, 1.00, 1.00,
                        elm.uniq_id)

    def _define_node_constraints(self):
        for lvl in self.building.levels.registry.values():
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
            if (elm.section.uniq_id not in defined_sections) and \
               elm.model_as['type'] != 'elastic':
                sec = elm.section
                # we don't define utility sections in OpenSees
                # (we instead use the appropriate elements)
                if sec.sec_type != 'utility':
                    # define material
                    mat = sec.material
                    if mat.uniq_id \
                       not in defined_materials:
                        self._define_material(mat)
                        defined_materials.append(mat.uniq_id)
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

            # define materials
            mats = elm.materials
            for mat in mats.values():
                if mat.uniq_id not in defined_materials:
                    self._define_material(mat)
                    defined_materials.append(mat.uniq_id)

            dofs = elm.materials.keys()
            mat_tags = [mat.uniq_id for mat in elm.materials.values()]

            # define the ZeroLength element
            ops.element('zeroLength', elm.uniq_id,
                        elm.node_i.uniq_id,
                        elm.node_j.uniq_id,
                        '-mat',
                        *mat_tags,
                        '-dir',
                        *dofs,
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

    def _read_node_displacements(self, data_retention='default'):
        if data_retention == 'default':
            node_list = self.building.list_of_all_nodes()
        elif data_retention == 'lightweight':
            node_list = self.building.list_of_parent_nodes()
        else:
            raise ValueError('Invalid data retention flag: '
                             + data_retention)
        for node in node_list:
            self._store_result(self.node_displacements,
                               node.uniq_id,
                               ops.nodeDisp(node.uniq_id))

    def _read_node_velocities(self, data_retention='default'):
        if data_retention == 'default':
            node_list = self.building.list_of_all_nodes()
        elif data_retention == 'lightweight':
            node_list = self.building.list_of_parent_nodes()
        else:
            raise ValueError('Invalid data retention flag: '
                             + data_retention)
        for node in node_list:
            self._store_result(self.node_velocities,
                               node.uniq_id,
                               ops.nodeVel(node.uniq_id))

    def _read_node_accelerations(self, data_retention='default'):
        if data_retention == 'default':
            node_list = self.building.list_of_all_nodes()
        elif data_retention == 'lightweight':
            node_list = self.building.list_of_parent_nodes()
        else:
            raise ValueError('Invalid data retention flag: '
                             + data_retention)
        for node in node_list:
            self._store_result(self.node_accelerations,
                               node.uniq_id,
                               ops.nodeAccel(node.uniq_id))

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

    def _read_frame_fiber_stress_strain(self):
        for elm in self.building.list_of_line_elements():
            if elm.model_as['type'] != 'fiber':
                continue
            uid = elm.uniq_id
            mat_id = elm.section.material.uniq_id
            result = []
            n_p = elm.n_p
            pts = elm.section.snap_points
            for pt in pts.keys():
                pt = list(pts.keys())[0]
                z_loc = pts[pt][0]
                y_loc = pts[pt][1]
                stress_strain = []
                for i in range(n_p):
                    stress_strain.append(ops.eleResponse(
                        uid, "section", str(i+1), "-fiber", str(y_loc),
                        str(z_loc), str(mat_id), "stressStrain"))
                result.append(stress_strain)
            self._store_result(self.fiber_stress_strain, uid, result)

    def _read_release_moment_rot(self):
        for release in self.building.list_of_endreleases():
            # don't store data for simple pin releases
            # WARNING
            # This code only monitors strong-axis-bending releases
            # If more releases are implemented in the future, this
            # will need to be updated
            if 6 in release.materials:
                if release.materials[6].name in ['auto_pinching',
                                                 'auto__panel_zone_spring',
                                                 'auto_IMK']:
                    # force, deformation in the global system
                    moment_i_global = ops.eleResponse(
                        release.uniq_id, 'force')[3:6]
                    # note: moment_j is the opposite of force_i by equilibrium
                    # no need to store it too
                    # rotation_global = ops.eleResponse(
                    #     release.uniq_id, 'deformation')[3:6]
                    rot_i_global = ops.nodeDisp(release.node_i.uniq_id)[3:6]
                    rot_j_global = ops.nodeDisp(release.node_j.uniq_id)[3:6]
                    rotation_global = np.array(rot_j_global) - \
                        np.array(rot_i_global)
                    # convert to the local system
                    vec_x = release.x_vec
                    vec_y = release.y_vec
                    vec_z = np.cross(vec_x, vec_y)
                    tmat_g2l = transformations.transformation_matrix(
                        vec_x, vec_y, vec_z)
                    moment_i = tmat_g2l @ -(np.array(moment_i_global))
                    deformation = tmat_g2l @ np.array(rotation_global)
                    self._store_result(
                        self.release_force_defo,
                        release.uniq_id,
                        [moment_i[2], deformation[2]])


    def _read_OpenSees_results(self, data_retention='default'):
        """
        Reads back the results from the OpenSeesPy domain
        Args:
            data_retention
            Can be either 'default', or 'lightweight'.
            - default stores everything implemented here
              caution: can fill up RAM
            - lightweight only stores node displ, vel and acc.
        """
        if data_retention == 'default':
            self._read_node_displacements(data_retention)
            self._read_node_velocities(data_retention)
            self._read_node_accelerations(data_retention)
            self._read_node_reactions()
            self._read_frame_element_forces()
            self._read_frame_fiber_stress_strain()
            self._read_release_moment_rot()
        if data_retention == 'lightweight':
            self._read_node_displacements(data_retention)
            self._read_node_velocities(data_retention)
            self._read_node_accelerations(data_retention)

    #########################
    # Visualization methods #
    #########################
    def deformed_shape(self, step=0, scaling=0.00, extrude_frames=False):
        return postprocessing_3D.deformed_shape(self,
                                                step,
                                                scaling,
                                                extrude_frames)

    def basic_forces(self, step=0,
                     scaling_global=1.00,
                     scaling_n=0.00,
                     scaling_q=0.00,
                     scaling_m=0.00,
                     scaling_t=0.00,
                     num_points=11):
        return postprocessing_3D.basic_forces(
            self,
            step,
            scaling_global,
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
        for lvl in self.building.levels.registry.values():
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
        ops.system('UmfPack')
        ops.numberer('Plain')
        # ops.constraints('Penalty', 1.e4, 1.e4)
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
        # tags = ops.getNodeTags()
        # print(len(tags))
        # ops.constraints('Penalty', 1.e9, 1.e9)
        ops.constraints('Transformation')
        ops.system("UmfPack")
        # note: using SparseSYM results in wrong eigendecomposition
        eigValues = np.array(ops.eigen(
            self.num_modes))
        self.periods = 2.00*np.pi / np.sqrt(eigValues)
        self._read_node_displacements()

    def table_shape(self, mode: int):
        data = {'names': [],
                'ux': [], 'uy': []}
        for lvl in self.building.levels.registry.values():
            data['names'].append(lvl.name)
            disp = np.zeros(6)
            # TODO debug  - something may be wrong here.
            for node in lvl.nodes_primary.registry.values():
                disp += np.array(self.node_displacements[node.uniq_id][mode-1])
            disp /= float(len(lvl.nodes_primary.registry))
            data['ux'].append(disp[0])
            data['uy'].append(disp[1])
        data['ux'] /= max(np.max(data['ux']), np.max(data['uy']))
        data['uy'] /= max(np.max(data['ux']), np.max(data['uy']))        
        return pd.DataFrame.from_dict(data)


@dataclass
class NonlinearAnalysis(Analysis):

    n_steps_success: int = field(default=0)

    def _run_gravity_analysis(self):
        ops.system('SparseSYM')
        ops.numberer('Plain')
        # ops.constraints('Penalty', 1.e15, 1.e15)
        ops.constraints('Transformation')
        ops.test('NormDispIncr', 1.0e-6, 100, 3)
        ops.algorithm('RaphsonNewton')
        ops.integrator('LoadControl', 1)
        ops.analysis('Static')
        check = ops.analyze(1)
        if check != 0:
            raise ValueError('Analysis Failed')

    def _acceptance_criteria(self):
        for elm in self.building.list_of_line_elements():

            if elm.model_as['type'] != 'elastic':
                continue
            mat = elm.section.material
            if mat.name == 'steel':
                capacity_t = mat.parameters['Fy']/mat.parameters['E0']
                capacity_c = -capacity_t
            else:
                raise ValueError('Unsupported material')
            strains = []
            x_vec = elm.x_axis
            y_vec = elm.y_axis
            z_vec = elm.z_axis
            T_global2local = np.vstack((x_vec, y_vec, z_vec))
            forces_global = self.eleForces[elm.uniq_id][-1][0:3]
            moments_global_ends = self.eleForces[elm.uniq_id][-1][3:6]

            moments_global_clear = \
                transformations.offset_transformation(
                    elm.offset_i, moments_global_ends, forces_global)

            ni, qyi, qzi = T_global2local @ forces_global
            ti, myi, mzi = T_global2local @ moments_global_clear

            wx, wy, wz = elm.udl_total()

            len_clr = elm.length_clear()
            t = np.linspace(0.00, len_clr, num=9)

            nx_vec = - t * wx - ni
            mz_vec = t**2 * 0.50 * wy + t * qyi - mzi
            my_vec = t**2 * 0.50 * wz + t * qzi + myi

            prop = elm.section.mesh.geometric_properties()
            area = prop['area']
            iy = prop['inertia']['ixx']
            iz = prop['inertia']['iyy']
            young_mod = elm.section.material.parameters['E0']

            for key, val in elm.section.snap_points.items():
                z, y = val
                stress = nx_vec/area \
                    + my_vec/iz * z \
                    - mz_vec/iy * y
                strain = stress / young_mod
                strains.extend(strain)
            emax = np.max(np.array(strains))
            emin = np.min(np.array(strains))
            if ((emax > capacity_t) or (emin < capacity_c)):
                raise ValueError(
                    "Acceptance criteria failed for element " +
                    str(elm.uniq_id))

    def plot_moment_rot(self, seq, step=0):
        """
        Given a LineElementSequence of a type that has any form of end
        release in bending around the strong axis, plots the moment
        rotation for all the analysis steps for both ends.
        """
        # In this case, we plot the moment-rotation relationship
        # for the IMK springs at the two ends, and also the
        # moment-rotation relationship of the panel zone spring
        endsegment_i = seq.end_segment_i
        endsegment_j = seq.end_segment_j
        if isinstance(seq, LineElementSequence_Steel_W_PanelZone_IMK):
            spring_i = endsegment_i.internal_elems[12]
            spring_j = endsegment_j.internal_elems[0]
        elif isinstance(seq, LineElementSequence_IMK):
            spring_i = endsegment_i.internal_elems[1]
            spring_j = endsegment_j.internal_elems[0]
        elif isinstance(seq, LineElementSequence_W_grav_sear_tab):
            spring_i = endsegment_i.internal_elems[1]
            spring_j = endsegment_j.internal_elems[0]
        else:
            return

        data_i = []
        data_j = []
        if step == 0:
            step = self.n_steps_success
        for i in range(step):
            data_i.append(self.release_force_defo[spring_i.uniq_id][i])
            data_j.append(self.release_force_defo[spring_j.uniq_id][i])
        data_i = np.array(data_i)
        data_j = np.array(data_j)

        fig, axs = plt.subplots(nrows=2, sharex=True)
        axs[0].grid()
        axs[1].grid()
        axs[0].plot(data_i[:, 1], data_i[:, 0]/(1.e3 * 12.),
                    color='lightblue')
        axs[0].scatter(data_i[-1, 1], data_i[-1, 0]/(1.e3 * 12.),
                       color='lightblue')
        axs[1].plot(data_j[:, 1], data_j[:, 0]/(1.e3 * 12.),
                    color='orange', label='end j')
        axs[1].scatter(data_j[-1, 1], data_j[-1, 0]/(1.e3 * 12.),
                       color='orange')
        axs[0].set_title('Strong-axis bending moment (kip-ft)')
        axs[0].set_ylabel('End i')
        axs[1].set_ylabel('End j')
        plt.xlabel('Rotation (rad)')
        plt.show()



    def plot_shear_deformation(self, seq, step=0):
        """
        Given a LineElementSequence of a type that has a panel zone,
        plots the shear-deformation for all the analysis steps.
        """
        if isinstance(seq, LineElementSequence_Steel_W_PanelZone_IMK):
            # In this case, we plot the moment-rotation relationship
            # for the IMK springs at the two ends, and also the
            # moment-rotation relationship of the panel zone spring
            endsegment_i = seq.end_segment_i
            endsegment_j = seq.end_segment_j
            spring_i = endsegment_i.internal_elems[12]
            spring_j = endsegment_j.internal_elems[0]
            data_i = []
            data_j = []
            if step == 0:
                step = self.n_steps_success
            for i in range(step):
                data_i.append(self.release_force_defo[spring_i.uniq_id][i])
                data_j.append(self.release_force_defo[spring_j.uniq_id][i])
            data_i = np.array(data_i)
            data_j = np.array(data_j)
 
            fig, axs = plt.subplots(nrows=2, sharex=True)
            axs[0].grid()
            axs[1].grid()
            axs[0].plot(data_i[:, 1], data_i[:, 0]/(1.e3 * 12.),
                        color='lightblue')
            axs[0].scatter(data_i[-1, 1], data_i[-1, 0]/(1.e3 * 12.),
                           color='lightblue')
            axs[1].plot(data_j[:, 1], data_j[:, 0]/(1.e3 * 12.),
                        color='orange', label='end j')
            axs[1].scatter(data_j[-1, 1], data_j[-1, 0]/(1.e3 * 12.),
                           color='orange')
            axs[0].set_title('Strong-axis bending moment (kip-ft)')
            axs[0].set_ylabel('End i')
            axs[1].set_ylabel('End j')
            plt.xlabel('Rotation (rad)')
            plt.show()


@dataclass
class PushoverAnalysis(NonlinearAnalysis):

    def _apply_lateral_load(self, direction, modeshape=None, node=None):
        distribution = self.building.level_masses()
        distribution = distribution / np.linalg.norm(distribution)

        # define the load pattern
        ops.timeSeries('Linear', 2)
        ops.pattern("Plain", 2, 2)

        if direction == 'x':
            load_dir = np.array([1., 0., 0., 0., 0., 0.])
        elif direction == 'y':
            load_dir = np.array([0., 1., 0., 0., 0., 0.])
        elif direction == 'z':
            load_dir = np.array([0., 0., 1., 0., 0., 0.])
        else:
            raise ValueError('Invalid direction')

        nodes = self.building.list_of_primary_nodes()
        if modeshape is not None:
            if direction not in ['x', 'y']:
                raise ValueError(
                    "Can't apply lateral loads based on the 1st " +
                    "mode shape in the z direction.")
            modeshape_ampl = modeshape / modeshape[-1]
        else:
            modeshape_ampl = np.ones(len(self.building.levels.registry.values()))

        # if a node is given, apply the incremental load on that node
        if node:
            ops.load(node.uniq_id, *(1.00*load_dir))
        else:
            for i, lvl in enumerate(self.building.levels.registry.values()):
                # if the level is restrained, no load applied
                if lvl.restraint != 'free':
                    continue
                # if there is a parent node, all load goes there
                if lvl.parent_node:
                    ops.load(lvl.parent_node.uniq_id,
                             *(distribution[i]*load_dir *
                               modeshape_ampl[i]))
                # if there isn't a parent node, distribute that story's load
                # in proportion to the mass of the nodes
                else:
                    node_list = lvl.nodes_primary.registry.values()
                    masses = np.array([n.mass[0] for n in node_list])
                    masses = masses/np.linalg.norm(masses)
                    for j, node in enumerate(node_list):
                        ops.load(node.uniq_id,
                                 *(distribution[i]*masses[j]*load_dir *
                                   modeshape_ampl[i]))

    def run(self, direction, target_displacements,
            control_node, displ_incr, modeshape=None, loaded_node=None):
        if direction == 'x':
            control_DOF = 0
        elif direction == 'y':
            control_DOF = 1
        elif direction == 'z':
            control_DOF = 2
        else:
            raise ValueError("Direction can be 'x', 'y' or 'z'")
        self._to_OpenSees_domain()
        self._define_dead_load()
        self._run_gravity_analysis()
        ops.wipeAnalysis()
        ops.loadConst('-time', 0.0)
        self._apply_lateral_load(direction, modeshape, loaded_node)

        ops.system('UmfPack')
        ops.numberer('Plain')
        # ops.constraints('Penalty', 1.e15, 1.e15)
        ops.constraints('Transformation')
        curr_displ = ops.nodeDisp(control_node.uniq_id, control_DOF+1)
        self._read_OpenSees_results()
        # self._acceptance_criteria()
        n_steps_success = 1

        total_fail = False
        num_subdiv = 0
        num_times = 0

        scale = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        steps = [25, 50, 75, 100, 250, 500]
        norm = [1.0e-6] * 5 + [1.0e-2]
        algor = ['Newton']*6

        try:

            for target_displacement in target_displacements:

                if total_fail:
                    break

                # determine push direction
                if curr_displ < target_displacement:
                    displ_incr = abs(displ_incr)
                    sign = +1.00
                else:
                    displ_incr = -abs(displ_incr)
                    sign = -1.00

                print('entering loop', target_displacement)
                while curr_displ * sign < target_displacement * sign:

                    # determine increment
                    if abs(curr_displ - target_displacement) < \
                       abs(displ_incr) * scale[num_subdiv]:
                        incr = sign * abs(curr_displ - target_displacement)
                    else:
                        incr = displ_incr * scale[num_subdiv]
                    ops.test('NormDispIncr', norm[num_subdiv],
                             steps[num_subdiv], 0)
                    ops.algorithm(algor[num_subdiv])
                    ops.integrator("DisplacementControl",
                                   control_node.uniq_id, control_DOF + 1,
                                   incr)
                    ops.analysis("Static")
                    flag = ops.analyze(1)
                    if flag != 0:
                        # analysis failed
                        if num_subdiv == len(scale) - 1:
                            # can't refine further
                            print()
                            print('===========================')
                            print('Analysis failed to converge')
                            print('===========================')
                            print()
                            total_fail = True
                            break
                        else:
                            # can still reduce step size
                            num_subdiv += 1
                            # how many times to run with reduced step size
                            num_times = 10
                    else:
                        # analysis was successful
                        if num_times != 0:
                            num_times -= 1
                        n_steps_success += 1
                        self._read_OpenSees_results()
                        # self._acceptance_criteria()
                        curr_displ = ops.nodeDisp(
                            control_node.uniq_id, control_DOF+1)
                        print('Target displacement: %.2f | Current: %.4f' %
                              (target_displacement, curr_displ), end='\r')
                        if num_subdiv != 0:
                            if num_times == 0:
                                num_subdiv -= 1
                                num_times = 10
        except KeyboardInterrupt:
            print("Analysis interrupted")
        # finished
        self._read_OpenSees_results()
        # self._acceptance_criteria()
        n_steps_success += 1

        print('Number of saved analysis steps:', n_steps_success)
        metadata = {'successful steps': n_steps_success}
        self.n_steps_success = n_steps_success
        return metadata

    def table_pushover_curve(self, direction, node):
        if not self.node_displacements:
            raise ValueError(
                'No results to plot. Run analysis first.')
        if direction == 'x':
            control_DOF = 0
        elif direction == 'y':
            control_DOF = 1
        elif direction == 'z':
            control_DOF = 2
        else:
            raise ValueError("Direction can be 'x', 'y' or 'z'")
        base_shear = []
        displacement = []
        for step in range(self.n_steps_success):
            base_shear.append(self.global_reactions(step)[control_DOF])
            displacement.append(
                self.node_displacements[node.uniq_id][step][control_DOF])
        base_shear = -np.array(base_shear)
        displacement = np.array(displacement)
        return displacement, base_shear

    def plot_pushover_curve(self, direction, node):
        displacement, base_shear = self.table_pushover_curve(direction, node)
        general_2D.line_plot_interactive(
            "Pushover Analysis Results<br>" + "Direction: " + direction,
            displacement, base_shear, 'spline+markers',
            "Displacement", "in", ".0f",
            "Base Shear", "lb", ".0f")

    def plot_brace_hysteresis(self, brace):
        drift = []
        resisting_force = []
        n_i = brace.node_i.uniq_id
        n_j = brace.node_j.uniq_id
        x_axis = brace.x_axis
        x_axis_horiz = np.array((x_axis[0], x_axis[1], 0.00))
        x_axis_horiz = x_axis_horiz / np.linalg.norm(x_axis_horiz)
        for step in range(self.n_steps_success):
            disp_i = self.node_displacements[n_i][step][0:3]
            disp_j = self.node_displacements[n_j][step][0:3]
            diff_disp = np.array(disp_j) - np.array(disp_i)
            disp_prj = np.dot(diff_disp, x_axis)
            drift.append(disp_prj)
            ielm = brace.end_segment_i.internal_elems[-1].uniq_id
            force = self.eleForces[ielm][step][0:3]
            force_prj = - np.dot(force, x_axis_horiz)
            resisting_force.append(force_prj)
        general_2D.line_plot_interactive(
            "Brace Resisting Force",
            drift, resisting_force, 'line',
            'Story drift', 'in', '.0f',
            'Resisting Force', 'lb', '.0f')


@dataclass
class NLTHAnalysis(NonlinearAnalysis):

    time_vector: List[float] = field(default_factory=list)
    a_g: dict = field(default_factory=dict)

    def run(self, analysis_time_increment,
            filename_x,
            filename_y,
            filename_z,
            file_time_incr,
            finish_time=0.00,
            damping_ratio=0.05,
            num_modes=None,
            printing=True,
            data_retention='default'):
        """
        Run the nonlinear time-history analysis
        Args:
            filename_x, y, z: Paths where the fixed-step ground acceleration
                              records are stored (single-column).
            file_time_incr:   The corresponding time increment
            finish_time: Specify a target time (s) to stop the analysis
                         the default value of 0.00 means that it will
                         run for the entire duration of the files.
            damping_ratio: Self explanatory.
            printing: Controls whether the current time is printed out
            data_retention: Can be 'default' or 'lightweight' (memroy saver).
                            See the docstring of `_read_OpenSees_results`.
        """

        logging.info('Running NLTH analysis')
        nss = []
        if filename_x:
            gm_vals_x = np.genfromtxt(filename_x)
            nss.append(len(gm_vals_x))
        if filename_y:
            gm_vals_y = np.genfromtxt(filename_y)
            nss.append(len(gm_vals_y))
        if filename_z:
            gm_vals_z = np.genfromtxt(filename_z)
            nss.append(len(gm_vals_z))

        logging.info(f'filename_x: {filename_x}')
        logging.info(f'filename_y: {filename_y}')
        logging.info(f'filename_z: {filename_z}')

        num_gm_points = np.min(np.array(nss))
        duration = num_gm_points * file_time_incr

        t = np.linspace(0.00, file_time_incr*num_gm_points, num_gm_points)
        if filename_x:
            self.a_g[0] = interp1d(
                t, gm_vals_x, bounds_error=False, fill_value=0.00)
        if filename_y:
            self.a_g[1] = interp1d(
                t, gm_vals_y, bounds_error=False, fill_value=0.00)
        if filename_z:
            self.a_g[2] = interp1d(
                t, gm_vals_z, bounds_error=False, fill_value=0.00)

        if finish_time == 0.00:
            target_timestamp = duration
        else:
            target_timestamp = finish_time

        logging.info('Defining model in OpenSees')
        self._to_OpenSees_domain()

        # gravity analysis
        logging.info('Defining dead loads')
        self._define_dead_load()
        logging.info('Starting analysis')
        self._run_gravity_analysis()
        self._read_OpenSees_results(data_retention)
        n_steps_success = 1

        import pdb
        pdb.set_trace()

        # time-history analysis
        ops.wipeAnalysis()
        ops.loadConst('-time', 0.0)
        curr_time = 0.00
        self.time_vector.append(curr_time)

        ops.numberer('Plain')
        # ops.constraints('Penalty', 1.e15, 1.e15)
        ops.constraints('Transformation')

        # define damping
        # if fundamental_period is None:
        #     s = float(np.sqrt(ops.eigen('-genBandArpack', 3))[0])
        #     # ops.rayleigh(0., 0., 0., 2. * damping_ratio / s)
        #     ops.rayleigh(0., 0., 0., 2. * damping_ratio / s)
        #     print('T = %.3f s' % (2.00 * np.pi / s))
        # else:
        #     ops.rayleigh(
        #         0.,
        #         0.,
        #         0.,
        #         2. * damping_ratio / (2. * np.pi / fundamental_period)
        #     )

        logging.info(f'Running eigenvalue analysis with {num_modes} modes')
        ops.system("UmfPack")
        ops.eigen(num_modes)
        logging.info(f'Eigenvalue analysis finished')
        ops.modalDamping(damping_ratio)
        logging.info(f'{damping_ratio*100:.2f}% modal damping defined')


        ops.system("SparseSYM")
        ops.test('NormDispIncr', 1e-6, 50, 0)
        ops.algorithm("KrylovNewton")
        # ops.integrator("TRBDF2")
        ops.analysis("Transient")


        self.define_lateral_load_pattern(
            filename_x,
            filename_y,
            filename_z,
            file_time_incr,
            damping_ratio,
            num_modes)

        num_subdiv = 0
        num_times = 0
        analysis_failed = False

        scale = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]

        # for analysis speed stats:
        total_step_count = 0
        now = datetime.now()
        import time
        time.sleep(2)

        try:

            while curr_time + common.EPSILON < target_timestamp:

                # # TODO create a method using this code
                # # to automate the fastest seup selection process
                # # debug
                # systems = ['SparseSYM', 'UmfPack']
                # algos = ['NewtonLineSearch', 'ModifiedNewton', 'KrylovNewton', 'SecantNewton', 'RaphsonNewton', 'PeriodicNewton', 'BFGS', 'Broyden']
                # integrs = [('Newmark', 0.50, 0.25), ('HHT', 1.0), ('GeneralizedAlpha', 1., 1.), ('TRBDF2',)]

                # print('Testing systems')

                # times = []
                # best_sys = 'UmfPack'
                # best_algo = 'Newton'
                # best_integ = ('Newmark', 0.50, 0.25)
                # for syst in systems:
                #     ops.system(syst)
                #     ops.algorithm(best_algo)
                #     ops.integrator(*best_integ)

                #     tt = datetime.now()
                #     check = ops.analyze(
                #         20, analysis_time_increment * scale[num_subdiv])
                #     ttt = datetime.now()
                #     times.append(ttt - tt)
                # mn = min(times)
                # imn = times.index(mn)
                # best_sys = systems[imn]
                # print()
                # print(times)
                # print(best_sys)
                # print()

                # times = []
                # for algo in algos:
                #     ops.system(best_sys)
                #     ops.algorithm(algo)
                #     ops.integrator(*best_integ)
                #     tt = datetime.now()
                #     check = ops.analyze(
                #         20, analysis_time_increment * scale[num_subdiv])
                #     ttt = datetime.now()
                #     times.append(ttt - tt)
                # mn = min(times)
                # imn = times.index(mn)
                # best_algo = algos[imn]
                # print()
                # print(times)
                # print(best_algo)
                # print()


                # times = []
                # for integ in integrs:
                #     ops.system(best_sys)
                #     ops.algorithm(best_algo)
                #     ops.integrator(*integ)
                #     tt = datetime.now()
                #     check = ops.analyze(
                #         20, analysis_time_increment * scale[num_subdiv])
                #     ttt = datetime.now()
                #     times.append(ttt - tt)
                # mn = min(times)
                # imn = times.index(mn)
                # best_integ = integrs[imn]
                # print()
                # print(times)
                # print(best_integ)
                # print()

                check = ops.analyze(
                    1, analysis_time_increment * scale[num_subdiv])

                # analysis speed stats
                total_step_count += 1
                speed = total_step_count / (datetime.now() - now).seconds
                if total_step_count % 50 == 0:
                    logging.info(f'Average speed: {speed:.2f} steps/s')
                    print(f'Average speed: {speed:.2f} steps/s')

                if check != 0:
                    # analysis failed
                    if num_subdiv == len(scale) - 1:
                        print()
                        print('===========================')
                        print('Analysis failed to converge')
                        print('===========================')
                        print()
                        logging.warning(
                            f"Analysis failed at time {curr_time:.5f}")
                        analysis_failed = True
                        break
                    else:
                        # can still reduce step size
                        num_subdiv += 1
                        # how many times to run with reduced step size
                        num_times = 10
                else:
                    # analysis was successful
                    if num_times != 0:
                        num_times -= 1
                    n_steps_success += 1
                    curr_time = ops.getTime()
                    self.time_vector.append(curr_time)
                    self._read_OpenSees_results(data_retention)
                    if printing:
                        print('Target timestamp: %.2f s | Current: %.4f s' %
                              (target_timestamp, curr_time), end='\r')
                    if num_subdiv != 0:
                        if num_times == 0:
                            num_subdiv -= 1
                            num_times = 10

        except KeyboardInterrupt:
            print("Analysis interrupted")
            logging.warning("Analysis interrupted")

        metadata = {'successful steps': n_steps_success,
                    'analysis_finished_successfully': not analysis_failed}
        self.n_steps_success = n_steps_success
        logging.info('Analysis finished')

        return metadata

    def plot_ground_motion(self, filename, file_time_incr, plotly=False):
        y = np.loadtxt(filename)
        n_points = len(y)
        x = np.arange(0.00, n_points * file_time_incr, file_time_incr)
        if plotly:
            general_2D.line_plot_interactive(
                "Ground motion record<br>" +
                filename,
                x, y,
                "line",
                "Time", "s", ".3f",
                "Absolute Acceleration", "g", ".4f")
        else:
            plt.figure()
            plt.plot(x, y, 'k')
            plt.xlabel('Time (s)')
            plt.ylabel('Acceleration (g)')
            plt.show()

    def table_node__history(self, node, dof, kind='Displacement'):

        if not self.node_displacements:
            raise ValueError(
                'No results to give. Run analysis first!')

        response_quant = []
        for step in range(self.n_steps_success):

            if kind == 'Displacement':
                response_quant.append(
                    self.node_displacements[node.uniq_id][step][dof])
            elif kind == 'Acceleration':
                time = self.time_vector[step]
                response_quant.append(
                    self.node_accelerations[
                        node.uniq_id][step][dof] / common.G_CONST +
                    float(self.a_g[dof](time)))
            else:
                raise ValueError('Invalid response type: %s' % (kind))

        assert(len(self.time_vector) == len(response_quant)), \
            'Something went wrong: ' + \
            'time - displacement dimensions do not match'

        return self.time_vector, response_quant

    def plot_node__history(self, node, dof, kind='Displacement'):

        time, response = self.table_node__history(node, dof, kind)

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

        general_2D.line_plot_interactive(
            kind + " time-history<br>" +
            "Node: " + str(node.uniq_id) +
            ", direction: " + direction,
            time,
            response,
            "line",
            "Time", "s", ".3f",
            kind, "in", ".1f")

    def define_lateral_load_pattern(
            self,
            filename_x,
            filename_y,
            filename_z,
            file_time_incr,
            damping_ratio,
            num_modes=3):

        error = True
        if filename_x:
            error = False
            # define X-direction TH
            ops.timeSeries('Path', 2, '-dt', file_time_incr,
                           '-filePath', filename_x,
                           '-factor', common.G_CONST)
            # pattern, direction, timeseries tag
            ops.pattern('UniformExcitation', 2, 1, '-accel', 2)

        if filename_y:
            error = False
            # define Y-direction TH
            ops.timeSeries('Path', 3, '-dt', file_time_incr,
                           '-filePath', filename_y,
                           '-factor', common.G_CONST)
            # pattern, direction, timeseries tag
            ops.pattern('UniformExcitation', 3, 2, '-accel', 3)

        if filename_z:
            error = False
            # define Z-direction TH
            ops.timeSeries('Path', 4, '-dt', file_time_incr,
                           '-filePath', filename_z,
                           '-factor', common.G_CONST)
            # pattern, direction, timeseries tag
            ops.pattern('UniformExcitation', 4, 3, '-accel', 4)

        if error:
            raise ValueError(
                "No input files specified.")
