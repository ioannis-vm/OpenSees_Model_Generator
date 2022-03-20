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
import os
import shelve
import pickle
import numpy as np
from scipy import integrate
import logging
from datetime import datetime
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import components
from model import Model
import node
from components import LineElement
from utility import common
from utility.graphics import postprocessing_3D
from utility.graphics import general_2D
from utility import transformations

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')

LN_ANALYSIS_SYSTEM = 'SparseSYM'
NL_ANALYSIS_SYSTEM = 'SparseSYM'
MD_ANALYSIS_SYSTEM = 'UmfPack'
CONSTRAINTS = ('Transformation',)
NUMBERER = 'Plain'


class AnalysisResult(TypedDict):

    uid: int
    results = List


@dataclass
class Analysis:

    building: Model = field(repr=False)
    log_file: str = field(default=None)
    output_directory: str = field(default=None)
    disk_storage: bool = field(default=False)
    store_forces: bool = field(default=True)
    store_reactions: bool = field(default=True)
    store_fiber: bool = field(default=True)
    store_release_force_defo: bool = field(default=True)
    specific_nodes: list = field(default_factory=list)

    def __post_init__(self):
        logging.basicConfig(
            filename=self.log_file,
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.DEBUG)

        if self.output_directory and not os.path.exists(self.output_directory):
            os.makedirs(
                self.output_directory,
                exist_ok=True)

        if not self.disk_storage:
            self.node_displacements = AnalysisResult()
            self.node_velocities = AnalysisResult()
            self.node_accelerations = AnalysisResult()
            if self.store_reactions:
                self.node_reactions = AnalysisResult()
            if self.store_forces:
                self.element_forces = AnalysisResult()
            if self.store_fiber:
                self.fiber_stress_strain = AnalysisResult()
            if self.store_release_force_defo:
                self.release_force_defo = AnalysisResult()
        else:
            self.node_displacements = \
                shelve.open(
                    f'{self.output_directory}/node_displacements',
                    writeback=True)
            self.node_velocities = \
                shelve.open(
                    f'{self.output_directory}/node_velocities',
                    writeback=True)
            self.node_accelerations = \
                shelve.open(
                    f'{self.output_directory}/node_accelerations',
                    writeback=True)
            if self.store_reactions:
                self.node_reactions = \
                    shelve.open(
                        f'{self.output_directory}/node_reactions',
                        writeback=True)
            if self.store_forces:
                self.element_forces = \
                    shelve.open(
                        f'{self.output_directory}/element_forces',
                        writeback=True)
            if self.store_fiber:
                self.fiber_stress_strain = \
                    shelve.open(
                        f'{self.output_directory}/fiber_stress_strain',
                        writeback=True)
            if self.store_release_force_defo:
                self.release_force_defo = \
                    shelve.open(
                        f'{self.output_directory}/release_force_defo',
                        writeback=True)

        logging.info('Analysis started')
        if not self.output_directory:
            curr = datetime.now()
            self.output_directory = curr.strftime(
                '/tmp/analysis_%Y-%m-%d_%I-%M-%S_%p')
            logging.info(
                'Using default output directory: '
                f'{self.output_directory}')
            if not os.path.exists(self.output_directory):
                os.makedirs(
                    self.output_directory,
                    exist_ok=True)

    def _sync_results(self):

        if self.disk_storage:
            self.node_displacements.sync()
            self.node_velocities.sync()
            self.node_accelerations.sync()
            if self.store_reactions:
                self.node_reactions.sync()
            if self.store_forces:
                self.element_forces.sync()
            if self.store_fiber:
                self.fiber_stress_strain.sync()
            if self.store_release_force_defo:
                self.release_force_defo.sync()

    def _write_results(self):

        def dump(varname, var):
            with open(
                    f'{self.output_directory}/{varname}.pcl',
                    'wb') as file:
                pickle.dump(var, file)

        if not self.disk_storage:

            logging.info('Storing analysis results')

            dump('node_displacements', self.node_displacements)
            dump('node_velocities', self.node_velocities)
            dump('node_accelerations', self.node_accelerations)
            if hasattr(self, 'node_reactions'):
                dump('node_reactions', self.node_reactions)
            if hasattr(self, 'element_forces'):
                dump('element_forces', self.element_forces)
            if hasattr(self, 'fiber_stress_strain'):
                dump('fiber_stress_strain', self.fiber_stress_strain)
            if hasattr(self, 'release_force_defo'):
                dump('release_force_defo', self.release_force_defo)

            logging.info('Analysis results stored successfully')

        else:
            self.node_displacements.close()
            self.node_velocities.close()
            self.node_accelerations.close()
            if hasattr(self, 'node_reactions'):
                self.node_reactions.close()
            if hasattr(self, 'element_forces'):
                self.element_forces.close()
            if hasattr(self, 'fiber_stress_strain'):
                self.fiber_stress_strain.close()
            if hasattr(self, 'release_force_defo'):
                self.release_force_defo.close()

        if hasattr(self, 'num_modes'):
            dump('num_modes', self.num_modes)
        if hasattr(self, 'periods'):
            dump('periods', self.periods)
        if hasattr(self, 'n_steps_success'):
            dump('n_steps_success', self.n_steps_success)
        if hasattr(self, 'time_vector'):
            dump('time_vector', self.time_vector)
        if hasattr(self, 'a_g'):
            dump('a_g', self.a_g)

    def read_results(self):

        def read(varname):
            with open(
                    f'{self.output_directory}/{varname}.pcl',
                    'rb') as file:
                return pickle.load(file)

        if not self.disk_storage:

            logging.info('Reading analysis results')

            self.node_displacements = read('node_displacements')
            self.node_velocities = read('node_velocities')
            self.node_accelerations = read('node_accelerations')
            if os.path.exists(f'{self.output_directory}/node_reactions.pcl'):
                self.node_reactions = read('node_reactions')
            if os.path.exists(f'{self.output_directory}/element_forces.pcl'):
                self.element_forces = read('element_forces')
            if os.path.exists(f'{self.output_directory}/fiber_stress_strain.pcl'):
                self.fiber_stress_strain = read('fiber_stress_strain')
            if os.path.exists(f'{self.output_directory}/release_force_defo.pcl'):
                self.release_force_defo = read('release_force_defo')

        else:

            self.node_displacements = \
                shelve.open(
                    f'{self.output_directory}/node_displacements',
                    writeback=True)
            self.node_velocities = \
                shelve.open(
                    f'{self.output_directory}/node_velocities',
                    writeback=True)
            self.node_accelerations = \
                shelve.open(
                    f'{self.output_directory}/node_accelerations',
                    writeback=True)
            if self.store_reactions:
                self.node_reactions = \
                    shelve.open(
                        f'{self.output_directory}/node_reactions',
                        writeback=True)
            if self.store_forces:
                self.element_forces = \
                    shelve.open(
                        f'{self.output_directory}/element_forces',
                        writeback=True)
            if self.store_fiber:
                self.fiber_stress_strain = \
                    shelve.open(
                        f'{self.output_directory}/fiber_stress_strain',
                        writeback=True)
            if self.store_release_force_defo:
                self.release_force_defo = \
                    shelve.open(
                        f'{self.output_directory}/release_force_defo',
                        writeback=True)

        # and regardless of storage method
        if os.path.exists(f'{self.output_directory}/num_modes.pcl'):
            self.num_modes = read('num_modes')
        if os.path.exists(f'{self.output_directory}/periods.pcl'):
            self.periods = read('periods')
        if os.path.exists(f'{self.output_directory}/n_steps_success.pcl'):
            self.n_steps_success = read('n_steps_success')
        if os.path.exists(f'{self.output_directory}/time_vector.pcl'):
            self.time_vector = read('time_vector')
        if os.path.exists(f'{self.output_directory}/a_g.pcl'):
            self.a_g = read('a_g')

        logging.info('Analysis results read successfully')


    def _define_material(self, material: components.Material):
        if material.ops_material == 'Steel01':
            ops.uniaxialMaterial(
                'Steel01',
                material.uid,
                material.parameters['Fy'],
                material.parameters['E0'],
                material.parameters['b'])
        elif material.ops_material == 'Steel02':
            ops.uniaxialMaterial(
                'Steel02',
                material.uid,
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
                material.uid,
                material.parameters['E0'],
                material.parameters['Fy'],
                *material.parameters['params']
            )
        elif material.ops_material == 'Elastic':
            ops.uniaxialMaterial(
                'Elastic',
                material.uid,
                material.parameters['E']
            )
        elif material.ops_material == 'ElasticPP':
            ops.uniaxialMaterial(
                'ElasticPP',
                material.uid,
                material.parameters['E0'],
                material.parameters['ey']
            )
        elif material.ops_material == 'Hysteretic':
            ops.uniaxialMaterial(
                'Hysteretic',
                material.uid,
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
                material.uid,
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
                material.uid,
                *material.parameters.values())

        else:
            raise ValueError("Unsupported material:" + material.ops_material)

    def _define_node(self, nd: node.Node):
        ops.node(nd.uid, *nd.coords)

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
        if nd.restraint_type == 'fixed':
            n_r = [1, 1, 1, 1, 1, 1]
            n_res = superimpose_restraints(n_g, n_r)
            ops.fix(nd.uid, *n_res)
        elif nd.restraint_type == 'pinned':
            n_r = [1, 1, 1, 0, 0, 0]
            n_res = superimpose_restraints(n_g, n_r)
            ops.fix(nd.uid, *n_res)
        elif nd.restraint_type == 'parent':
            n_r = [0, 0, 1, 1, 1, 0]
            n_res = superimpose_restraints(n_g, n_r)
            ops.fix(nd.uid, *n_res)
        elif nd.restraint_type == 'free':
            n_r = [0, 0, 0, 0, 0, 0]
            n_res = superimpose_restraints(n_g, n_r)
            if 1 in n_res:
                ops.fix(nd.uid, *n_res)
        else:
            raise ValueError("Invalid restraint type")

        # mass
        if max(nd.mass) > common.EPSILON:
            if len(nd.mass) == 3:
                mass = list(nd.mass)+[common.EPSILON]*3
            else:
                mass = nd.mass
            ops.mass(nd.uid, *mass)

    def _define_elastic_section(self, sec: components.Section):

        # # using AISC database properties
        # # RBS sections won't work
        # ops.section('Elastic',
        #             sec.uid,
        #             sec.material.parameters['E0'],
        #             sec.properties['A'],
        #             sec.properties['Ix'],
        #             sec.properties['Iy'],
        #             sec.material.parameters['G'],
        #             sec.properties['J'])

        # using mesh properties
        ops.section('Elastic',
                    sec.uid,
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
                    sec.uid,
                    '-GJ',
                    sec.properties['J']*sec.material.parameters['G'])
        for piece in pieces:
            area = piece.area
            z_loc = piece.centroid.x
            y_loc = piece.centroid.y
            ops.fiber(y_loc,
                      z_loc,
                      area,
                      sec.material.uid)

    def _define_line_element(self, elm: LineElement):

        if np.linalg.norm(elm.offset_i) + \
           np.linalg.norm(elm.offset_j) > common.EPSILON:
            ops.geomTransf(elm.geom_transf,
                           elm.uid,
                           *elm.z_axis,
                           '-jntOffset', *elm.offset_i, *elm.offset_j)
        else:
            ops.geomTransf(elm.geom_transf,
                           elm.uid,
                           *elm.z_axis)

        if elm.section.sec_type != 'utility':
            if elm.model_as['type'] == 'elastic':
                ops.element('elasticBeamColumn', elm.uid,
                            elm.node_i.uid,
                            elm.node_j.uid,
                            elm.section.properties['A'],
                            elm.section.material.parameters['E0'],
                            elm.section.material.parameters['G'],
                            elm.section.properties['J'],
                            elm.section.properties['Iy'],
                            elm.section.properties['Ix'],
                            elm.uid)
                # or using the mesh properties (difference is negligible)
                # ops.element('elasticBeamColumn', elm.uid,
                #             elm.node_i.uid,
                #             elm.node_j.uid,
                #             elm.section.mesh.geometric_properties()['area'],
                #             elm.section.material.parameters['E0'],
                #             elm.section.material.parameters['G'],
                #             elm.section.properties['J'],
                #             elm.section.mesh.geometric_properties()['inertia']['iyy'],
                #             elm.section.mesh.geometric_properties()['inertia']['ixx'],
                #             elm.uid)
            else:
                ops.beamIntegration(
                    'Lobatto', elm.uid, elm.section.uid, elm.n_p)
                ops.element('forceBeamColumn',
                            elm.uid,
                            elm.node_i.uid,
                            elm.node_j.uid,
                            elm.uid,
                            elm.uid)
        else:
            ops.element('elasticBeamColumn', elm.uid,
                        elm.node_i.uid,
                        elm.node_j.uid,
                        1.00, common.STIFF, common.STIFF,
                        1.00, 1.00, 1.00,
                        elm.uid)

    def _define_node_constraints(self):
        for lvl in self.building.levels.registry.values():
            if lvl.parent_node:
                ops.rigidDiaphragm(
                    3,
                    lvl.parent_node.uid,
                    *[nd.uid
                      for nd in lvl.list_of_primary_nodes()])

    def _to_OpenSees_domain(self):
        """
        Defines the building model in OpenSeesPy
        """

        logging.info('Defining model in OpenSees')

        def define_node(nd, defined_nodes):
            if nd.uid not in defined_nodes:
                self._define_node(nd)
                defined_nodes.append(nd.uid)

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
            if (elm.section.uid not in defined_sections) and \
               (elm.model_as['type'] != 'elastic'):
                sec = elm.section
                # we don't define utility sections in OpenSees
                # (we instead use the appropriate elements)
                if sec.sec_type != 'utility':
                    # define material
                    mat = sec.material
                    if mat.uid \
                       not in defined_materials:
                        self._define_material(mat)
                        defined_materials.append(mat.uid)
                    if elm.model_as['type'] == 'elastic':
                        self._define_elastic_section(sec)
                    elif elm.model_as['type'] == 'fiber':
                        n_x = elm.model_as['n_x']
                        n_y = elm.model_as['n_y']
                        self._define_fiber_section(sec, n_x, n_y)
                    else:
                        raise ValueError("Invalid modeling type")
                    defined_sections.append(elm.section.uid)

            self._define_line_element(elm)

        # define zerolength elements representing end releases
        for elm in self.building.list_of_endreleases():

            define_node(elm.node_i, defined_nodes)
            define_node(elm.node_j, defined_nodes)

            # define materials
            mats = elm.materials
            for mat in mats.values():
                if mat.uid not in defined_materials:
                    self._define_material(mat)
                    defined_materials.append(mat.uid)

            dofs = elm.materials.keys()
            mat_tags = [mat.uid for mat in elm.materials.values()]

            # define the ZeroLength element
            ops.element('zeroLength', elm.uid,
                        elm.node_i.uid,
                        elm.node_j.uid,
                        '-mat',
                        *mat_tags,
                        '-dir',
                        *dofs,
                        '-orient',
                        *elm.x_vec,
                        *elm.y_vec)

        # define parent nodes
        for nd in self.building.list_of_parent_nodes():
            define_node(nd, defined_nodes)

        # define constraints
        self._define_node_constraints()

    def _define_dead_load(self):
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        for elm in self.building.list_of_line_elements():
            ops.eleLoad('-ele', elm.uid,
                        '-type', '-beamUniform',
                        elm.udl_total()[1],
                        elm.udl_total()[2],
                        elm.udl_total()[0])

        for nd in self.building.list_of_all_nodes():
            ops.load(nd.uid, *nd.load_total())

    ####################################################
    # Methods that read back information from OpenSees #
    ####################################################

    def _store_result(self, analysis_result: AnalysisResult,
                      uid: int, result: list):
        if str(uid) in analysis_result:
            analysis_result[str(uid)].append(result)
        else:
            analysis_result[str(uid)] = [result]

    def _read_node_displacements(self):
        node_list = self.building.list_of_all_nodes()
        for nd in node_list:
            if self.specific_nodes:
                if nd.uid not in self.specific_nodes:
                    continue
            self._store_result(self.node_displacements,
                               str(nd.uid),
                               ops.nodeDisp(nd.uid))

    def _read_node_velocities(self):
        node_list = self.building.list_of_all_nodes()
        for nd in node_list:
            if self.specific_nodes:
                if nd.uid not in self.specific_nodes:
                    continue
            self._store_result(self.node_velocities,
                               str(nd.uid),
                               ops.nodeVel(nd.uid))

    def _read_node_accelerations(self):
        node_list = self.building.list_of_all_nodes()
        for nd in node_list:
            if self.specific_nodes:
                if nd.uid not in self.specific_nodes:
                    continue
            self._store_result(self.node_accelerations,
                               str(nd.uid),
                               ops.nodeAccel(nd.uid))

    def _read_node_reactions(self):
        ops.reactions()
        for nd in self.building.list_of_primary_nodes():
            if nd.restraint_type != 'free':
                uid = nd.uid
                local_reaction = np.array(ops.nodeReaction(uid))
                self._store_result(self.node_reactions,
                                   str(uid),
                                   local_reaction)

    def _read_frame_element_forces(self):
        for elm in self.building.list_of_line_elements():
            uid = elm.uid
            forces = np.array(ops.eleForce(uid))
            self._store_result(self.element_forces,
                               str(uid),
                               forces)

    def _read_frame_fiber_stress_strain(self):
        for elm in self.building.list_of_line_elements():
            if elm.model_as['type'] != 'fiber':
                continue
            uid = elm.uid
            mat_id = elm.section.material.uid
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
            self._store_result(self.fiber_stress_strain, str(uid), result)

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
                        release.uid, 'force')[3:6]
                    # note: moment_j is the opposite of force_i by equilibrium
                    # no need to store it too
                    # rotation_global = ops.eleResponse(
                    #     release.uid, 'deformation')[3:6]
                    rot_i_global = ops.nodeDisp(release.node_i.uid)[3:6]
                    rot_j_global = ops.nodeDisp(release.node_j.uid)[3:6]
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
                        str(release.uid),
                        [moment_i[2], deformation[2]])


    def _read_OpenSees_results(self):
        """
        Reads back the results from the OpenSeesPy domain
        Args:
            data_retention
            Can be either 'default', or 'lightweight'.
            - default stores everything implemented here
              caution: can fill up RAM
            - lightweight only stores node displ, vel and acc.
        """
        self._read_node_displacements()
        self._read_node_velocities()
        self._read_node_accelerations()
        if self.store_reactions:
            self._read_node_reactions()
        if self.store_forces:
            self._read_frame_element_forces()
        if self.store_fiber:
            self._read_frame_fiber_stress_strain()
        if self.store_release_force_defo:
            self._read_release_moment_rot()

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
            for nd in lvl.list_of_primary_nodes():
                if nd.restraint_type != 'free':
                    uid = nd.uid
                    x = nd.coords[0]
                    y = nd.coords[1]
                    z = nd.coords[2]
                    local_reaction = self.node_reactions[str(uid)][step]
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
        ops.system(LN_ANALYSIS_SYSTEM)
        ops.numberer(NUMBERER)
        ops.constraints(*CONSTRAINTS)
        ops.test('NormDispIncr', 1.0e-8, 20, 3)
        ops.algorithm('Newton')
        ops.integrator('LoadControl', 1.0)
        ops.analysis('Static')
        ops.analyze(1)
        logging.info('Analysis Finished')


@dataclass
class LinearGravityAnalysis(LinearAnalysis):
    def run(self):
        self._to_OpenSees_domain()
        self._define_dead_load()
        self._run_gravity_analysis()
        self._read_OpenSees_results()
        self._write_results()


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
        for nd in all_nodes:
            for i in range(self.num_modes):
                self._store_result(
                    self.node_displacements,
                    str(nd.uid),
                    ops.nodeEigenvector(
                        nd.uid,
                        i+1))

    def run(self):
        self._to_OpenSees_domain()
        # tags = ops.getNodeTags()
        # print(len(tags))
        ops.constraints(*CONSTRAINTS)
        ops.system(MD_ANALYSIS_SYSTEM)
        # note: using SparseSYM results in wrong eigendecomposition
        eigValues = np.array(ops.eigen(
            self.num_modes))
        self.periods = 2.00*np.pi / np.sqrt(eigValues)
        self._read_node_displacements()
        self._write_results()

    def modal_participation_factors(self, direction):

        dof_dir = {'x': 0, 'y': 1, 'z': 2}
        ntgs = ops.getNodeTags()
        gammas = np.zeros(self.num_modes)
        mstars = np.zeros(self.num_modes)
        Mn_tot = 0.
        for ntg in ntgs:
            if self.building.retrieve_node(ntg).restraint_type == 'free':
                node_mass = ops.nodeMass(ntg)
                Mn_tot += node_mass[dof_dir[direction]]
        for mode in range(self.num_modes):
            Ln = 0.
            Mn = 0.
            for ntg in ntgs:
                node_mass = ops.nodeMass(ntg)
                node_phi = ops.nodeEigenvector(ntg, mode+1)
                Ln += node_phi[dof_dir[direction]] * \
                    node_mass[dof_dir[direction]]
                for dof in range(6):
                    Mn += (node_phi[dof]**2) * node_mass[dof]
            gammas[mode] = Ln/Mn
            mstars[mode] = Ln**2/Mn
        mstars /= Mn_tot
        return (gammas, mstars)


@dataclass
class NonlinearAnalysis(Analysis):

    n_steps_success: int = field(default=0)

    def _run_gravity_analysis(self):
        print(f'Setting system to {NL_ANALYSIS_SYSTEM}')
        ops.system(NL_ANALYSIS_SYSTEM)
        ops.numberer(NUMBERER)
        ops.constraints(*CONSTRAINTS)
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
            forces_global = self.element_forces[str(elm.uid)][-1][0:3]
            moments_global_ends = self.element_forces[str(elm.uid)][-1][3:6]

            moments_global_clear = \
                transformations.offset_transformation(
                    elm.offset_i, moments_global_ends, forces_global)

            ni, qyi, qzi = T_global2local @ forces_global
            ti, myi, mzi = T_global2local @ moments_global_clear

            wx, wy, wz = elm.udl_total()

            len_clr = elm.length_clear
            t = np.linspace(0.00, len_clr, num=9)

            nx_vec = - t * wx - ni
            mz_vec = t**2 * 0.50 * wy + t * qyi - mzi
            my_vec = t**2 * 0.50 * wz + t * qzi + myi

            prop = elm.section.mesh.geometric_properties()
            area = prop['area']
            iy = prop['inertia']['ixx']
            iz = prop['inertia']['iyy']
            young_mod = elm.section.material.parameters['E0']

            for val in elm.section.snap_points.values():
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
                    str(elm.uid))

    def retrieve_node_displacement(self, uid):
        res = np.zeros((self.n_steps_success, 6))
        for i in range(self.n_steps_success):
            res[i] = self.node_displacements[str(uid)][i]
        return res

    def retrieve_node_acceleration(self, uid):
        res = np.zeros((self.n_steps_success, 6))
        for i in range(self.n_steps_success):
            res[i] = self.node_accelerations[str(uid)][i]
        return res

    def retrieve_node_velocity(self, uid):
        res = np.zeros((self.n_steps_success, 6))
        for i in range(self.n_steps_success):
            res[i] = self.node_velocities[str(uid)][i]
        return res

    def retrieve_node_abs_acceleration(self, uid):
        res = np.zeros((self.n_steps_success, 6))
        for i in range(self.n_steps_success):
            res[i] = self.node_accelerations[str(uid)][i]
        for j in range(3):
            if j in self.a_g:
                a_g = interp1d(
                    self.a_g[j][:, 0], self.a_g[j][:, 1],
                    bounds_error=False, fill_value=0.00)
                res[:, j] += a_g(self.time_vector)*common.G_CONST
        return res

    def retrieve_node_abs_velocity(self, uid):
        res = np.zeros((self.n_steps_success, 6))
        for i in range(self.n_steps_success):
            res[i] = self.node_velocities[str(uid)][i]
        for j in range(3):
            if j in self.a_g:
                a_g = interp1d(
                    self.a_g[j][:, 0], self.a_g[j][:, 1],
                    bounds_error=False, fill_value=0.00)
                v_g = integrate.cumulative_trapezoid(
                    a_g(self.time_vector)*common.G_CONST,
                    self.time_vector, initial=0)
                res[:, j] = res[:, j] + v_g
        return res

    def retrieve_release_force_defo(self, uid):
        force_defo = np.array(self.release_force_defo[str(uid)])
        return force_defo


@dataclass
class PushoverAnalysis(NonlinearAnalysis):

    def _apply_lateral_load(self, direction, modeshape=None, nd=None):
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

        if modeshape is not None:
            if direction not in ['x', 'y']:
                raise ValueError(
                    "Can't apply lateral loads based on the 1st " +
                    "mode shape in the z direction.")
            modeshape_ampl = modeshape / modeshape[-1]
        else:
            modeshape_ampl = np.ones(len(self.building.levels.registry.values()))

        # if a node is given, apply the incremental load on that node
        if nd:
            ops.load(nd.uid, *(1.00*load_dir))
        else:
            for i, lvl in enumerate(self.building.levels.registry.values()):
                # if the level is restrained, no load applied
                if lvl.restraint != 'free':
                    continue
                # if there is a parent node, all load goes there
                if lvl.parent_node:
                    ops.load(lvl.parent_nd.uid,
                             *(distribution[i]*load_dir *
                               modeshape_ampl[i]))
                # if there isn't a parent node, distribute that story's load
                # in proportion to the mass of the nodes
                else:
                    node_list = lvl.nodes_primary.registry.values()
                    masses = np.array([n.mass[0] for n in node_list])
                    masses = masses/np.linalg.norm(masses)
                    for j, nd in enumerate(node_list):
                        ops.load(nd.uid,
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

        print(f'Setting system to {NL_ANALYSIS_SYSTEM}')
        ops.system(NL_ANALYSIS_SYSTEM)
        ops.numberer(NUMBERER)
        ops.constraints(*CONSTRAINTS)
        curr_displ = ops.nodeDisp(control_node.uid, control_DOF+1)
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
                                   control_node.uid, control_DOF + 1,
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
                            control_node.uid, control_DOF+1)
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
        self._write_results()
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
                self.node_displacements[str(nd.uid)][step][control_DOF])
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
        n_i = brace.node_i.uid
        n_j = brace.node_j.uid
        x_axis = brace.x_axis
        x_axis_horiz = np.array((x_axis[0], x_axis[1], 0.00))
        x_axis_horiz = x_axis_horiz / np.linalg.norm(x_axis_horiz)
        for step in range(self.n_steps_success):
            disp_i = self.node_displacements[str(n_i)][step][0:3]
            disp_j = self.node_displacements[str(n_j)][step][0:3]
            diff_disp = np.array(disp_j) - np.array(disp_i)
            disp_prj = np.dot(diff_disp, x_axis)
            drift.append(disp_prj)
            ielm = brace.end_segment_i.internal_elems[-1].uid
            force = self.element_forces[str(ielm)][step][0:3]
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

    # def run(self, analysis_time_increment,
    #         filename_x,
    #         filename_y,
    #         filename_z,
    #         file_time_incr,
    #         finish_time=0.00,
    #         damping_ratio=0.05,
    #         num_modes=None,
    #         printing=True):
    def run(self, analysis_time_increment,
            filename_x,
            filename_y,
            filename_z,
            file_time_incr,
            finish_time=0.00,
            skip_steps=1,
            damping={'type': None},
            printing=True):
        """
        Run the nonlinear time-history analysis
        Args:
            filename_x, y, z: Paths where the fixed-step ground acceleration
                              records are stored (single-column).
            file_time_incr:   The corresponding time increment
            finish_time: Specify a target time (s) to stop the analysis
                         the default value of 0.00 means that it will
                         run for the entire duration of the files.
            damping: Can be any of:
                     {'type': None},
                     {'type': 'rayleigh', 'ratio': r, 'periods': [t1, t2]},
                     {'type': 'modal', 'num_modes': n}
            printing: Controls whether the current time is printed out
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
            self.a_g[0] = np.column_stack((t, gm_vals_x))
            # interp1d(
            #     t, gm_vals_x, bounds_error=False, fill_value=0.00)
        if filename_y:
            self.a_g[1] = np.column_stack((t, gm_vals_y))
            # interp1d(
            #     t, gm_vals_y, bounds_error=False, fill_value=0.00)
        if filename_z:
            self.a_g[2] = np.column_stack((t, gm_vals_z))
            # interp1d(
            #     t, gm_vals_z, bounds_error=False, fill_value=0.00)

        if finish_time == 0.00:
            target_timestamp = duration
        else:
            target_timestamp = finish_time

        logging.info('Defining model in OpenSees')
        self._to_OpenSees_domain()

        # gravity analysis
        logging.info('Defining dead loads')
        self._define_dead_load()
        logging.info('Starting gravity analysis')
        self._run_gravity_analysis()
        self._read_OpenSees_results()
        n_steps_success = 1

        # time-history analysis
        ops.wipeAnalysis()
        ops.loadConst('-time', 0.0)
        curr_time = 0.00
        self.time_vector.append(curr_time)

        damping_type = damping.get('type')

        if damping_type == 'rayleigh':

            logging.info(
            f'Using Rayleigh damping')
            wi = 2 * np.pi / damping['periods'][0]
            zetai = damping['ratio']
            wj = 2 * np.pi / damping['periods'][1]
            zetaj = damping['ratio']
            A = np.array([[1/wi, wi], [1/wj, wj]])
            b = np.array([zetai, zetaj])
            x = np.linalg.solve(A, 2*b)
            ops.numberer(NUMBERER)
            ops.constraints(*CONSTRAINTS)
            print(f'Setting system to {NL_ANALYSIS_SYSTEM}')
            ops.system(NL_ANALYSIS_SYSTEM)
            ops.rayleigh(x[0], 0.0, 0.0, x[1])
            # https://portwooddigital.com/2020/11/08/rayleigh-damping-coefficients/
            # thanks prof. Scott!

        if damping_type == 'modal':
            # tags = ops.getNodeTags()
            # num_nodes = len(tags) - 4
            # num_modeshapes = 3*num_nodes
            # print(len(tags))

            num_modes = damping['num_modes']
            # num_modes = num_modeshapes
            damping_ratio = damping['ratio']
            logging.info(
                'Running eigenvalue analysis'
                f' with {num_modes} modes')
            ops.system(MD_ANALYSIS_SYSTEM)
            ops.eigen(num_modes)
            ops.system(NL_ANALYSIS_SYSTEM)
            # ops.systemSize()
            logging.info('Eigenvalue analysis finished')
            ops.modalDamping(damping['ratio'])
            logging.info(
                f'{damping_ratio*100:.2f}% '
                'modal damping defined')

        logging.info('Starting transient analysis')
        ops.test('NormDispIncr', 1e-6, 50, 0)
        ops.algorithm("ModifiedNewton")
        ops.integrator('TRBDF2')
        ops.analysis("Transient")

        self.define_lateral_load_pattern(
            filename_x,
            filename_y,
            filename_z,
            file_time_incr)

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
                    # provide run speed statistics
                    logging.info(f'Average speed: {speed:.2f} steps/s')
                    print(f'Average speed: {speed:.2f} steps/s')
                if total_step_count % 200 == 0:
                    # dump results to disk if applicable
                    self._sync_results()

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
                    curr_time = ops.getTime()
                    if num_times != 0:
                        num_times -= 1
                    if total_step_count % skip_steps == 0:
                        self._read_OpenSees_results()
                        self.time_vector.append(curr_time)
                        n_steps_success += 1
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
        self._write_results()
        return metadata

    def define_lateral_load_pattern(
            self,
            filename_x,
            filename_y,
            filename_z,
            file_time_incr):

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

    def plot_ground_motion(self, filename, file_time_incr,
                           gmunit='g',
                           plotly=False):
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
                "Absolute Acceleration", gmunit, ".4f")
        else:
            plt.figure()
            plt.plot(x, y, 'k')
            plt.xlabel('Time (s)')
            plt.ylabel(f'Acceleration ({gmunit})')
            plt.show()

