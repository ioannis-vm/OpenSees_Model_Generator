"""
Model Generator for OpenSees ~ solver
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from typing import Any
from dataclasses import dataclass, field
import os
import shelve
import pickle
import logging
import numpy as np
import numpy.typing as npt
from scipy import integrate  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
import openseespy.opensees as ops  # type: ignore
from datetime import datetime
import time
import matplotlib.pyplot as plt  # type: ignore
from .load_case import LoadCase
from .mesh import subdivide_polygon
# from .import components
from .model import Model
# from .components import elasticBeamColumnElement
from .import common
# from .graphics import postprocessing_3D
from .graphics import general_2D
from .import transformations
from .collections import Collection
from .gen.querry import LoadCaseQuerry
if TYPE_CHECKING:
    from section import Section

nparr = npt.NDArray[np.float64]


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')

LN_ANALYSIS_SYSTEM = 'SparseSYM'
NL_ANALYSIS_SYSTEM = 'SparseSYM'
MD_ANALYSIS_SYSTEM = 'UmfPack'
CONSTRAINTS = ('Transformation',)
NUMBERER = 'Plain'



@dataclass(repr=False)
class Results:
    node_displacements: Collection = field(init=False)
    node_velocities: Collection = field(init=False)
    node_accelerations: Collection = field(init=False)
    node_reactions: Collection = field(init=False)
    element_forces: Collection = field(init=False)
    fiber_stress_strain: Collection = field(init=False)
    release_force_defo: Collection = field(init=False)

    def __post_init__(self):
        self.node_displacements = Collection(self)
        self.node_velocities = Collection(self)
        self.node_accelerations = Collection(self)
        self.node_reactions = Collection(self)
        self.element_forces = Collection(self)
        self.fiber_stress_strain = Collection(self)
        self.release_force_defo = Collection(self)


@dataclass(repr=False)
class AnalysisStorageSettings:
    """
    """
    store_forces: bool = field(default=True)
    store_reactions: bool = field(default=True)
    store_fiber: bool = field(default=True)
    store_release_force_defo: bool = field(default=True)
    specific_nodes: list[str] = field(default_factory=list)
    specific_elements: list[str] = field(default_factory=list)


@dataclass(repr=False)
class Analysis:
    """
    todo - work in progress
    """
    mdl: Model
    load_case: LoadCase
    log_file: Optional[str] = field(default=None)
    output_directory: Optional[str] = field(default=None)
    settings: AnalysisStorageSettings = field(default_factory=AnalysisStorageSettings)
    results: Results = field(default_factory=Results)

    def __post_init__(self):
        logging.basicConfig(
            filename=self.log_file,
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p')
        self.logger = logging.getLogger('OpenSees_Model_Generator')
        self.logger.setLevel(logging.DEBUG)

        if self.output_directory and not os.path.exists(self.output_directory):
            os.makedirs(
                self.output_directory,
                exist_ok=True)

        self.logger.info('Analysis started')
        if not self.output_directory:
            curr = datetime.now()
            self.output_directory = curr.strftime(
                '/tmp/analysis_%Y-%m-%d_%I-%M-%S_%p')
            self.logger.info(
                'Using default output directory: '
                f'{self.output_directory}')
            if not os.path.exists(self.output_directory):
                os.makedirs(
                    self.output_directory,
                    exist_ok=True)

        # initialize result collections
        for uid in self.mdl.dict_of_all_nodes():
            self.results.node_displacements.registry[uid] = {}
            self.results.node_velocities.registry[uid] = {}
            self.results.node_accelerations.registry[uid] = {}
            self.results.node_reactions.registry[uid] = {}
        for uid in self.mdl.dict_of_elastic_beamcolumn_elements():
            self.results.element_forces.registry[uid] = {}
        for uid in self.mdl.dict_of_force_beamcolumn_elements():
            self.results.element_forces.registry[uid] = {}

    def _write_results_to_disk(self):
        self.logger.info('Storing main analysis results')
        self.logger.info(f'  Path:{self.output_directory}/main_results.pcl')
        with open(f'{self.output_directory}/main_results.pcl',
                  'wb') as file:
            pickle.dump(self.results, file)
        self.logger.info('Main analysis results stored successfully')

    def read_results_from_disk(self):
        self.logger.info('Reading analysis results')
        self.logger.info(f'  Path:{self.output_directory}/main_results.pcl')
        with open(f'{self.output_directory}/main_results.pcl',
                  'rb') as file:
            self.results = pickle.load(file)
        self.logger.info('Main analysis results read successfully')















































        
    # def _define_oal(self, material: components.Material):
    #     if material.ops_material == 'Steel01':
    #         ops.uniaxialMaterial(
    #             'Steel01',
    #             int(material.uid),
    #             material.parameters['Fy'],
    #             material.parameters['E0'],
    #             material.parameters['b'])
    #     elif material.ops_material == 'Steel02':
    #         ops.uniaxialMaterial(
    #             'Steel02',
    #             int(material.uid),
    #             material.parameters['Fy'],
    #             material.parameters['E0'],
    #             material.parameters['b'],
    #             *material.parameters['params'],
    #             material.parameters['a1'],
    #             material.parameters['a2'],
    #             material.parameters['a3'],
    #             material.parameters['a4'],
    #             material.parameters['sigInit'])
    #     elif material.ops_material == 'UVCuniaxial':
    #         ops.uniaxialMaterial(
    #             'UVCuniaxial',
    #             int(material.uid),
    #             material.parameters['E0'],
    #             material.parameters['Fy'],
    #             *material.parameters['params']
    #         )
    #     elif material.ops_material == 'Elastic':
    #         ops.uniaxialMaterial(
    #             'Elastic',
    #             int(material.uid),
    #             material.parameters['E']
    #         )
    #     elif material.ops_material == 'ElasticPP':
    #         ops.uniaxialMaterial(
    #             'ElasticPP',
    #             int(material.uid),
    #             material.parameters['E0'],
    #             material.parameters['ey']
    #         )
    #     elif material.ops_material == 'Hysteretic':
    #         ops.uniaxialMaterial(
    #             'Hysteretic',
    #             int(material.uid),
    #             material.parameters['M1y'],
    #             material.parameters['gamma1_y'],
    #             material.parameters['M2y'],
    #             material.parameters['gamma2_y'],
    #             material.parameters['M3y'],
    #             material.parameters['gamma3_y'],
    #             - material.parameters['M1y'],
    #             - material.parameters['gamma1_y'],
    #             - material.parameters['M2y'],
    #             - material.parameters['gamma2_y'],
    #             - material.parameters['M3y'],
    #             - material.parameters['gamma3_y'],
    #             material.parameters['pinchX'],
    #             material.parameters['pinchY'],
    #             material.parameters['damage1'],
    #             material.parameters['damage2'],
    #             material.parameters['beta']
    #         )
    #     elif material.ops_material == 'Bilin':

    #         ops.uniaxialMaterial(
    #             'Bilin',
    #             int(material.uid),
    #             material.parameters['initial_stiffness'],
    #             material.parameters['b+'],
    #             material.parameters['b-'],
    #             material.parameters['my+'],
    #             material.parameters['my-'],
    #             material.parameters['lamda'],
    #             material.parameters['lamda'],
    #             material.parameters['lamda'],
    #             material.parameters['lamda'],
    #             1.00, 1.00, 1.00, 1.00,
    #             material.parameters['theta_p+'],
    #             material.parameters['theta_p-'],
    #             material.parameters['theta_pc+'],
    #             material.parameters['theta_pc-'],
    #             material.parameters['residual_plus'],
    #             material.parameters['residual_minus'],
    #             material.parameters['theta_u'],
    #             material.parameters['theta_u'],
    #             material.parameters['d+'],
    #             material.parameters['d-']
    #         )

    #     elif material.ops_material == 'Pinching4':

    #         ops.uniaxialMaterial(
    #             'Pinching4',
    #             int(material.uid),
    #             *material.parameters.values())

    #     else:
    #         raise ValueError("Unsupported material:" + material.ops_material)










































    






















#     def _define_elastic_section(self, sec: Section):

#         # # using AISC database properties
#         # # RBS sections won't work
#         # ops.section('Elastic',
#         #             int(sec.uid),
#         #             sec.material.parameters['E0'],
#         #             sec.properties['A'],
#         #             sec.properties['Ix'],
#         #             sec.properties['Iy'],
#         #             sec.material.parameters['G'],
#         #             sec.properties['J'])

#         # using mesh properties
#         assert sec.mesh
#         assert sec.properties
#         assert sec.material.parameters
#         ops.section('Elastic',
#                     int(sec.uid),
#                     sec.material.parameters['E0'],
#                     sec.mesh.geometric_properties()['area'],
#                     sec.mesh.geometric_properties()['inertia']['ixx'],
#                     sec.mesh.geometric_properties()['inertia']['iyy'],
#                     sec.material.parameters['G'],
#                     sec.properties['J'])

































































#     def _define_node_constraints(self):
#         for lvl in self.mdl.levels.registry.values():
#             if lvl.parent_node:
#                 ops.rigidDiaphragm(
#                     3,
#                     int(lvl.parent_node.uid),
#                     *[int(nd.uid)
#                       for nd in lvl.list_of_primary_nodes()])












    def _to_OpenSees_domain(self):
        """
        Defines the model in OpenSeesPy
        """
        self.logger.info('Defining model in OpenSees')

        # initialize
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        # ~~~~~~~~~~~~~~~ #
        # Node definition #
        # ~~~~~~~~~~~~~~~ #
        
        # keep track of defined nodes
        defined_nodes = {}

        primary_nodes = self.mdl.dict_of_primary_nodes()
        internal_nodes = self.mdl.dict_of_internal_nodes()

        for uid, node in primary_nodes.items():
            if uid in defined_nodes:
                raise KeyError(f'Node already defined: {uid}')
            defined_nodes[uid] = node
            ops.node(node.uid, *node.coords)
        for uid, node in internal_nodes.items():
            if uid in defined_nodes:
                raise KeyError(f'Node already defined: {uid}')
            defined_nodes[uid] = node
            ops.node(node.uid, *node.coords)

        # restraints
        for uid, node in primary_nodes.items():
            ops.fix(node.uid, *node.restraint)
        for uid, node in internal_nodes.items():
            # (this is super unusual, but who knows..)
            ops.fix(node.uid, *node.restraint)
        # for uid, node in self.mdl.dict_of_parent_nodes().items():
        #     n_r = [False, False, True, True, True, False]
        #     ops.fix(node.uid, *n_r)

        # lumped nodal mass
        for uid, node in primary_nodes.items():
            if np.max(np.array(
                    (*self.load_case
                     .node_mass.registry[node.uid]
                     .total(),))) < common.EPSILON:
                continue
            ops.mass(node.uid, *self.load_case.node_mass
                     .registry[node.uid].total())



        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # Elastic BeamColumn element definition #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        # keep track of defined elements
        defined_elements = {}

        elms = self.mdl.dict_of_elastic_beamcolumn_elements().values()

        # define line elements
        for elm in elms:
            ops.geomTransf(*elm.geomtransf.ops_args())
            ops.element(*elm.ops_args())
                           

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # Fiber BeamColumn element definition #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        # keep track of defined elements
        defined_elements = {}
        defined_sections = {}
        defined_materials = {}

        elms = self.mdl.dict_of_force_beamcolumn_elements().values()

        for elm in elms:
            sec = elm.section
            parts = sec.section_parts.values()
            n_x = elm.section.n_x
            n_y = elm.section.n_y
            if sec.uid not in defined_sections:
                ops.section(*sec.ops_args())
                defined_sections[sec.uid] = sec
                for part in parts:
                    mat = part.ops_material
                    if mat.uid not in defined_materials:
                        ops.uniaxialMaterial(
                            *mat.ops_args())
                        defined_materials[mat.uid] = mat
                    pieces = subdivide_polygon(
                        part.outside_shape, part.holes, n_x, n_y)
                    for piece in pieces:
                        area = piece.area
                        z_loc = piece.centroid.x
                        y_loc = piece.centroid.y
                        ops.fiber(y_loc,
                                  z_loc,
                                  area,
                                  part.ops_material.uid)
                    plt.show()
            ops.beamIntegration(*elm.integration.ops_args())
            ops.geomTransf(*elm.geomtransf.ops_args())
            ops.element(*elm.ops_args())
                

                    











































            


        # # define zerolength elements representing end releases
        # for elm in self.mdl.list_of_endreleases():

        #     define_node(elm.node_i, defined_nodes)
        #     define_node(elm.node_j, defined_nodes)

        #     # define materials
        #     mats = elm.materials
        #     for mat in mats.values():
        #         if mat.uid not in defined_materials:
        #             self._define_material(mat)
        #             defined_materials.append(mat.uid)

        #     dofs = elm.materials.keys()
        #     mat_tags = [int(mat.uid) for mat in elm.materials.values()]

        #     # define the ZeroLength element
        #     ops.element('zeroLength', int(elm.uid),
        #                 int(elm.node_i.uid),
        #                 int(elm.node_j.uid),
        #                 '-mat',
        #                 *mat_tags,
        #                 '-dir',
        #                 *dofs,
        #                 '-orient',
        #                 *elm.x_vec,
        #                 *elm.y_vec)

        # # define parent nodes
        # for nd in self.mdl.list_of_parent_nodes():
        #     define_node(nd, defined_nodes)

        # # define constraints
        # self._define_node_constraints()

    def _define_loads(self):
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        for elm in self.mdl.list_of_beamcolumn_elements():
            udl_total = self.load_case.line_element_udl.registry[elm.uid].total()
            ops.eleLoad('-ele', int(elm.uid),
                        '-type', '-beamUniform',
                        udl_total[1],
                        udl_total[2],
                        udl_total[0])

        for nd in self.mdl.list_of_all_nodes():
            ops.load(nd.uid, *self.load_case.node_loads.registry[nd.uid].total())































    ####################################################
    # Methods that read back information from OpenSees #
    ####################################################










    
#     def _store_result(self,
#                       analysis_result: dict[str, Any],
#                       uid: str, result: Any):
#         if uid in analysis_result:
#             analysis_result[uid].append(result)  # type: ignore
#         else:
#             analysis_result[uid] = [result]  # type: ignore

    def _read_node_displacements(self, step, nodes):
        for nd in nodes:
            if self.settings.specific_nodes:
                if nd.uid not in self.settings.specific_nodes:
                    continue
            val = ops.nodeDisp(nd.uid)
            self.results.node_displacements.registry[nd.uid][step] = val

    def _read_node_velocities(self, step, nodes):
        for nd in nodes:
            if self.settings.specific_nodes:
                if nd.uid not in self.settings.specific_nodes:
                    continue
            val = ops.nodeVel(nd.uid)
            self.results.node_velocities.registry[nd.uid][step] = val

    def _read_node_accelerations(self, step, nodes):
        for nd in nodes:
            if self.settings.specific_nodes:
                if nd.uid not in self.settings.specific_nodes:
                    continue
            val = ops.nodeAccel(nd.uid)
            self.results.node_accelerations.registry[nd.uid][step] = val

    def _read_node_reactions(self, step, nodes):
        ops.reactions()
        for nd in nodes:
            if True in nd.restraint:
                val: nparr = np.array(ops.nodeReaction(nd.uid))
                self.results.node_reactions.registry[nd.uid][step] = val

    def _read_frame_element_forces(self, step, elems):
        for elm in elems:
            uid = elm.uid
            forces: nparr = np.array(ops.eleForce(uid))
            self.results.element_forces.registry[uid][step] = forces

#     def _read_frame_fiber_stress_strain(self):
#         for elm in self.mdl.list_of_line_elements():
#             if elm.model_as['type'] != 'fiber':
#                 continue
#             uid = elm.uid
#             mat_id = elm.section.material.uid
#             result = []
#             n_p = elm.n_p
#             pts = elm.section.snap_points
#             for pt in pts.keys():
#                 pt = list(pts.keys())[0]
#                 z_loc = pts[pt][0]
#                 y_loc = pts[pt][1]
#                 stress_strain = []
#                 for i in range(n_p):
#                     stress_strain.append(ops.eleResponse(
#                         int(uid), "section", str(i+1), "-fiber", str(y_loc),
#                         str(z_loc), str(mat_id), "stressStrain"))
#                 result.append(stress_strain)
#             self._store_result(self.fiber_stress_strain, uid, result)

#     def _read_release_moment_rot(self):
#         for release in self.mdl.list_of_endreleases():
#             # don't store data for simple pin releases
#             # WARNING
#             # This code only monitors strong-axis-bending releases
#             # If more releases are implemented in the future, this
#             # will need to be updated
#             if 6 in release.materials:
#                 if release.materials[6].name in ['auto_pinching',
#                                                  'auto__panel_zone_spring',
#                                                  'auto_IMK']:
#                     # force, deformation in the global system
#                     moment_i_global = ops.eleResponse(
#                         int(release.uid), 'force')[3:6]
#                     # note: moment_j is the opposite of force_i by equilibrium
#                     # no need to store it too
#                     # rotation_global = ops.eleResponse(
#                     #     release.uid, 'deformation')[3:6]
#                     rot_i_global = ops.nodeDisp(int(release.node_i.uid))[3:6]
#                     rot_j_global = ops.nodeDisp(int(release.node_j.uid))[3:6]
#                     rotation_global: nparr = np.array(rot_j_global) - \
#                         np.array(rot_i_global)
#                     # convert to the local system
#                     vec_x = release.x_vec
#                     vec_y = release.y_vec
#                     vec_z: nparr = np.cross(vec_x, vec_y)
#                     tmat_g2l = transformations.transformation_matrix(
#                         vec_x, vec_y, vec_z)
#                     moment_i = tmat_g2l @ -(np.array(moment_i_global))
#                     deformation = tmat_g2l @ np.array(rotation_global)
#                     self._store_result(
#                         self.release_force_defo,
#                         str(release.uid),
#                         [moment_i[2], deformation[2]])

    def _read_OpenSees_results(
            self,
            step,
            nodes,
            elastic_beamcolumn_elements,
            force_beamcolumn_elements):
        self._read_node_displacements(step, nodes)
        self._read_node_velocities(step, nodes)
        self._read_node_accelerations(step, nodes)
        if self.settings.store_reactions:
            self._read_node_reactions(step, nodes)
        if self.settings.store_forces:
            self._read_frame_element_forces(
                step, elastic_beamcolumn_elements)
            self._read_frame_element_forces(
                step, force_beamcolumn_elements)
        # if self.settings.store_fiber:
        #     self._read_frame_fiber_stress_strain()
        # if self.settings.store_release_force_defo:
        #     self._read_release_moment_rot()

#     ##################################
#     # Numeric Result Post-processing #
#     ##################################

    def global_reactions(self, step):
        reactions = np.full(6, 0.00)
        for lvl in self.mdl.levels.registry.values():
            for nd in lvl.nodes.registry.values():
                if True in nd.restraint:
                    uid = nd.uid
                    x = nd.coords[0]
                    y = nd.coords[1]
                    z = nd.coords[2]
                    local_reaction = self.results.node_reactions.registry[uid][step]
                    global_reaction: nparr = np.array([
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
class StaticAnalysis(Analysis):
    def run(self):
        self._to_OpenSees_domain()
        self._define_loads()
        nodes = self.mdl.list_of_all_nodes()
        elastic_elems = self.mdl.list_of_elastic_beamcolumn_elements()
        force_elems = self.mdl.list_of_force_beamcolumn_elements()
        step = 0
        ops.system(LN_ANALYSIS_SYSTEM)
        ops.numberer(NUMBERER)
        ops.constraints(*CONSTRAINTS)
        ops.test('NormDispIncr', 1.0e-8, 20, 3)
        ops.algorithm('Newton')
        ops.integrator('LoadControl', 1.0)
        ops.analysis('Static')
        ops.analyze(1)
        self._read_OpenSees_results(
            step,
            nodes,
            elastic_elems,
            force_elems)
        self.logger.info('Analysis Finished')
        self._write_results_to_disk()


@dataclass
class ModalAnalysis(Analysis):
    """
    Runs a modal analysis.
    """
    num_modes: int = field(default=1, repr=False)
    periods: list[float] = field(default_factory=list, repr=False)

    ####################################################
    # Methods that read back information from OpenSees #
    ####################################################

    def _read_node_displacements(self):
        nodes = self.mdl.list_of_all_nodes()
        for nd in nodes:
            for i in range(self.num_modes):
                val = ops.nodeEigenvector(
                        nd.uid,
                        i+1)
                self.results.node_displacements.registry[nd.uid][i] = val

    def run(self):
        self._to_OpenSees_domain()
        # tags = ops.getNodeTags()
        # print(len(tags))
        ops.constraints(*CONSTRAINTS)
        ops.system(MD_ANALYSIS_SYSTEM)
        # note: using SparseSYM results in wrong eigendecomposition
        eigValues: nparr = np.array(ops.eigen(
            self.num_modes))
        self.periods = 2.00*np.pi / np.sqrt(eigValues)
        self._read_node_displacements()
        self._write_results_to_disk()

    def modal_participation_factors(self, direction):

        dof_dir = {'x': 0, 'y': 1, 'z': 2}
        ntgs = ops.getNodeTags()
        gammas = np.zeros(self.num_modes)
        mstars = np.zeros(self.num_modes)
        Mn_tot = 0.
        for ntg in ntgs:
            if self.mdl.retrieve_node(ntg).\
               restraint_type in ['free', 'parent']:
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
        return (gammas, mstars, Mn_tot)


@dataclass
class NonlinearAnalysis(Analysis):

    n_steps_success: int = field(default=0, repr=False)

    def _run_gravity_analysis(self, system):
        print(f'Setting system to {system}')
        ops.system(system)
        ops.numberer(NUMBERER)
        ops.constraints(*CONSTRAINTS)
        ops.test('NormDispIncr', 1.0e-6, 100, 3)
        ops.algorithm('RaphsonNewton')
        ops.integrator('LoadControl', 1)
        ops.analysis('Static')
        check = ops.analyze(1)
        if check != 0:
            raise ValueError('Analysis Failed')

    # def _acceptance_criteria(self):
    #     for elm in self.mdl.list_of_line_elements():

    #         if elm.model_as['type'] != 'elastic':
    #             continue
    #         mat = elm.section.material
    #         if mat.name == 'steel':
    #             capacity_t = mat.parameters['Fy']/mat.parameters['E0']
    #             capacity_c = -capacity_t
    #         else:
    #             raise ValueError('Unsupported material')
    #         strains = []
    #         x_vec = elm.x_axis
    #         y_vec = elm.y_axis
    #         z_vec = elm.z_axis
    #         trans_global2local: nparr = np.vstack((x_vec, y_vec, z_vec))
    #         forces_global: nparr= np.array(self.element_forces[str(elm.uid)][-1][0:3])
    #         moments_global_ends: nparr = np.array(self.element_forces[str(elm.uid)][-1][3:6])

    #         moments_global_clear = \
    #             transformations.offset_transformation(
    #                 elm.offset_i, moments_global_ends, forces_global)

    #         ni, qyi, qzi = trans_global2local @ forces_global
    #         ti, myi, mzi = trans_global2local @ moments_global_clear

    #         wx, wy, wz = elm.udl.total()

    #         len_clr = elm.length_clear
    #         t = np.linspace(0.00, len_clr, num=9)

    #         nx_vec = - t * wx - ni
    #         mz_vec = t**2 * 0.50 * wy + t * qyi - mzi
    #         my_vec = t**2 * 0.50 * wz + t * qzi + myi

    #         prop = elm.section.mesh.geometric_properties()
    #         area = prop['area']
    #         iy = prop['inertia']['ixx']
    #         iz = prop['inertia']['iyy']
    #         young_mod = elm.section.material.parameters['E0']

    #         for val in elm.section.snap_points.values():
    #             z, y = val
    #             stress = nx_vec/area \
    #                 + my_vec/iz * z \
    #                 - mz_vec/iy * y
    #             strain = stress / young_mod
    #             strains.extend(strain)
    #         emax = np.max(np.array(strains))
    #         emin = np.min(np.array(strains))
    #         if ((emax > capacity_t) or (emin < capacity_c)):
    #             raise ValueError(
    #                 "Acceptance criteria failed for element " +
    #                 str(elm.uid))

    def retrieve_node_displacement(self, uid):
        res = np.zeros((self.n_steps_success, 6))
        for i in range(self.n_steps_success):
            res[i] = self.results.node_displacements[uid][i]
        return res

    def retrieve_node_acceleration(self, uid):
        res = np.zeros((self.n_steps_success, 6))
        for i in range(self.n_steps_success):
            res[i] = self.results.node_accelerations[uid][i]
        return res

    def retrieve_node_velocity(self, uid):
        res = np.zeros((self.n_steps_success, 6))
        for i in range(self.n_steps_success):
            res[i] = self.results.node_velocities[uid][i]
        return res

    def retrieve_node_abs_acceleration(self, uid):
        res = np.zeros((self.n_steps_success, 6))
        for i in range(self.n_steps_success):
            res[i] = self.results.node_accelerations[uid][i]
        for j in range(3):
            if j in self.a_g:
                a_g = interp1d(
                    self.a_g[j][:, 0], self.a_g[j][:, 1],
                    bounds_error=False, fill_value=0.00)
                res[:, j] += a_g(self.time_vector)*common.G_CONST
                # todo: update to support SI
        return res

    def retrieve_node_abs_velocity(self, uid):
        res = np.zeros((self.n_steps_success, 6))
        for i in range(self.n_steps_success):
            res[i] = self.results.node_velocities[uid][i]
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

    # def retrieve_release_force_defo(self, uid):
    #     force_defo: nparr = np.array(self.release_force_defo[str(uid)])
    #     return force_defo


@dataclass
class PushoverAnalysis(NonlinearAnalysis):

    def _apply_lateral_load(self, direction, modeshape=None, nd=None):
        querry = LoadCaseQuerry(self.mdl, self.load_case)
        distribution = querry.level_masses()
        distribution = distribution / np.linalg.norm(distribution)

        # define the load pattern
        ops.timeSeries('Linear', 2)
        ops.pattern("Plain", 2, 2)

        if direction == 'x':
            load_dir: nparr = np.array([1., 0., 0., 0., 0., 0.])
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
            modeshape_ampl = np.ones(len(self.mdl.levels.registry.values()))

        # if a node is given, apply the incremental load on that node
        if nd:
            ops.load(int(nd.uid), *(1.00*load_dir))
        else:
            for i, lvl in enumerate(self.mdl.levels.registry.values()):
                # if there is a parent node, all load goes there
                # if lvl.parent_node:
                #     pass
                # #     ops.load(int(lvl.parent_node.uid),
                # #              *(distribution[i]*load_dir *
                # #                modeshape_ampl[i]))
                # # if there isn't a parent node, distribute that story's load
                # # in proportion to the mass of the nodes
                # else:
                node_list = lvl.nodes.registry.values()
                masses: nparr = np.array([self.load_case.node_mass.registry[n.uid].total()[0] for n in node_list])
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

        nodes = self.mdl.list_of_all_nodes()
        elastic_elems = self.mdl.list_of_elastic_beamcolumn_elements()
        force_elems = self.mdl.list_of_force_beamcolumn_elements()

        self._to_OpenSees_domain()
        self._define_loads()
        self._run_gravity_analysis(NL_ANALYSIS_SYSTEM)
        ops.wipeAnalysis()
        ops.loadConst('-time', 0.0)
        self._apply_lateral_load(direction, modeshape, loaded_node)

        ops.numberer(NUMBERER)
        ops.constraints(*CONSTRAINTS)
        curr_displ = ops.nodeDisp(int(control_node.uid), control_DOF+1)
        n_steps_success = 0
        self._read_OpenSees_results(
            n_steps_success,
            nodes,
            elastic_elems,
            force_elems)

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
                                   int(control_node.uid), control_DOF + 1,
                                   incr)
                    ops.system(NL_ANALYSIS_SYSTEM)
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
                        self._read_OpenSees_results(
                            n_steps_success,
                            nodes,
                            elastic_elems,
                            force_elems)
                        # self._acceptance_criteria()
                        curr_displ = ops.nodeDisp(
                            int(control_node.uid), control_DOF+1)
                        print('Target displacement: %.2f | Current: %.4f' %
                              (target_displacement, curr_displ), end='\r')
                        if num_subdiv != 0:
                            if num_times == 0:
                                num_subdiv -= 1
                                num_times = 10

        except KeyboardInterrupt:
            print("Analysis interrupted")

        n_steps_success += 1
        self._read_OpenSees_results(
            n_steps_success,
            nodes,
            elastic_elems,
            force_elems)
        print('Number of saved analysis steps:', n_steps_success)
        metadata = {'successful steps': n_steps_success}
        self.n_steps_success = n_steps_success
        self._write_results_to_disk()
        return metadata

    def table_pushover_curve(self, direction, nd):
        if direction == 'x':
            control_DOF = 0
        elif direction == 'y':
            control_DOF = 1
        elif direction == 'z':
            control_DOF = 2
        else:
            raise ValueError("Direction can be 'x', 'y' or 'z'")
        base_shear_lst = []
        displacement_lst = []
        for step in range(self.n_steps_success):
            base_shear_lst.append(self.global_reactions(step)[control_DOF])
            displacement_lst.append(
                self.results.node_displacements.registry[nd.uid][step][control_DOF])
        base_shear: nparr = -np.array(base_shear_lst)
        displacement: nparr = np.array(displacement_lst)
        return displacement, base_shear

    def plot_pushover_curve(self, direction, nd):
        # todo: units
        displacement, base_shear = self.table_pushover_curve(direction, nd)
        general_2D.line_plot_interactive(
            "Pushover Analysis Results<br>" + "Direction: " + direction,
            displacement, base_shear, 'spline+markers',
            "Displacement", "in", ".0f",
            "Base Shear", "lb", ".0f")

    # def plot_brace_hysteresis(self, brace):
    #     drift = []
    #     resisting_force = []
    #     n_i = brace.node_i.uid
    #     n_j = brace.node_j.uid
    #     x_axis = brace.x_axis
    #     x_axis_horiz: nparr = np.array((x_axis[0], x_axis[1], 0.00))
    #     x_axis_horiz = x_axis_horiz / np.linalg.norm(x_axis_horiz)
    #     for step in range(self.n_steps_success):
    #         disp_i = self.node_displacements[str(n_i)][step][0:3]
    #         disp_j = self.node_displacements[str(n_j)][step][0:3]
    #         diff_disp: nparr = np.array(disp_j) - np.array(disp_i)
    #         disp_prj = np.dot(diff_disp, x_axis)
    #         drift.append(disp_prj)
    #         ielm = brace.end_segment_i.internal_elems[-1].uid
    #         force = self.element_forces[str(ielm)][step][0:3]
    #         force_prj = - np.dot(force, x_axis_horiz)
    #         resisting_force.append(force_prj)
    #     general_2D.line_plot_interactive(
    #         "Brace Resisting Force",
    #         drift, resisting_force, 'line',
    #         'Story drift', 'in', '.0f',
    #         'Resisting Force', 'lb', '.0f')


@dataclass
class NLTHAnalysis(NonlinearAnalysis):

    time_vector: list[float] = field(default_factory=list, repr=False)
    a_g: dict[int, npt.NDArray[np.float64]] = field(default_factory=dict, repr=False)

    def _define_lateral_load_pattern(
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
                           '-factor', common.G_CONST_IMPERIAL)
            # pattern, direction, timeseries tag
            ops.pattern('UniformExcitation', 2, 1, '-accel', 2)

        if filename_y:
            error = False
            # define Y-direction TH
            ops.timeSeries('Path', 3, '-dt', file_time_incr,
                           '-filePath', filename_y,
                           '-factor', common.G_CONST_IMPERIAL)
            # pattern, direction, timeseries tag
            ops.pattern('UniformExcitation', 3, 2, '-accel', 3)

        if filename_z:
            error = False
            # define Z-direction TH
            ops.timeSeries('Path', 4, '-dt', file_time_incr,
                           '-filePath', filename_z,
                           '-factor', common.G_CONST_IMPERIAL)
            # pattern, direction, timeseries tag
            ops.pattern('UniformExcitation', 4, 3, '-accel', 4)

        if error:
            raise ValueError(
                "No input files specified.")

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

        self.logger.info('Running NLTH analysis')

        nodes = self.mdl.list_of_all_nodes()
        elastic_elems = self.mdl.list_of_elastic_beamcolumn_elements()
        force_elems = self.mdl.list_of_force_beamcolumn_elements()

        damping_type = damping.get('type')

        if damping_type == 'rayleigh':
            system = NL_ANALYSIS_SYSTEM
        elif damping_type == 'modal':
            system = MD_ANALYSIS_SYSTEM
        else:
            system = NL_ANALYSIS_SYSTEM

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

        self.logger.info(f'filename_x: {filename_x}')
        self.logger.info(f'filename_y: {filename_y}')
        self.logger.info(f'filename_z: {filename_z}')

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

        self.logger.info('Defining model in OpenSees')
        self._to_OpenSees_domain()

        # gravity analysis
        self.logger.info('Defining dead loads')
        self._define_loads()
        self.logger.info('Starting gravity analysis')
        self._run_gravity_analysis(system)
        n_steps_success = 0
        self._read_OpenSees_results(
            n_steps_success,
            nodes,
            elastic_elems,
            force_elems
        )

        # time-history analysis
        ops.wipeAnalysis()
        ops.loadConst('-time', 0.0)
        curr_time = 0.00
        self.time_vector.append(curr_time)

        if damping_type == 'rayleigh':
            self.logger.info('Using Rayleigh damping')
            wi = 2 * np.pi / damping['periods'][0]
            zetai = damping['ratio']
            wj = 2 * np.pi / damping['periods'][1]
            zetaj = damping['ratio']
            A: nparr = np.array([[1/wi, wi], [1/wj, wj]])
            b: nparr = np.array([zetai, zetaj])
            x: nparr = np.linalg.solve(A, 2*b)
            ops.numberer(NUMBERER)
            ops.constraints(*CONSTRAINTS)
            ops.system(system)
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
            self.logger.info(
                'Running eigenvalue analysis'
                f' with {num_modes} modes')
            ops.numberer(NUMBERER)
            ops.constraints(*CONSTRAINTS)
            ops.system(system)
            ops.eigen(num_modes)
            # ops.system(NL_ANALYSIS_SYSTEM)
            # ops.systemSize()
            self.logger.info('Eigenvalue analysis finished')
            ops.modalDamping(damping['ratio'])
            self.logger.info(
                f'{damping_ratio*100:.2f}% '
                'modal damping defined')

        self.logger.info('Starting transient analysis')
        ops.test('NormDispIncr', 1e-6, 50, 0)
        ops.algorithm("KrylovNewton")
        # ops.integrator('Newmark', 0.50, 0.25)
        ops.integrator('TRBDF2')
        ops.analysis("Transient")

        self._define_lateral_load_pattern(
            filename_x,
            filename_y,
            filename_z,
            file_time_incr)

        num_subdiv = 0
        num_times = 0
        analysis_failed = False

        scale = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]

        total_step_count = 0
        now = time.perf_counter()

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

                #     tt = time.perf_counter()
                #     check = ops.analyze(
                #         20, analysis_time_increment * scale[num_subdiv])
                #     ttt = time.perf_counter()
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
                #     tt = time.perf_counter()
                #     check = ops.analyze(
                #         20, analysis_time_increment * scale[num_subdiv])
                #     ttt = time.perf_counter()
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
                #     tt = time.perf_counter()
                #     check = ops.analyze(
                #         20, analysis_time_increment * scale[num_subdiv])
                #     ttt = time.perf_counter()
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
                speed = total_step_count / (time.perf_counter() - now)
                if total_step_count % 50 == 0:
                    # provide run speed statistics
                    self.logger.info(f'Average speed: {speed:.2f} steps/s')
                if check != 0:
                    # analysis failed
                    if num_subdiv == len(scale) - 1:
                        print()
                        print('===========================')
                        print('Analysis failed to converge')
                        print('===========================')
                        print()
                        self.logger.warning(
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
                        n_steps_success += 1
                        self._read_OpenSees_results(
                            n_steps_success,
                            nodes,
                            elastic_elems,
                            force_elems
                        )
                        self.time_vector.append(curr_time)
                    if printing:
                        print('Target timestamp: %.2f s | Current: %.4f s' %
                              (target_timestamp, curr_time), end='\r')
                    if num_subdiv != 0:
                        if num_times == 0:
                            num_subdiv -= 1
                            num_times = 10

        except KeyboardInterrupt:
            print("Analysis interrupted")
            self.logger.warning("Analysis interrupted")

        metadata = {'successful steps': n_steps_success,
                    'analysis_finished_successfully': not analysis_failed}
        self.n_steps_success = len(self.time_vector)
        self.logger.info('Analysis finished')
        self._write_results_to_disk()
        return metadata

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

    def plot_node_displacement_history(self, node, direction,
                                       plotly=False):
        time = self.time_vector
        uid = node.uid
        results = []
        for k in range(self.n_steps_success):
            results.append(self.results.node_displacements.registry[uid][k][direction])
        vals = np.array(results)
        if plotly:
            general_2D.line_plot_interactive(
                f"Node {uid} displacement history",
                time, vals,
                "line",
                "Time", "s", ".3f",
                "Rel. Displacement", 'in', ".4f")
        else:
            plt.figure()
            plt.plot(time, vals, 'k')
            plt.xlabel('Time (s)')
            plt.ylabel('Displacement (in)')
            plt.show()
