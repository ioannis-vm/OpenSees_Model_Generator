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

# pylint: disable=no-member


from __future__ import annotations
from typing import Optional
from typing import Any
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
import os
import pickle
import logging
import time
import numpy as np
import numpy.typing as npt
from scipy import integrate  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
import plotly.express as px
import pandas as pd
import openseespy.opensees as ops  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from .load_case import LoadCase
from .mesh import subdivide_polygon
# from .import components
from .model import Model
from .ops.element import ElasticBeamColumn
from .import common
# from .graphics import postprocessing_3d
from .graphics import general_2d
from .import transformations
from .collections import Collection
from .gen.querry import LoadCaseQuerry
if TYPE_CHECKING:
    from .ops.uniaxial_material import UniaxialMaterial

nparr = npt.NDArray[np.float64]


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')

LN_ANALYSIS_SYSTEM = 'SparseSYM'
NL_ANALYSIS_SYSTEM = 'SparseSYM'
MD_ANALYSIS_SYSTEM = 'UmfPack'
CONSTRAINTS = ('Transformation',)
NUMBERER = 'Plain'


def test_uniaxial_material(
        mat: UniaxialMaterial,
        input_deformations: list[float],
        incr: float,
        plot: bool
):
    """
    Generates force-deformation pairs using the given uniaxial material.
    Used for testing and validation.
    """

    forces = [0.00]
    deformations = [0.00]

    # define model
    ops.wipe()
    ops.model('basic', '-ndm', 1, '-ndf', 1)
    ops.node(0, 0.00)
    ops.node(1, 0.00)
    ops.fix(0, 1)
    ops.uniaxial_material(  # type: ignore
        *mat.ops_args())  # type: ignore
    ops.element('zeroLength', 0, 0, 1, '-mat', mat.uid, '-dir', 1)
    # define load pattern
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(1, 1.00)
    # run analysis
    ops.numberer('Plain')
    ops.constraints('Plain')

    curr_defo = ops.nodeDisp(1, 1)
    n_steps_success = 0
    total_fail = False
    num_subdiv = 0
    num_times = 0
    scale = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    steps = [25, 50, 75, 100, 250, 500]
    norm = [1.0e-6] * 5 + [1.0e-2]
    algor = ['Newton']*6
    for target_defo in input_deformations:
        # determine push direction
        if curr_defo < target_defo:
            incr = abs(incr)
            sign = +1.00
        else:
            incr = -abs(incr)
            sign = -1.00
        print('entering loop', target_defo)
        while curr_defo * sign < target_defo * sign:
            # determine increment
            if abs(curr_defo - target_defo) < \
               abs(incr) * scale[num_subdiv]:
                incr_anl = sign * abs(curr_defo - target_defo)
            else:
                incr_anl = incr * scale[num_subdiv]
            ops.test('NormDispIncr', norm[num_subdiv],
                     steps[num_subdiv], 0)
            ops.algorithm(algor[num_subdiv])
            # ops.integrator("ArcLength", 1.00e1, 1.00e-7)
            ops.integrator("DisplacementControl", 1, 1, incr_anl)
            ops.system('FullGeneral')
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
                # can still reduce step size
                num_subdiv += 1
                # how many times to run with reduced step size
                num_times = 10
            else:
                # analysis was successful
                n_steps_success += 1
                if num_times != 0:
                    num_times -= 1
                if num_subdiv != 0:
                    if num_times == 0:
                        num_subdiv -= 1
                        num_times = 10
                ops.reactions()
                curr_defo = ops.nodeDisp(1, 1)
                reaction = -ops.nodeReaction(0)[0]
                forces.append(reaction)
                deformations.append(curr_defo)
            if total_fail:
                break
    print('Number of saved analysis steps:', n_steps_success)
    results: nparr = np.column_stack((deformations, forces))
    dframe = pd.DataFrame(results)
    dframe.columns = ['deformation', 'froce']
    if plot:
        fig = px.line(dframe, x='deformation', y='force', markers=True)
        fig.show()
    return dframe


@dataclass(repr=False)
class Results:
    """
    Results object. Stores analysis results.
    """
    node_displacements: Collection[
        int, dict[int, list[float]]] = field(init=False)
    node_velocities: Collection[
        int, dict[int, list[float]]] = field(init=False)
    node_accelerations: Collection[
        int, dict[int, list[float]]] = field(init=False)
    node_reactions: Collection[
        int, dict[int, list[float]]] = field(init=False)
    element_forces: Collection[
        int, dict[int, nparr]] = field(init=False)
    # fiber_stress_strain: Collection = field(init=False)
    release_force_defo: Collection[
        int, dict[int, list[float]]] = field(init=False)
    periods: Optional[nparr] = field(default=None)
    n_steps_success: Optional[int] = field(default=None)
    metadata: Optional[dict[str, object]] = field(default=None)

    def __post_init__(self):
        self.node_displacements = Collection(self)
        self.node_velocities = Collection(self)
        self.node_accelerations = Collection(self)
        self.node_reactions = Collection(self)
        self.element_forces = Collection(self)
        # self.fiber_stress_strain = Collection(self)
        self.release_force_defo = Collection(self)


@dataclass(repr=False)
class AnalysisStorageSettings:
    """
    Analysis storage settings object.
    Controls what will be stored and how.
    """
    log_file: Optional[str] = field(default=None)
    store_forces: bool = field(default=True)
    store_reactions: bool = field(default=True)
    store_fiber: bool = field(default=True)
    store_release_force_defo: bool = field(default=True)
    specific_nodes: list[int] = field(default_factory=list)
    pickle_results: bool = field(default=False)


@dataclass(repr=False)
class Analysis:
    """
    """
    mdl: Model
    load_cases: dict[str, LoadCase]
    output_directory: Optional[str] = field(default=None)
    silent: bool = field(default=False)
    settings: AnalysisStorageSettings = field(
        default_factory=AnalysisStorageSettings)
    results: dict[str, Results] = field(default_factory=dict)
    logger: Optional[object] = field(default=None)

    def log(self, msg: str):
        """
        Adds a message to the log file
        """
        if self.settings.log_file:
            self.logger.info(msg)  # type: ignore

    def print(self, thing: Any):
        """
        Prints a message to stdout
        """
        if not self.silent:
            print(thing)

    def _init_results(self):

        # initialize logger
        if self.settings.log_file:
            logging.basicConfig(
                filename=self.settings.log_file,
                format='%(asctime)s %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p')
            self.logger = logging.getLogger('OpenSees_Model_Generator')
            self.logger.setLevel(logging.DEBUG)

        # initialize output directory
        if self.output_directory and not os.path.exists(self.output_directory):
            os.makedirs(
                self.output_directory,
                exist_ok=True)

        if self.settings.pickle_results and not self.output_directory:
            raise ValueError('Speficy an output directory for the results.')

        # initialize result collections
        assert isinstance(self.load_cases, dict)
        for case_name in self.load_cases:
            node_uids = []
            if self.settings.specific_nodes:
                node_uids.extend(self.settings.specific_nodes)
            else:
                node_uids.extend(nd.uid for nd in self.mdl.list_of_all_nodes())
                node_uids.extend(
                    [n.uid for n in self.load_cases[case_name]
                     .parent_nodes.values()])
            self.results[case_name] = Results()
            for uid in node_uids:
                self.results[case_name].node_displacements[uid] = {}
                self.results[case_name].node_velocities[uid] = {}
                self.results[case_name].node_accelerations[uid] = {}
                self.results[case_name].node_reactions[uid] = {}

            if self.settings.store_forces:
                for uid in self.mdl.dict_of_elastic_beamcolumn_elements():
                    self.results[case_name].element_forces[uid] = {}
            if self.settings.store_fiber:
                for uid in self.mdl.dict_of_disp_beamcolumn_elements():
                    self.results[case_name].element_forces[uid] = {}
            if self.settings.store_release_force_defo:
                for uid in self.mdl.dict_of_zerolength_elements():
                    self.results[case_name].release_force_defo[uid] = {}

        self.log('Analysis started')

    def _write_results_to_disk(self):
        """
        Pickles the results
        """
        with open(f'{self.output_directory}/main_results.pcl',
                  'wb') as file:
            pickle.dump(self.results, file)

    def read_results_from_disk(self):
        """
        Reads back results from a pickle file
        """
        with open(f'{self.output_directory}/main_results.pcl',
                  'rb') as file:
            self.results = pickle.load(file)

    def _to_opensees_domain(self, case_name):
        """
        Defines the model in OpenSeesPy
        """

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
        parent_nodes = {}
        parent_node_to_lvl = {}
        for lvl_uid, node in self.load_cases[case_name].parent_nodes.items():
            parent_nodes[node.uid] = node
            parent_node_to_lvl[node.uid] = lvl_uid

        all_nodes = {}
        all_nodes.update(primary_nodes)
        all_nodes.update(internal_nodes)
        all_nodes.update(parent_nodes)
        for uid, node in all_nodes.items():
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
        for uid in parent_nodes:
            n_r = [False, False, True, True, True, False]
            ops.fix(uid, *n_r)

        # lumped nodal mass
        for uid, node in all_nodes.items():
            if np.max(np.array(
                    (*self.load_cases[case_name]
                     .node_mass[node.uid]
                     .val,))) < common.EPSILON:
                continue
            ops.mass(node.uid, *self.load_cases[case_name].node_mass
                     [node.uid].val)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # Elastic BeamColumn element definition #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        # keep track of defined elements
        defined_elements = {}

        elms = self.mdl.dict_of_elastic_beamcolumn_elements().values()

        # define line elements
        for elm in elms:
            if elm.visibility.skip_opensees_definition:
                continue
            ops.geomTransf(*elm.geomtransf.ops_args())
            ops.element(*elm.ops_args())

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # Fiber BeamColumn element definition #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        # keep track of defined elements
        defined_sections = {}
        defined_materials = {}

        elms = self.mdl.dict_of_disp_beamcolumn_elements().values()

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

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # ZeroLength element definition #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        elms = self.mdl.list_of_zerolength_elements()

        # define zerolength elements
        for elm in elms:
            for mat in elm.mats:
                if mat.uid not in defined_materials:
                    ops.uniaxialMaterial(
                        *mat.ops_args())
                    defined_materials[mat.uid] = mat
            ops.element(*elm.ops_args())
            defined_elements[elm.uid] = elm

        # node constraints
        for uid in parent_nodes:
            lvl = self.mdl.levels[parent_node_to_lvl[uid]]
            nodes = lvl.nodes.values()
            good_nodes = [n for n in nodes if n.coords[2] == lvl.elevation]
            ops.rigidDiaphragm(
                3, uid, *[nd.uid for nd in good_nodes])

    def _define_loads(self, case_name):
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        for elm in self.mdl.list_of_beamcolumn_elements():
            if elm.visibility.skip_opensees_definition:
                continue
            udl_total = (self.load_cases[case_name]
                         .line_element_udl[elm.uid].val)
            ops.eleLoad('-ele', elm.uid,
                        '-type', '-beamUniform',
                        udl_total[1],
                        udl_total[2],
                        udl_total[0])

        for node in self.mdl.list_of_all_nodes():
            ops.load(node.uid, *self.load_cases[case_name]
                     .node_loads[node.uid].val)

    ####################################################
    # Methods that read back information from OpenSees #
    ####################################################

    def _read_node_displacements(self, case_name, step, nodes):
        for node in nodes:
            if self.settings.specific_nodes:
                if node.uid not in self.settings.specific_nodes:
                    continue
            val = ops.nodeDisp(node.uid)
            self.results[case_name].node_displacements[node.uid][step] = val

    def _read_node_velocities(self, case_name, step, nodes):
        for node in nodes:
            if self.settings.specific_nodes:
                if node.uid not in self.settings.specific_nodes:
                    continue
            val = ops.nodeVel(node.uid)
            self.results[case_name].node_velocities[node.uid][step] = val

    def _read_node_accelerations(self, case_name, step, nodes):
        for node in nodes:
            if self.settings.specific_nodes:
                if node.uid not in self.settings.specific_nodes:
                    continue
            val = ops.nodeAccel(node.uid)
            self.results[case_name].node_accelerations[node.uid][step] = val

    def _read_node_reactions(self, case_name, step, nodes):
        ops.reactions()
        for node in nodes:
            if True in node.restraint:
                val = ops.nodeReaction(node.uid)
                self.results[case_name].node_reactions[node.uid][step] = val

    def _read_frame_element_forces(self, case_name, step, elems):
        for elm in elems:
            uid = elm.uid
            global_values: nparr = np.array(ops.eleForce(uid))
            forces_global = global_values[0:3]
            moments_global_ends = global_values[3:6]
            moments_global_clear = transformations.offset_transformation(
                elm.geomtransf.offset_i, moments_global_ends, forces_global)
            x_vec = elm.geomtransf.x_axis
            y_vec = elm.geomtransf.y_axis
            z_vec = elm.geomtransf.z_axis
            transf_global2local: nparr = np.vstack((x_vec, y_vec, z_vec))
            n_i, qy_i, qz_i = transf_global2local @ forces_global
            t_i, my_i, mz_i = transf_global2local @ moments_global_clear
            forces: nparr = np.array((n_i, qy_i, qz_i, t_i, my_i, mz_i))
            self.results[case_name].element_forces[uid][step] = forces

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

    def _read_release_moment_rot(self, case_name, step, zerolength_elms):
        for release in zerolength_elms:
            # force_global = ops.eleResponse(
            #     release.uid, 'force')[:3]
            moment_global = ops.eleResponse(
                release.uid, 'force')[3:6]
            # disp_local = ops.eleResponse(
            #     release.uid, 'deformation')[:3]
            rot_local = ops.eleResponse(
                release.uid, 'deformation')[3:6]
            # note: j quantities are the opposite of those of
            # i by equilibrium
            # no need to store them too
            # rotation_global = ops.eleResponse(
            #     release.uid, 'deformation')[3:6]
            # convert to the local system
            vec_x = release.vecx
            vec_y = release.vecyp
            vec_z: nparr = np.cross(vec_x, vec_y)
            tmat_g2l = transformations.transformation_matrix(
                vec_x, vec_y, vec_z)
            moment_local = tmat_g2l @ -(np.array(moment_global))
            self.results[case_name].release_force_defo[release.uid][step] = \
                [*rot_local, *moment_local]

    def _read_opensees_results(
            self,
            case_name,
            step,
            nodes,
            elastic_beamcolumn_elements,
            disp_beamcolumn_elements,
            zerolength_elements):
        self._read_node_displacements(case_name, step, nodes)
        self._read_node_velocities(case_name, step, nodes)
        self._read_node_accelerations(case_name, step, nodes)
        if self.settings.store_reactions:
            self._read_node_reactions(case_name, step, nodes)
        if self.settings.store_forces:
            self._read_frame_element_forces(
                case_name,
                step, elastic_beamcolumn_elements)
            self._read_frame_element_forces(
                case_name,
                step, disp_beamcolumn_elements)
        # if self.settings.store_fiber:
        #     self._read_frame_fiber_stress_strain()
        if self.settings.store_release_force_defo:
            self._read_release_moment_rot(
                case_name,
                step, zerolength_elements)

    ##################################
    # Numeric Result Post-processing #
    ##################################

    def global_reactions(self, case_name, step):
        """
        Calculates and returns the global reaction forces.
        """
        reactions = np.full(6, 0.00)
        for lvl in self.mdl.levels.values():
            for node in lvl.nodes.values():
                if True in node.restraint:
                    uid = node.uid
                    x_coord = node.coords[0]
                    y_coord = node.coords[1]
                    z_coord = node.coords[2]
                    local_reaction = \
                        self.results[case_name].node_reactions[uid][step]
                    global_reaction: nparr = np.array([
                        local_reaction[0],
                        local_reaction[1],
                        local_reaction[2],
                        local_reaction[3] + local_reaction[2] * y_coord
                        - local_reaction[1] * z_coord,
                        local_reaction[4] + local_reaction[0] * z_coord
                        - local_reaction[2] * x_coord,
                        local_reaction[5] + local_reaction[1] * x_coord
                        - local_reaction[0] * y_coord
                    ])
                    reactions += global_reaction
        return reactions


@dataclass
class StaticAnalysis(Analysis):
    """
    Static analysis.  Stores all results (for each load case) in one
    single step.
    """
    def run(self):
        """
        Runs the static analysis.
        """
        self._init_results()
        for case_name in self.load_cases:
            self._to_opensees_domain(case_name)
            self._define_loads(case_name)
            nodes = self.mdl.list_of_all_nodes()
            nodes.extend(self.load_cases[case_name].parent_nodes.values())
            elastic_elems = [
                elm for elm in self.mdl.list_of_elastic_beamcolumn_elements()
                if not elm.visibility.skip_opensees_definition]
            disp_elems = [
                elm for elm in self.mdl.list_of_disp_beamcolumn_elements()
                if not elm.visibility.skip_opensees_definition]
            zerolength_elems = self.mdl.list_of_zerolength_elements()
            step = 0
            ops.system(LN_ANALYSIS_SYSTEM)
            ops.numberer(NUMBERER)
            ops.constraints(*CONSTRAINTS)
            ops.test('NormDispIncr', 1.0e-8, 20, 3)
            ops.algorithm('Newton')
            ops.integrator('LoadControl', 1.0)
            ops.analysis('Static')
            ops.analyze(1)
            self._read_opensees_results(
                case_name,
                step,
                nodes,
                elastic_elems,
                disp_elems,
                zerolength_elems)
            if self.settings.pickle_results:
                self._write_results_to_disk()


@dataclass
class ModalAnalysis(Analysis):
    """
    Runs a modal analysis.
    """
    num_modes: int = field(default=1, repr=False)

    ####################################################
    # Methods that read back information from OpenSees #
    ####################################################

    def _read_node_displacements_modal(self, case_name):
        nodes = self.mdl.list_of_all_nodes()
        nodes.extend(self.load_cases[case_name].parent_nodes.values())
        for node in nodes:
            for i in range(self.num_modes):
                val = ops.nodeEigenvector(
                        node.uid,
                        i+1)
                self.results[case_name].node_displacements[node.uid][i] = val

    def _read_frame_element_forces_modal(self, case_name, elems):
        # note: opensees does not output the element forces that correspond
        # to the displacement field of each obtained mode.
        # to overcome this, we run a separate analysis imposing those
        # displacements to get the element forces.

        # note: work in progress. currently this doesn't fully work

        for case_name in self.load_cases:
            for step in range(self.num_modes):
                for elm in elems:
                    # if step == 3:
                    #     import pdb
                    #     pdb.set_trace()
                    assert isinstance(elm, ElasticBeamColumn)
                    # displacements at the two ends (global system)
                    u_i = (self.results[case_name].node_displacements
                           [elm.nodes[0].uid][step][0:3])
                    r_i = (self.results[case_name].node_displacements
                           [elm.nodes[0].uid][step][3:6])
                    u_j = (self.results[case_name].node_displacements
                           [elm.nodes[1].uid][step][0:3])
                    r_j = (self.results[case_name].node_displacements
                           [elm.nodes[1].uid][step][3:6])
                    offset_i = elm.geomtransf.offset_i
                    offset_j = elm.geomtransf.offset_j
                    u_i_o = transformations.offset_transformation(
                        offset_i, np.array(u_i), np.array(r_i))
                    u_j_o = transformations.offset_transformation(
                        offset_j, np.array(u_j), np.array(r_j))

                    x_vec = elm.geomtransf.x_axis
                    y_vec = elm.geomtransf.y_axis
                    z_vec: nparr = np.cross(x_vec, y_vec)

                    # global -> local transformation matrix
                    transf_global2local = \
                        transformations.transformation_matrix(
                            x_vec, y_vec, z_vec)
                    u_i_local = transf_global2local @ u_i_o
                    r_i_local = transf_global2local @ r_i
                    u_j_local = transf_global2local @ u_j_o
                    r_j_local = transf_global2local @ r_j

                    # # element UDL
                    udl = (self.load_cases[case_name]
                           .line_element_udl[elm.uid].val)
                    # note: modal analsis doesn't account for applied loads.
                    # this will cause issues with plotting if loads
                    # have been applied.
                    if np.linalg.norm(udl) > common.EPSILON:
                        raise ValueError('Loads applied at modal load case.')

                    # stiffness matrix terms
                    length = elm.clear_length()
                    etimesa = elm.section.e_mod * elm.section.area
                    etimesi_maj = elm.section.e_mod * elm.section.i_x
                    # eimin = elm.section.e_mod * elm.section.Iy
                    gtimesj = elm.section.g_mod * elm.section.j_mod

                    # deformations
                    d_l = u_j_local[0] - u_i_local[0]
                    theta_tor = r_j_local[0] - r_i_local[0]
                    du_xy = u_j_local[1] - u_i_local[1]
                    # du_xz = u_j_local[2] - u_i_local[2]

                    # axial load
                    n_i = etimesa / length * d_l

                    # strong axis bending
                    mz_i = \
                        (4.00 * etimesi_maj / length * r_i_local[2]
                         + 2.00 * etimesi_maj / length * r_j_local[2]
                         - 6.00 * etimesi_maj / length**2 * du_xy)
                    qy_i = (6.00 * etimesi_maj / length**2
                            * (r_i_local[2] + r_j_local[2])
                            - 12.00 * etimesi_maj / length**3 * du_xy)

                    # # weak axis bending
                    # # work in progress...
                    # myi = \
                    #     -(4.00 * eimin / length * (-r_i_local[1])
                    #      + 2.00 * eimin / length * (-r_j_local[1])
                    #      - 6.00 * eimin / length * (du_xz))
                    # qzi = (6.00 * eimin / length**2
                    #        * ((-r_i_local[2]) + (-r_j_local[2]))
                    #        - 12.00 * eimin / length**3 * (du_xz))
                    myi = 0.00
                    qzi = 0.00

                    # torsion
                    t_i = gtimesj / length * theta_tor

                    # store results
                    (self.results[case_name].element_forces
                     [elm.uid][step]) = \
                        np.array((n_i, qy_i, qzi, t_i, myi, mz_i))

    def run(self):
        """
        Runs the modal analysis.
        """
        self._init_results()
        for case_name in self.load_cases:
            self._to_opensees_domain(case_name)
            # tags = ops.getNodeTags()
            # self.print(len(tags))
            ops.constraints(*CONSTRAINTS)
            ops.system(MD_ANALYSIS_SYSTEM)
            # note: using SparseSYM results in wrong eigendecomposition
            num_inertial_nodes = 0
            ndtags = ops.getNodeTags()
            for node in ndtags:
                for j in range(6):
                    if ops.nodeMass(node, j+1) > 0.00:
                        num_inertial_nodes += 1
            eigenvalues: nparr = np.array(ops.eigen(
                self.num_modes))
            self.results[case_name].periods = 2.00*np.pi / np.sqrt(eigenvalues)
            self._read_node_displacements_modal(case_name)
            if self.settings.store_forces:
                self._read_frame_element_forces_modal(
                    case_name, self.mdl.list_of_beamcolumn_elements())
        if self.settings.pickle_results:
            self._write_results_to_disk()

    def modal_participation_factors(self, case_name, direction):
        """
        Calculates modal participation factors
        """
        dof_dir = {'x': 0, 'y': 1, 'z': 2}
        ntgs = ops.getNodeTags()
        gammas = np.zeros(self.num_modes)
        mstars = np.zeros(self.num_modes)
        mn_tot = 0.
        for ntg in ntgs:
            node_mass = self.load_cases[case_name].node_mass[ntg].val
            mn_tot += node_mass[dof_dir[direction]]
        for mode in range(self.num_modes):
            l_n = 0.
            m_n = 0.
            for ntg in ntgs:
                node_mass = self.load_cases[case_name].node_mass[ntg].val
                node_phi = ops.nodeEigenvector(ntg, mode+1)
                l_n += node_phi[dof_dir[direction]] * \
                    node_mass[dof_dir[direction]]
                for dof in range(6):
                    m_n += (node_phi[dof]**2) * node_mass[dof]
            gammas[mode] = l_n/m_n
            mstars[mode] = l_n**2/m_n
        mstars /= mn_tot
        return (gammas, mstars, mn_tot)


@dataclass
class NonlinearAnalysis(Analysis):
    """
    Nonlinear analysis parent class.
    """
    def _run_gravity_analysis(self, system):
        self.print(f'Setting system to {system}')
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
    #         forces_global: nparr = np.array(
    #             self.element_forces[str(elm.uid)][-1][0:3])
    #         moments_global_ends: nparr = np.array(
    #             self.element_forces[str(elm.uid)][-1][3:6])

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

    def retrieve_node_displacement(self, uid, case_name):
        """
        Returns the displacement of a node for all analysis steps
        """
        res = np.zeros((self.results[case_name]
                        .n_steps_success, 6))  # type: ignore
        num = len(self.results[case_name].node_displacements[uid])
        for i in range(num):
            res[i] = self.results[case_name].node_displacements[uid][i]
        dframe = pd.DataFrame(res, columns=['ux', 'uy', 'uz',
                                            'urx', 'ury', 'urz'])
        dframe.index.name = 'step'
        return dframe

    def retrieve_node_acceleration(self, uid, case_name):
        """
        Returns the acceleration of a node for all analysis steps
        """
        res = np.zeros((self.results[case_name]
                        .n_steps_success, 6))  # type: ignore
        num = len(self.results[case_name].node_accelerations[uid])
        for i in range(num):
            res[i] = self.results[case_name].node_accelerations[uid][i]
        dframe = pd.DataFrame(res, columns=['ax', 'ay', 'az',
                                            'arx', 'ary', 'arz'])
        dframe.index.name = 'step'
        return dframe

    def retrieve_node_velocity(self, uid, case_name):
        """
        Returns the velocity of a node for all analysis steps
        """
        res = np.zeros((self.results[case_name].
                        n_steps_success, 6))  # type: ignore
        num = len(self.results[case_name].node_velocities[uid])
        for i in range(num):
            res[i] = self.results[case_name].node_velocities[uid][i]
        dframe = pd.DataFrame(res, columns=['vx', 'vy', 'vz',
                                            'vrx', 'vry', 'vrz'])
        dframe.index.name = 'step'
        return dframe

    def retrieve_node_abs_acceleration(self, uid, case_name):
        """
        Returns the absolute acceleration of a node for all analysis
        steps
        """
        res = np.zeros((self.results[case_name]
                        .n_steps_success, 6))  # type: ignore
        num = len(self.results[case_name].node_accelerations[uid])
        assert isinstance(self, NLTHAnalysis)
        assert self.a_g is not None
        for i in range(num):
            res[i] = self.results[case_name].node_accelerations[uid][i]
        for j in range(3):
            if j in self.a_g:
                a_g = interp1d(
                    self.a_g[j][:, 0], self.a_g[j][:, 1],
                    bounds_error=False, fill_value=0.00)
                res[:, j] += a_g(self.time_vector)*common.G_CONST_IMPERIAL
                # TODO: update to support SI
        dframe = pd.DataFrame(res, columns=['abs ax', 'abs ay', 'abs az',
                                            'abs arx', 'abs ary', 'abs arz'])
        dframe.index.name = 'step'
        return dframe

    def retrieve_node_abs_velocity(self, uid, case_name):
        """
        Returns the absolute velocity of a node for all analysis steps
        """
        res = np.zeros((self.results[case_name]
                        .n_steps_success, 6))  # type: ignore
        num = len(self.results[case_name].node_velocities[uid])
        assert isinstance(self, NLTHAnalysis)
        assert self.a_g is not None
        for i in range(num):
            res[i] = self.results[case_name].node_velocities[uid][i]
        for j in range(3):
            if j in self.a_g:
                a_g = interp1d(
                    self.a_g[j][:, 0], self.a_g[j][:, 1],
                    bounds_error=False, fill_value=0.00)
                v_g = integrate.cumulative_trapezoid(
                    a_g(self.time_vector)*common.G_CONST_IMPERIAL,
                    self.time_vector, initial=0)
                res[:, j] = res[:, j] + v_g
        dfrmae = pd.DataFrame(res, columns=['abs vx', 'abs vy', 'abs vz',
                                            'abs vrx', 'abs vry', 'abs vrz'])
        dfrmae.index.name = 'step'
        return dfrmae

    def retrieve_release_force_defo(self, uid, case_name):
        """
        Returns the force-deformation of a zerolength element for all
        analysis steps
        """
        num = len(self.results[case_name].release_force_defo[uid])
        res = np.zeros((num, 6))
        for i in range(num):
            res[i] = self.results[case_name].release_force_defo[uid][i]
        dframe = pd.DataFrame(res, columns=[
            'u1', 'u2', 'u3', 'q1', 'q2', 'q3'])
        dframe.index.name = 'step'
        return dframe


@dataclass
class PushoverAnalysis(NonlinearAnalysis):
    """
    Pushover analysis
    """
    def _apply_lateral_load(
            self, case_name, direction, modeshape=None, node=None):
        querry = LoadCaseQuerry(self.mdl, self.load_cases[case_name])
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
            modeshape_ampl = np.ones(len(self.mdl.levels.values()))

        # if a node is given, apply the incremental load on that node
        if node:
            ops.load(node.uid, *(1.00*load_dir))
        else:
            for i, lvl in enumerate(self.mdl.levels.values()):
                if lvl.uid == 0:
                    continue
                if self.load_cases[case_name].parent_nodes:
                    node_list = [self.load_cases[case_name]
                                 .parent_nodes[lvl.uid]]
                else:
                    node_list = list(lvl.nodes.values())
                masses: nparr = np.array(
                    [self.load_cases[case_name].node_mass[
                        n.uid].val[0] for n in node_list])
                masses = masses/np.linalg.norm(masses)
                for j, some_node in enumerate(node_list):
                    ops.load(some_node.uid,
                             *(distribution[i]*masses[j]*load_dir *
                               modeshape_ampl[i]))

    def run(self, direction, target_displacements,
            control_node, displ_incr, modeshape=None, loaded_node=None):
        """
        Run pushover analysis
        Arguments:
          direction: can be any of 'x', 'y', 'z'
          target_displacements (list[float]): a list of target displcaments.
            each time a target is reached, the analysis continues until
            the next target is reached, flipping the direction as necessary.
          control_node (Node): analysis control node (of which the
            direciton is querried)
          displ_incr (float): initial displacement increment.
          modeshape (nparr): array containing a mode shape that is
            used to distribute the applied incremental loads. If no
            mode shape is specified, the distribution is uniform.
          loaded_node (Node): if a loaded node is specified, all
            incremental load is applied entirely on that node.
            Otherwise, the incremental loads are distributed to all nodes.
        """
        if direction == 'x':
            control_dof = 0
        elif direction == 'y':
            control_dof = 1
        elif direction == 'z':
            control_dof = 2
        else:
            raise ValueError("Direction can be 'x', 'y' or 'z'")

        self._init_results()

        for case_name in self.load_cases:
            nodes = self.mdl.list_of_all_nodes()
            nodes.extend(self.load_cases[case_name].parent_nodes.values())
            elastic_elems = [
                elm for elm in self.mdl.list_of_elastic_beamcolumn_elements()
                if not elm.visibility.skip_opensees_definition]
            disp_elems = [
                elm for elm in self.mdl.list_of_disp_beamcolumn_elements()
                if not elm.visibility.skip_opensees_definition]
            zerolength_elems = self.mdl.list_of_zerolength_elements()

            self._to_opensees_domain(case_name)
            self._define_loads(case_name)
            self._run_gravity_analysis(NL_ANALYSIS_SYSTEM)

            curr_displ = ops.nodeDisp(control_node.uid, control_dof+1)
            n_steps_success = 0
            self._read_opensees_results(
                case_name,
                n_steps_success,
                nodes,
                elastic_elems,
                disp_elems,
                zerolength_elems)

            ops.wipeAnalysis()
            ops.loadConst('-time', 0.0)
            self._apply_lateral_load(
                case_name, direction, modeshape, loaded_node)
            ops.numberer(NUMBERER)
            ops.constraints(*CONSTRAINTS)

            total_fail = False
            num_subdiv = 0
            num_times = 0

            scale = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
            steps = [25, 50, 75, 100, 250, 500]
            norm = [1.0e-6] * 5 + [1.0e-2]

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

                    self.print(f'entering loop: {target_displacement}')
                    while curr_displ * sign < target_displacement * sign:

                        # determine increment
                        if abs(curr_displ - target_displacement) < \
                           abs(displ_incr) * scale[num_subdiv]:
                            incr = sign * abs(curr_displ - target_displacement)
                        else:
                            incr = displ_incr * scale[num_subdiv]
                        ops.test('NormDispIncr', norm[num_subdiv],
                                 steps[num_subdiv], 0)
                        ops.algorithm('RaphsonNewton')
                        # ops.integrator("ArcLength", 1.00e1, 1.00e-7)
                        ops.integrator("DisplacementControl",
                                       int(control_node.uid), control_dof + 1,
                                       incr)
                        ops.system(NL_ANALYSIS_SYSTEM)
                        ops.analysis("Static")
                        flag = ops.analyze(1)
                        if flag != 0:
                            # analysis failed
                            if num_subdiv == len(scale) - 1:
                                # can't refine further
                                self.print('===========================')
                                self.print('Analysis failed to converge')
                                self.print('===========================')
                                total_fail = True
                                break
                            # can still reduce step size
                            num_subdiv += 1
                            # how many times to run with reduced step size
                            num_times = 10
                        else:
                            # analysis was successful
                            if num_times != 0:
                                num_times -= 1
                            n_steps_success += 1
                            self._read_opensees_results(
                                case_name,
                                n_steps_success,
                                nodes,
                                elastic_elems,
                                disp_elems,
                                zerolength_elems)
                            # self._acceptance_criteria()
                            curr_displ = ops.nodeDisp(
                                int(control_node.uid), control_dof+1)
                            print('Target displacement: '
                                  f'{target_displacement:.2f}'
                                  f' | Current: {curr_displ:.4f}', end='\r')
                            if num_subdiv != 0:
                                if num_times == 0:
                                    num_subdiv -= 1
                                    num_times = 10

            except KeyboardInterrupt:
                self.print("Analysis interrupted")

            n_steps_success += 1
            self._read_opensees_results(
                case_name,
                n_steps_success,
                nodes,
                elastic_elems,
                disp_elems,
                zerolength_elems)
            self.print(f'Number of saved analysis steps: {n_steps_success}')
            metadata = {'successful steps': n_steps_success}
            self.results[case_name].n_steps_success = n_steps_success
            self.results[case_name].metadata = metadata  # type: ignore
        # done with all cases.
        if self.settings.pickle_results:
            self._write_results_to_disk()

    def table_pushover_curve(self, case_name, direction, node):
        """
        Returns the force deformation results
        """
        if direction == 'x':
            control_dof = 0
        elif direction == 'y':
            control_dof = 1
        elif direction == 'z':
            control_dof = 2
        else:
            raise ValueError("Direction can be 'x', 'y' or 'z'")
        base_shear_lst = []
        displacement_lst = []
        for step in range(self.results[case_name]
                          .n_steps_success):  # type:ignore
            base_shear_lst.append(
                self.global_reactions(case_name, step)[control_dof])
            displacement_lst.append(
                self.results[case_name].node_displacements[
                    node.uid][step][control_dof])
        base_shear: nparr = -np.array(base_shear_lst)
        displacement: nparr = np.array(displacement_lst)
        return displacement, base_shear

    def plot_pushover_curve(self, case_name, direction, node):
        """
        Plots the pushover curve
        """
        # TODO: units
        displacement, base_shear = self.table_pushover_curve(
            case_name, direction, node)
        general_2d.line_plot_interactive(
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
    #     for step in range(self.results[case_name].n_steps_success):
    #         disp_i = self.node_displacements[str(n_i)][step][0:3]
    #         disp_j = self.node_displacements[str(n_j)][step][0:3]
    #         diff_disp: nparr = np.array(disp_j) - np.array(disp_i)
    #         disp_prj = np.dot(diff_disp, x_axis)
    #         drift.append(disp_prj)
    #         ielm = brace.end_segment_i.internal_elems[-1].uid
    #         force = self.element_forces[str(ielm)][step][0:3]
    #         force_prj = - np.dot(force, x_axis_horiz)
    #         resisting_force.append(force_prj)
    #     general_2d.line_plot_interactive(
    #         "Brace Resisting Force",
    #         drift, resisting_force, 'line',
    #         'Story drift', 'in', '.0f',
    #         'Resisting Force', 'lb', '.0f')


def define_lateral_load_pattern(
        filename_x,
        filename_y,
        filename_z,
        file_time_incr):
    """
    Defines the load pattern for a time-history analysis from
    previously parsed files with a constant dt
    """

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


def plot_ground_motion(filename, file_time_incr,
                       gmunit='g',
                       plotly=False):
    """
    Plots a ground motion input file.
    """
    y_vals = np.loadtxt(filename)
    n_points = len(y_vals)
    x_vals = np.arange(0.00, n_points * file_time_incr, file_time_incr)
    if plotly:
        general_2d.line_plot_interactive(
            "Ground motion record<br>" +
            filename,
            x_vals, y_vals,
            "line",
            "Time", "s", ".3f",
            "Absolute Acceleration", gmunit, ".4f")
    else:
        plt.figure()
        plt.plot(x_vals, y_vals, 'k')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Acceleration ({gmunit})')
        plt.show()


@dataclass
class NLTHAnalysis(NonlinearAnalysis):
    """
    Dynamic nonlinear time-history analysis
    """
    time_vector: list[float] = field(default_factory=list, repr=False)
    a_g: dict[int, npt.NDArray[np.float64]] = field(
        default_factory=dict, repr=False)

    def run(self,
            analysis_time_increment: float,
            filename_x: str,
            filename_y: str,
            filename_z: str,
            file_time_incr: float,
            finish_time=0.00,
            skip_steps=1,
            damping={'type': None},
            print_progress=True):
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
            print_progress: Controls whether the current time is printed out
        """

        self._init_results()
        self.log('Running NLTH analysis')

        nodes = self.mdl.list_of_all_nodes()
        # note: only runs the first load case provided.
        # nlth should not have load cases.
        # will be fixed in the future.
        case_name = list(self.load_cases.keys())[0]
        nodes.extend(self.load_cases[case_name].parent_nodes.values())
        elastic_elems = [
            elm for elm in self.mdl.list_of_elastic_beamcolumn_elements()
            if not elm.visibility.skip_opensees_definition]
        disp_elems = [
            elm for elm in self.mdl.list_of_disp_beamcolumn_elements()
            if not elm.visibility.skip_opensees_definition]
        zerolength_elems = self.mdl.list_of_zerolength_elements()

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

        self.log(f'filename_x: {filename_x}')
        self.log(f'filename_y: {filename_y}')
        self.log(f'filename_z: {filename_z}')

        num_gm_points = np.min(np.array(nss))
        duration = num_gm_points * file_time_incr

        t_vec = np.linspace(0.00, file_time_incr*num_gm_points, num_gm_points)
        if filename_x:
            self.a_g[0] = np.column_stack((t_vec, gm_vals_x))
        else:
            self.a_g[0] = np.column_stack((t_vec, np.zeros(len(t_vec))))
        if filename_y:
            self.a_g[1] = np.column_stack((t_vec, gm_vals_y))
        else:
            self.a_g[1] = np.column_stack((t_vec, np.zeros(len(t_vec))))
        if filename_z:
            self.a_g[2] = np.column_stack((t_vec, gm_vals_z))
        else:
            self.a_g[2] = np.column_stack((t_vec, np.zeros(len(t_vec))))

        if finish_time == 0.00:
            target_timestamp = duration
        else:
            target_timestamp = finish_time

        self.log('Defining model in OpenSees')
        self._to_opensees_domain(case_name)

        # gravity analysis
        self.log('Defining dead loads')
        self._define_loads(case_name)
        self.log('Starting gravity analysis')
        self._run_gravity_analysis(system)
        n_steps_success = 0
        self._read_opensees_results(
            case_name,
            n_steps_success,
            nodes,
            elastic_elems,
            disp_elems,
            zerolength_elems
        )

        # time-history analysis
        ops.wipeAnalysis()
        ops.loadConst('-time', 0.0)
        curr_time = 0.00
        self.time_vector.append(curr_time)

        if damping_type == 'rayleigh':
            self.log('Using Rayleigh damping')
            w_i = 2 * np.pi / damping['periods'][0]
            zeta_i = damping['ratio']
            w_j = 2 * np.pi / damping['periods'][1]
            zeta_j = damping['ratio']
            a_mat: nparr = np.array([[1/w_i, w_i], [1/w_j, w_j]])
            b_vec: nparr = np.array([zeta_i, zeta_j])
            x_sol: nparr = np.linalg.solve(a_mat, 2*b_vec)
            ops.numberer(NUMBERER)
            ops.constraints(*CONSTRAINTS)
            ops.system(system)
            ops.rayleigh(x_sol[0], 0.0, 0.0, x_sol[1])
            # https://portwooddigital.com/2020/11/08/rayleigh-damping-coefficients/
            # thanks prof. Scott!

        if damping_type == 'modal':
            # tags = ops.getNodeTags()
            # num_nodes = len(tags) - 4
            # num_modeshapes = 3*num_nodes
            # self.print(len(tags))

            num_modes = damping['num_modes']
            # num_modes = num_modeshapes
            damping_ratio = damping['ratio']
            self.log(
                'Running eigenvalue analysis'
                f' with {num_modes} modes')
            ops.numberer(NUMBERER)
            ops.constraints(*CONSTRAINTS)
            ops.system(system)
            ops.eigen(num_modes)
            # ops.system(NL_ANALYSIS_SYSTEM)
            # ops.systemSize()
            self.log('Eigenvalue analysis finished')
            ops.modalDamping(damping['ratio'])
            self.log(
                f'{damping_ratio*100:.2f}% '
                'modal damping defined')

        self.log('Starting transient analysis')
        ops.test('NormDispIncr', 1e-6, 50, 0)
        ops.algorithm("KrylovNewton")
        # ops.integrator('Newmark', 0.50, 0.25)
        ops.integrator('TRBDF2')
        ops.analysis("Transient")

        define_lateral_load_pattern(
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
                # algos = ['NewtonLineSearch',
                #          'ModifiedNewton',
                #          'KrylovNewton',
                #          'SecantNewton',
                #          'RaphsonNewton',
                #          'PeriodicNewton',
                #          'BFGS',
                #          'Broyden']
                # integrs = [('Newmark', 0.50, 0.25),
                #            ('HHT', 1.0),
                #            ('GeneralizedAlpha', 1., 1.),
                #            ('TRBDF2',)]

                # self.print('Testing systems')

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
                # self.print()
                # self.print(times)
                # self.print(best_sys)
                # self.print()

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
                # self.print()
                # self.print(times)
                # self.print(best_algo)
                # self.print()
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
                # self.print()
                # self.print(times)
                # self.print(best_integ)
                # self.print()

                check = ops.analyze(
                    1, analysis_time_increment * scale[num_subdiv])

                # analysis speed stats
                total_step_count += 1
                speed = total_step_count / (time.perf_counter() - now)
                if total_step_count % 50 == 0:
                    # provide run speed statistics
                    # debug
                    print(f'Average speed: {speed:.2f} steps/s')
                    self.log(f'Average speed: {speed:.2f} steps/s')
                if check != 0:
                    # analysis failed
                    if num_subdiv == len(scale) - 1:
                        self.print('===========================')
                        self.print('Analysis failed to converge')
                        self.print('===========================')
                        self.logger.warning(  # type: ignore
                            "Analysis failed"  # type: ignore
                            f" at time {curr_time:.5f}")  # type: ignore
                        analysis_failed = True
                        break
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
                        self._read_opensees_results(
                            case_name,
                            n_steps_success,
                            nodes,
                            elastic_elems,
                            disp_elems,
                            zerolength_elems
                        )
                        self.time_vector.append(curr_time)
                    if print_progress:
                        print('Target timestamp: '
                              f'{target_timestamp:.2f} s '
                              f'| Current: {curr_time:.4f} s',
                              end='\r')
                    if num_subdiv != 0:
                        if num_times == 0:
                            num_subdiv -= 1
                            num_times = 10

        except KeyboardInterrupt:
            self.print("Analysis interrupted")
            self.logger.warning("Analysis interrupted")  # type: ignore

        metadata = {'successful steps': n_steps_success,
                    'analysis_finished_successfully': not analysis_failed}
        self.results[case_name].n_steps_success = len(self.time_vector)
        self.log('Analysis finished')
        if self.settings.pickle_results:
            self._write_results_to_disk()
        return metadata

    def plot_node_displacement_history(self, case_name,
                                       node, direction,
                                       plotly=False):
        """
        Plots the displacement history of the specified node.
        """
        time_vec = self.time_vector
        uid = node.uid
        results = []
        for k in range(self.results[case_name].n_steps_success):  # type:ignore
            results.append(self.results[case_name].node_displacements[
                uid][k][direction])
        vals: nparr = np.array(results)
        if plotly:
            general_2d.line_plot_interactive(
                f"Node {uid} displacement history",
                time_vec, vals,
                "line",
                "Time", "s", ".3f",
                "Rel. Displacement", 'in', ".4f")
        else:
            plt.figure()
            plt.plot(time_vec, vals, 'k')
            plt.xlabel('Time (s)')
            plt.ylabel('Displacement (in)')
            plt.show()


@dataclass
class ModalResponseSpectrumAnalysis:
    """
    Modal response spectrum analysis
    """
    mdl: Model
    load_case: LoadCase
    num_modes: int
    periods: nparr
    spectrum: nparr
    direction: str
    modal_q0: Optional[nparr] = field(default=None)
    vb_modal: Optional[nparr] = field(default=None)
    anl: Optional[Analysis] = field(default=None)

    def run(self):
        """
        Run the modal response spectrum analysis
        """
        spectrum_ifun = interp1d(
            self.periods, self.spectrum, kind='linear')
        anl = ModalAnalysis(
            self.mdl, {self.load_case.name: self.load_case},
            num_modes=self.num_modes)
        anl.settings.pickle_results = False
        anl.settings.store_fiber = False
        anl.settings.store_forces = True
        anl.settings.store_reactions = False
        anl.run()
        case_name = self.load_case.name
        gammas, mstars, mtot = anl.modal_participation_factors(
            case_name, self.direction)
        periods = anl.results[case_name].periods
        if self.mdl.settings.imperial_units:
            g_const = common.G_CONST_IMPERIAL
        else:
            g_const = common.G_CONST_SI
        vb_modal = np.zeros(self.num_modes)
        modal_q0 = np.zeros(self.num_modes)
        for i in range(self.num_modes):
            vb_modal[i] = ((spectrum_ifun(periods[i]))  # type: ignore
                           * mstars[i]  # type: ignore
                           * mtot * g_const)  # type: ignore
            modal_q0[i] = (
                gammas[i] * (spectrum_ifun(periods[i])  # type: ignore
                             / (2.*np.pi / periods[i])**2  # type: ignore
                             * g_const))  # type: ignore
        self.modal_q0 = modal_q0
        self.vb_modal = vb_modal
        self.anl = anl

    def combined_node_disp(self, node_uid):
        """
        Returns the SRSS-combined node displacement of a node
        """
        all_vals = []
        assert self.anl is not None
        for i in range(self.num_modes):
            vals = (np.array(  # type: ignore
                self.anl.results[self.load_case.name]  # type: ignore
                .node_displacements[node_uid][i])  # type: ignore
                    * self.modal_q0[i])  # type: ignore
            all_vals.append(vals)
        all_vals_np: nparr = np.column_stack(all_vals)
        return np.sqrt(np.sum(all_vals_np**2, axis=1))

    def combined_node_disp_diff(self, node_i_uid, node_j_uid):
        """
        Returns the SRSS-combined displacement difference between two
        nodes
        """
        all_vals = []
        assert self.anl is not None
        for i in range(self.num_modes):
            vals_i = (np.array(  # type: ignore
                self.anl.results[self.load_case.name]  # type: ignore
                .node_displacements[node_i_uid][i])  # type: ignore
                      * self.modal_q0[i])  # type: ignore
            vals_j = (np.array(  # type: ignore
                self.anl.results[self.load_case.name]  # type: ignore
                .node_displacements[node_j_uid][i])  # type: ignore
                      * self.modal_q0[i])  # type: ignore
            vals = vals_i - vals_j
            all_vals.append(vals)
        all_vals_np: nparr = np.column_stack(all_vals)
        return np.sqrt(np.sum(all_vals_np**2, axis=1))

    def combined_basic_forces(self, element_uid):
        """
        Returns the SRSS-combined basic forces of a line element
        """
        all_vals = []
        assert self.anl is not None
        for i in range(self.num_modes):
            vals = (np.array(  # type: ignore
                self.anl.results[self.load_case.name]  # type: ignore
                .element_forces[element_uid][i])  # type: ignore
                    * self.modal_q0[i])  # type: ignore
            all_vals.append(vals)
        all_vals_np: nparr = np.column_stack(all_vals)
        return np.sqrt(np.sum(all_vals_np**2, axis=1))
