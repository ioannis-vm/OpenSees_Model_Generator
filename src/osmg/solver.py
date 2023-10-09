"""
Defines Analysis objects.

"""

#
#   _|_|      _|_|_|  _|      _|    _|_|_|
# _|    _|  _|        _|_|  _|_|  _|
# _|    _|    _|_|    _|  _|  _|  _|  _|_|
# _|    _|        _|  _|      _|  _|    _|
#   _|_|    _|_|_|    _|      _|    _|_|_|
#
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

# pylint: disable=no-member
# pylint: disable=dangerous-default-value
# pylint: disable=unused-argument
# pylint: disable=multiple-statements
# flake8: noqa: E701  # I like multiple statements on one line, sometimes.
# mypy: disable-error-code="attr-defined"


from __future__ import annotations
from typing import Optional
from typing import Any
from typing import Union
from dataclasses import dataclass, field
import pkgutil
import os
import shutil
import pickle
import logging
import textwrap
from time import perf_counter
from tqdm import tqdm
import dill
import numpy as np
import numpy.typing as npt
from scipy import integrate
from scipy.interpolate import interp1d
import pandas as pd

# import openseespy.opensees as ops
import opensees.openseespy as ops # Use Claudio's package instead

import matplotlib.pyplot as plt
from .load_case import LoadCase
from .model import Model
from .ops import element
from . import common
from .graphics import general_2d
from . import transformations
from .obj_collections import Collection
from .gen.query import LoadCaseQuery
from .ops import uniaxial_material

nparr = npt.NDArray[np.float64]

CONSTRAINTS = ("Transformation",)
NUMBERER = "RCM"


def script_append(tcl_script, lines):
    with open(tcl_script, 'a', encoding='utf-8') as f:
        f.writelines([f'{line}\n' for line in lines])


@dataclass(repr=False)
class Results:
    """
    Stores analysis results.

    Attributes:
      node_displacements: Displacements, stored in a dictionary. The
        keys correspond to the node UIDs. Each value is a dictionary,
        of which the keys are the analysis step. Each value of that is
        list containing the displacement of each DOF.
      node_velocities: Similar to node_displacements.
      node_accelerations: Similar to node_displacements.
      node_reactions: Similar to node displacements.
      element_forces: Basic forces of beamcolumn elements. The nested
        structure is similar to that of node_displacements.
      release_foce_defo: Force-deformation pairs of zerolength
        elements. The nested structure is similar to that of
        node_displacements.
      periods: Optional, stores the periods for modal analyses.
      n_steps_success: Total number of steps of the analysis.
      metadata: Optional metadata that depend on the type of analysis.

    """

    node_displacements: Collection[int, dict[int, list[float]]] = field(
        init=False
    )
    node_velocities: Collection[int, dict[int, list[float]]] = field(
        init=False
    )
    node_accelerations: Collection[int, dict[int, list[float]]] = field(
        init=False
    )
    node_reactions: Collection[int, dict[int, list[float]]] = field(init=False)
    element_forces: Collection[int, dict[int, nparr]] = field(init=False)
    # fiber_stress_strain: Collection = field(init=False)
    release_force_defo: Collection[int, dict[int, list[float]]] = field(
        init=False
    )
    periods: Optional[nparr] = field(default=None)
    n_steps_success: int = field(default=0)
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
class AnalysisSettings:
    """
    Analysis settings object.

    Attributes:
      log_file: If specified, the log messages are written to this
        file.
      silent: If True, no messages are printed to the console.
      store_forces: If True, store the element forces.
      store_reactions: If True, store the reaction forces.
      store_fiber: If True, store fiber section results.
      store_release_force_defo: If True, store the release forces and
        deformations.
      specific_nodes: List of node numbers to store. If empty, all
        nodes are stored.
      pickle_results: If True, the results are stored using pickle.
      solver: The solver to use. The default is 'UmfPack'.
      restrict_dof: An optional list of booleans. If it is provided,
        the corresponding global DOF will be restricted. If all items
        are False no DOFs are restricted, but a very flexible spring
        support will be added to all primary nodes, which can help
        ease convergence issues in cases of rigid body motion modes.

    """

    log_file: Optional[str] = field(default=None)
    silent: bool = field(default=False)
    store_forces: bool = field(default=True)
    store_reactions: bool = field(default=True)
    store_fiber: bool = field(default=True)
    store_release_force_defo: bool = field(default=True)
    specific_nodes: list[int] = field(default_factory=list)
    pickle_results: bool = field(default=False)
    solver: str = field(default='UmfPack')
    restrict_dof: Optional[list[bool]] = field(default=None)


@dataclass(repr=False)
class Warnings:
    """
    Analysis warnings. Helps avoid issuing repeated warnings.
    """

    parent_analysis: Analysis
    issued_warnings: list[str] = field(default_factory=list)

    def issue(self, message: str) -> None:
        """
        Shows unique warning messages.

        Arguments:
          message: Warning message.

        """

        if message not in self.issued_warnings:
            self.parent_analysis.log(f'WARNING: {message}')
            self.issued_warnings.append(message)


@dataclass(repr=False)
class Analysis:
    """
    Parent analysis class.

    Attributes:
      mdl: a given model
      load_cases: Dictionary containing load case names and
        load case objects in which those load cases reside.
      output_directory: Where to place the results
        when it is requested for them to be pickled.
      settings: analysis settings
      results: analysis results
      logger: Logger object
      warning: Warnings object

    """

    mdl: Model
    load_cases: dict[str, LoadCase]
    output_directory: Optional[str] = field(default=None)
    settings: AnalysisSettings = field(
        default_factory=AnalysisSettings
    )
    results: dict[str, Results] = field(default_factory=dict)
    logger: Optional[object] = field(default=None)
    warning: Warnings = field(init=False)

    def __post_init__(self):
        # instantiate a Warnings object
        self.warning = Warnings(self)

    def log(self, msg: str) -> None:
        """
        Adds a message to the log file.

        """

        if self.logger:
            # logger might not have been initialized yet
            self.logger.info(msg)

    def print(self, thing: Any, end: str = '\n') -> None:
        """
        Prints a message to stdout.

        """

        if not self.settings.silent:
            print(thing, end=end)
        if self.logger:
            # logger might not have been initialized yet
            self.log(thing)

    def _init_results(self):

        # initialize output directory
        if self.output_directory and not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

        # initialize logger
        if self.settings.log_file:
            logging.basicConfig(
                filename=self.settings.log_file,
                filemode="w",
                format="%(asctime)s %(message)s",
                datefmt="%m/%d/%Y %I:%M:%S %p",
            )
            self.logger = logging.getLogger("OpenSees_Model_Generator")
            self.logger.setLevel(logging.DEBUG)

        if self.settings.pickle_results and not self.output_directory:
            raise ValueError("Specify an output directory for the results.")

        # initialize result collections
        assert isinstance(self.load_cases, dict)
        for case_name in self.load_cases:
            node_uids = []
            if self.settings.specific_nodes:
                node_uids.extend(self.settings.specific_nodes)
            else:
                node_uids.extend(nd.uid for nd in self.mdl.list_of_all_nodes())
                node_uids.extend(
                    [
                        n.uid
                        for n in self.load_cases[
                            case_name
                        ].parent_nodes.values()
                    ]
                )
            self.results[case_name] = Results()
            for uid in node_uids:
                self.results[case_name].node_displacements[uid] = {}
                self.results[case_name].node_velocities[uid] = {}
                self.results[case_name].node_accelerations[uid] = {}
                self.results[case_name].node_reactions[uid] = {}

            if self.settings.store_forces:
                for uid in self.mdl.dict_of_specific_element(element.ElasticBeamColumn):
                    self.results[case_name].element_forces[uid] = {}
                for uid in self.mdl.dict_of_specific_element(element.DispBeamColumn):
                    self.results[case_name].element_forces[uid] = {}
                for uid in self.mdl.dict_of_specific_element(element.TrussBar):
                    self.results[case_name].element_forces[uid] = {}
            if self.settings.store_fiber:
                for uid in self.mdl.dict_of_specific_element(element.DispBeamColumn):
                    self.results[case_name].element_forces[uid] = {}
            if self.settings.store_release_force_defo:
                for uid in self.mdl.dict_of_specific_element(element.ZeroLength):
                    self.results[case_name].release_force_defo[uid] = {}

        self.log("Analysis started")

    def _write_results_to_disk(self):
        """
        Pickles the results.

        """

        with open(f"{self.output_directory}/main_results.pcl", "wb") as file:
            pickle.dump(self.results, file)

    def read_results_from_disk(self):
        """
        Reads back results from a pickle file.

        """

        with open(f"{self.output_directory}/main_results.pcl", "rb") as file:
            self.results = pickle.load(file)

    def write_tcl_model(self, case_name, tcl_script):
        """
        Generates a tcl script defining the model

        """

        # determine if a file already exists
        if os.path.exists(tcl_script):
            # remove it if it does
            os.remove(tcl_script)


        # initialize
        script_append(tcl_script, [
            'wipe',
            'model basic -ndm 3 -ndf 6'
        ])

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
                raise KeyError(f"Node already defined: {uid}")
            defined_nodes[uid] = node
            script_append(
                tcl_script,
                [f'node {node.uid} '+' '.join(str(x) for x in node.coords)])

        # restraints
        for uid, node in primary_nodes.items():
            script_append(
                tcl_script,
                [
                    f'fix {node.uid} '+' '.join(str(int(x)) for x in node.restraint)
                ])
            
        for uid, node in internal_nodes.items():
            # (this is super unusual, but who knows..)
            script_append(tcl_script,
                [
                f'fix {node.uid} '+' '.join(str(int(x)) for x in node.restraint)
            ])
        for node in parent_nodes.values():
            script_append(tcl_script,
                          [
                f'fix {node.uid} '+' '.join(str(int(x)) for x in node.restraint)
            ])

        # lumped nodal mass
        for uid, node in all_nodes.items():
            # retrieve osmg node mass
            specified_mass = self.load_cases[case_name].node_mass[node.uid].val
            # replace zeros with a small mass
            # (to be able to capture all mode shapes)
            specified_mass[specified_mass == 0] = common.EPSILON
            # verify that all mass values are non-negative
            assert np.size(specified_mass[specified_mass < 0.00]) == 0
            # assign mass to the opensees node
            script_append(tcl_script,
                          [
                f'mass {node.uid} '+' '.join(str(x) for x in specified_mass)
            ])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # Elastic BeamColumn element definition #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        # keep track of defined elements
        defined_elements = {}

        elms = list(self.mdl.dict_of_specific_element(
            element.ElasticBeamColumn).values())

        # define line elements
        for elm in elms:
            if elm.visibility.skip_opensees_definition:
                continue
            script_append(tcl_script, [
                f'geomTransf '+' '.join([str(x) for x in elm.geomtransf.ops_args()]),
                f'element '+' '.join([str(x) for x in elm.ops_args()]),
            ])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # Fiber BeamColumn element definition #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        # keep track of defined elements
        defined_sections: dict[int, object] = {}
        defined_materials: dict[int, object] = {}

        elms = self.mdl.list_of_specific_element(element.DispBeamColumn)

        def define_material(mat, defined_materials):
            """
            A cute recursive function that defines materials with
            predecessors.

            """

            # if the actual material has not been defined yet,
            if mat.uid not in defined_materials:
                while (
                    hasattr(mat, "predecessor")
                    and mat.predecessor.uid not in defined_materials
                ):
                    # define predecessor
                    define_material(mat.predecessor, defined_materials)
                # and also define the actual material
                script_append(tcl_script,[
                    f'uniaxialMaterial '+' '.join([str(x) for x in mat.ops_args()])
                ])
                defined_materials[mat.uid] = mat

        for elm in elms:
            sec = elm.section
            parts = sec.section_parts.values()
            if sec.uid not in defined_sections:
                script_append(tcl_script, [
                    f'section '+' '.join([str(x) for x in sec.ops_args()])+' {'
                ])
                defined_sections[sec.uid] = sec
                for part in parts:
                    mat = part.ops_material
                    define_material(mat, defined_materials)
                    pieces = part.cut_into_tiny_little_pieces()
                    for piece in pieces:
                        area = piece.area
                        z_loc = piece.centroid.x
                        y_loc = piece.centroid.y
                        script_append(tcl_script, [
                            f'    fiber {y_loc} {z_loc} {area} {part.ops_material.uid}'
                        ])
                script_append(tcl_script, ['}'])
        for elm in elms:
            # note: OpenSees has a substantially different syntax for this
            args = elm.ops_args()
            elmtype, uid, node_i, node_j, transftag, integrtag = args
            integr_type, integr_uid, integr_section, integr_np = elm.integration.ops_args()
            script_append(tcl_script, [
                f'geomTransf '+' '.join([str(x) for x in elm.geomtransf.ops_args()]),
                f'element {elmtype} {uid} {node_i} {node_j} {integr_np} {integr_section} {transftag} -Integration {integr_type}'
            ])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # ZeroLength element definition #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        elms = self.mdl.list_of_specific_element(element.ZeroLength)

        # define zerolength elements
        for elm in elms:
            for mat in elm.mats:
                define_material(mat, defined_materials)
            script_append(tcl_script, [
                f'element '+' '.join([str(x) for x in elm.ops_args()]),
            ])
            defined_elements[elm.uid] = elm

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # TwoNodeLink element definition #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        elms = self.mdl.list_of_specific_element(element.TwoNodeLink)

        # define twonodelink elements
        for elm in elms:
            for mat in elm.mats:
                define_material(mat, defined_materials)
            script_append(tcl_script, [
                f'element '+' '.join([str(x) for x in elm.ops_args()]),
            ])
            defined_elements[elm.uid] = elm

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # TrussBar element definition #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        elms = self.mdl.list_of_specific_element(element.TrussBar)

        # define TrussBar elements
        for elm in elms:
            define_material(elm.mat, defined_materials)
            script_append(tcl_script, [
                f'element '+' '.join([str(x) for x in elm.ops_args()]),
            ])
            defined_elements[elm.uid] = elm

        # ~~~~~~~~~~~~~~~~ #
        # node constraints #
        # ~~~~~~~~~~~~~~~~ #

        if self.load_cases[case_name].equaldof is None:
            for uid in parent_nodes:
                lvl = self.mdl.levels[parent_node_to_lvl[uid]]
                nodes = lvl.nodes.values()
                good_nodes = [n for n in nodes if n.coords[2] == lvl.elevation]
                script_append(tcl_script, [
                    f'rigidDiaphragm 3 {uid} '+' '.join([str(x) for x in [nd.uid for nd in good_nodes]]),
                ])
        else:
            dof = self.load_cases[case_name].equaldof
            for uid in parent_nodes:
                lvl = self.mdl.levels[parent_node_to_lvl[uid]]
                nodes = lvl.nodes.values()
                good_nodes = [n for n in nodes if n.coords[2] == lvl.elevation]
                for node in good_nodes:
                    if node.uid == uid:
                        continue
                    script_append(tcl_script, [
                        f'equalDOF {uid} {node.uid} {dof}'
                    ])

        # ~~~~~~~~~~~~~~~~~ #
        # node spring field #
        # ~~~~~~~~~~~~~~~~~ #
        # This is an attempt to resolve convergence issues during
        # rigid body motion.
        # It can also be used to restrict motion to particular DOFs

        if self.settings.restrict_dof is not None:

            restrict_dof_list = self.settings.restrict_dof
            uidgen = self.mdl.uid_generator
            fix_mat_uid = uidgen.new('uniaxial material')
            release_mat_uid = uidgen.new('uniaxial material')
            # define a very stiff spring
            script_append(tcl_script ,[
                f'uniaxialMaterial Elastic {fix_mat_uid} {common.STIFF}',
            ])
            # define a very soft spring
            script_append(tcl_script, [
                f'uniaxialMaterial Elastic {release_mat_uid} {common.TINY}',
            ])
            material_list = [
                fix_mat_uid
                if item == True
                else release_mat_uid
                for item in restrict_dof_list]
            for node in self.mdl.list_of_primary_nodes():
                # define a conjugate node that is fixed.
                node_uid = uidgen.new('node')
                elm_uid = uidgen.new('element')
                script_append(tcl_script,
                    [f'node {node_uid} '+' '.join(str(x) for x in node.coords)])
                script_append(tcl_script,
                              [
                    f'fix {node_uid} 1 1 1 1 1 1'
                ])
                # connect the nodes to the fixed node using the soft
                # spring.
                args = ['twoNodeLink', elm_uid, node.uid, node_uid,
                    '-mat',
                    *[material_list],
                    '-dir', 1, 2, 3, 4, 5, 6]
                script_append(tcl_script,
                              [
                    f'element twoNodeLink {elm_uid} {node.uid} {node_uid} -mat '+' '.join(str(x) for x in material_list)+' -dir 1 2 3 4 5 6'
                ])

    def write_tcl_node_recorders(self, tcl_script, uids, dofs, types, base_path):
        """
        Adds node recorders to a tcl script.

        Arguments:
          tcl_script: path to the tcl script in which the recorder
            commands will be appended
          uids: uids of the nodes to include
          dofs: list of dofs, 1-6.
          types: list of types of recorders to include (disp, vel,
            accel, ...)
          base_path: base path. Filenames are controlled by the type
            of recorder.

        """
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        for tp in types:
            path = f'{base_path}/{tp}_recorder.txt'
            script_append(
                tcl_script,
                [
                    f'recorder Node -file {path} -time -node '+' '.join(str(x) for x in uids)+' -dof '+' '.join(str(x) for x in dofs)+f' {tp}'
                ]
            )

    def _to_opensees_domain(self, case_name):
        """
        Defines the model in OpenSeesPy.

        """

        # initialize
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)

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
                raise KeyError(f"Node already defined: {uid}")
            defined_nodes[uid] = node
            ops.node(node.uid, *node.coords)

        # restraints
        for uid, node in primary_nodes.items():
            ops.fix(node.uid, *node.restraint)
            
        for uid, node in internal_nodes.items():
            # (this is super unusual, but who knows..)
            ops.fix(node.uid, *node.restraint)

        for node in parent_nodes.values():
            ops.fix(node.uid, *node.restraint)

        # lumped nodal mass
        for uid, node in all_nodes.items():
            # retrieve osmg node mass
            specified_mass = self.load_cases[case_name].node_mass[node.uid].val
            # replace zeros with a small mass
            # (to be able to capture all mode shapes)
            specified_mass[specified_mass == 0] = common.EPSILON
            # verify that all mass values are non-negative
            assert np.size(specified_mass[specified_mass < 0.00]) == 0
            # assign mass to the opensees node
            ops.mass(node.uid, *specified_mass)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # Elastic BeamColumn element definition #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        # keep track of defined elements
        defined_elements = {}

        elms = list(self.mdl.dict_of_specific_element(
            element.ElasticBeamColumn).values())

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
        defined_sections: dict[int, object] = {}
        defined_materials: dict[int, object] = {}

        elms = self.mdl.list_of_specific_element(element.DispBeamColumn)

        def define_material(mat, defined_materials):
            """
            A cute recursive function that defines materials with
            predecessors.

            """

            # if the actual material has not been defined yet,
            if mat.uid not in defined_materials:
                while (
                    hasattr(mat, "predecessor")
                    and mat.predecessor.uid not in defined_materials
                ):
                    # define predecessor
                    define_material(mat.predecessor, defined_materials)
                # and also define the actual material
                ops.uniaxialMaterial(*mat.ops_args())
                defined_materials[mat.uid] = mat

        for elm in elms:
            sec = elm.section
            parts = sec.section_parts.values()
            if sec.uid not in defined_sections:
                fibers = []
                for part in parts:
                    mat = part.ops_material
                    define_material(mat, defined_materials)
                    pieces = part.cut_into_tiny_little_pieces()
                    for piece in pieces:
                        area = piece.area
                        z_loc = piece.centroid.x
                        y_loc = piece.centroid.y
#                       ops.fiber(y_loc, z_loc, area, part.ops_material.uid)
                        fibers.append([y_loc, z_loc, area, part.ops_material.uid])

                fiber_commands = ';\n     '.join("fiber "+" ".join(map(str,fiber)) for fiber in fibers)
                cmd = textwrap.dedent(f"""
                section {' '.join(map(str,sec.ops_args()))} {{
                  {fiber_commands}
                }}
                """)
                ops.eval(cmd)
#               ops.section(*sec.ops_args())
                defined_sections[sec.uid] = sec

            ops.beamIntegration(*elm.integration.ops_args())
            ops.geomTransf(*elm.geomtransf.ops_args())
            ops.element(*elm.ops_args())

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # ZeroLength element definition #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        elms = self.mdl.list_of_specific_element(element.ZeroLength)

        # define zerolength elements
        for elm in elms:
            for mat in elm.mats:
                define_material(mat, defined_materials)
            ops.element(*elm.ops_args())
            defined_elements[elm.uid] = elm

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # TwoNodeLink element definition #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        elms = self.mdl.list_of_specific_element(element.TwoNodeLink)

        # define twonodelink elements
        for elm in elms:
            for mat in elm.mats:
                define_material(mat, defined_materials)
            ops.element(*elm.ops_args())
            defined_elements[elm.uid] = elm

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # TrussBar element definition #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        elms = self.mdl.list_of_specific_element(element.TrussBar)

        # define TrussBar elements
        for elm in elms:
            define_material(elm.mat, defined_materials)
            ops.element(*elm.ops_args())
            defined_elements[elm.uid] = elm

        # ~~~~~~~~~~~~~~~~ #
        # node constraints #
        # ~~~~~~~~~~~~~~~~ #

        if self.load_cases[case_name].equaldof is None:
            for uid in parent_nodes:
                lvl = self.mdl.levels[parent_node_to_lvl[uid]]
                nodes = lvl.nodes.values()
                good_nodes = [n for n in nodes if n.coords[2] == lvl.elevation]
                ops.rigidDiaphragm(3, uid, *[nd.uid for nd in good_nodes])
        else:
            dof = self.load_cases[case_name].equaldof
            for uid in parent_nodes:
                lvl = self.mdl.levels[parent_node_to_lvl[uid]]
                nodes = lvl.nodes.values()
                good_nodes = [n for n in nodes if n.coords[2] == lvl.elevation]
                for node in good_nodes:
                    if node.uid == uid:
                        continue
                    ops.equalDOF(uid, node.uid, dof)

        # ~~~~~~~~~~~~~~~~~ #
        # node spring field #
        # ~~~~~~~~~~~~~~~~~ #
        # This is an attempt to resolve convergence issues during
        # rigid body motion.
        # It can also be used to restrict motion to particular DOFs

        if self.settings.restrict_dof is not None:

            restrict_dof_list = self.settings.restrict_dof
            uidgen = self.mdl.uid_generator
            fix_mat_uid = uidgen.new('uniaxial material')
            release_mat_uid = uidgen.new('uniaxial material')
            # define a very stiff spring
            ops.uniaxialMaterial('Elastic', fix_mat_uid, common.STIFF)
            # define a very soft spring
            ops.uniaxialMaterial('Elastic', release_mat_uid, common.TINY)
            material_list = [
                fix_mat_uid
                if item == True
                else release_mat_uid
                for item in restrict_dof_list]
            for node in self.mdl.list_of_primary_nodes():
                # define a conjugate node that is fixed.
                node_uid = uidgen.new('node')
                elm_uid = uidgen.new('element')
                ops.node(node_uid, *node.coords)
                ops.fix(node_uid, 1, 1, 1, 1, 1, 1)
                # connect the nodes to the fixed node using the soft
                # spring.
                args = ['twoNodeLink', elm_uid, node.uid, node_uid,
                    '-mat',
                    *[material_list],
                    '-dir', 1, 2, 3, 4, 5, 6]
                ops.element(
                    'twoNodeLink', elm_uid, node.uid, node_uid,
                    '-mat',
                    *material_list,
                    '-dir', 1, 2, 3, 4, 5, 6
                )

    def write_tcl_loads(self, case_name, tcl_script, factor=1.00):

        script_append(tcl_script, ['pattern Plain 1 Linear {'])
        elms_with_udl: list[Union[
            element.ElasticBeamColumn, element.DispBeamColumn
        ]] = []
        elms_with_udl.extend(
            [elm for elm in self.mdl.list_of_specific_element(element.ElasticBeamColumn)
             if isinstance(elm, element.ElasticBeamColumn)])
        elms_with_udl.extend(
            [elm for elm in self.mdl.list_of_specific_element(element.DispBeamColumn)
             if isinstance(elm, element.DispBeamColumn)])
        for elm in elms_with_udl:
            if elm.visibility.skip_opensees_definition:
                continue
            udl_total = (
                self.load_cases[case_name].line_element_udl[elm.uid].val
            )
            if not np.isclose(np.sqrt(udl_total @ udl_total), 0.00):
                script_append(tcl_script, [
                    f'eleLoad -ele {elm.uid} -type -beamUniform {udl_total[1]*factor} {udl_total[2]*factor} {udl_total[0]*factor}'
                ])

        for node in self.mdl.list_of_all_nodes():
            script_append(tcl_script, [
                f'load {node.uid} '+' '.join(str(x) for x in self.load_cases[case_name].node_loads[node.uid].val*factor)
            ])
        script_append(tcl_script, ['}'])

    def _define_loads(self, case_name, factor=1.00):

        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        elms_with_udl: list[Union[
            element.ElasticBeamColumn, element.DispBeamColumn
        ]] = []
        elms_with_udl.extend(
            [elm for elm in self.mdl.list_of_specific_element(element.ElasticBeamColumn)
             if isinstance(elm, element.ElasticBeamColumn)])
        elms_with_udl.extend(
            [elm for elm in self.mdl.list_of_specific_element(element.DispBeamColumn)
             if isinstance(elm, element.DispBeamColumn)])
        for elm in elms_with_udl:
            if elm.visibility.skip_opensees_definition:
                continue
            udl_total = (
                self.load_cases[case_name].line_element_udl[elm.uid].val
            )
            if not np.isclose(np.sqrt(udl_total @ udl_total), 0.00):
                ops.eleLoad(
                    "-ele",
                    elm.uid,
                    "-type",
                    "-beamUniform",
                    udl_total[1]*factor,
                    udl_total[2]*factor,
                    udl_total[0]*factor,
                )

        for node in self.mdl.list_of_all_nodes():
            ops.load(
                node.uid, *(self.load_cases[case_name].node_loads[node.uid].val*factor)
            )


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
            if isinstance(elm, (element.ElasticBeamColumn, element.DispBeamColumn)):
                moments_global_clear = transformations.offset_transformation(
                    elm.geomtransf.offset_i, moments_global_ends, forces_global
                )
                x_vec = elm.geomtransf.x_axis
                y_vec = elm.geomtransf.y_axis
                z_vec = elm.geomtransf.z_axis
            else:
                moments_global_clear = moments_global_ends
                x_vec, y_vec, z_vec = (
                    transformations.local_axes_from_points_and_angle(
                        np.array(elm.nodes[0].coords),
                        np.array(elm.nodes[1].coords),
                        0.00))
            transf_global2local: nparr = np.vstack((x_vec, y_vec, z_vec))
            n_i, qy_i, qz_i = transf_global2local @ forces_global
            t_i, my_i, mz_i = transf_global2local @ moments_global_clear
            forces: nparr = np.array((n_i, qy_i, qz_i, t_i, my_i, mz_i))
            self.results[case_name].element_forces[uid][step] = forces

    def _read_release_moment_rot(self, case_name, step, zerolength_elms):
        for release in zerolength_elms:
            # force_global = ops.eleResponse(
            #     release.uid, 'force')[:3]
            moment_global = ops.eleResponse(release.uid, "force")[3:6]
            # disp_local = ops.eleResponse(
            #     release.uid, 'deformation')[:3]
            rot_local = ops.eleResponse(release.uid, "deformation")[3:6]
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
                vec_x, vec_y, vec_z
            )
            moment_local = tmat_g2l @ -(np.array(moment_global))
            self.results[case_name].release_force_defo[release.uid][step] = [
                *rot_local,
                *moment_local,
            ]

    def _read_opensees_results(
        self,
        case_name,
        step,
        nodes,
        line_elements,
        zerolength_elements,
    ):
        self._read_node_displacements(case_name, step, nodes)
        self._read_node_velocities(case_name, step, nodes)
        self._read_node_accelerations(case_name, step, nodes)
        if self.settings.store_reactions:
            self._read_node_reactions(case_name, step, nodes)
        if self.settings.store_forces:
            self._read_frame_element_forces(
                case_name, step, line_elements
            )
        # if self.settings.store_fiber:
        #     self._read_frame_fiber_stress_strain()
        if self.settings.store_release_force_defo:
            self._read_release_moment_rot(case_name, step, zerolength_elements)

    ##################################
    # Numeric Result Post-processing #
    ##################################

    def global_reactions(self, case_name, step):
        """
        Calculates and returns the global reaction forces.

        """

        reactions = np.full(6, 0.00)
        # for lvl in self.mdl.levels.values():
        lvl = list(self.mdl.levels.values())[0] # temporary fix
        for node in lvl.nodes.values():
            if True in node.restraint:
                uid = node.uid
                x_coord = node.coords[0]
                y_coord = node.coords[1]
                z_coord = node.coords[2]
                local_reaction = np.array(
                    self.results[case_name].node_reactions[
                        uid
                    ][step])
                # bug fix: It has been observed that sometimes
                # OpenSees reports reactions to unrestrained DOFs.
                # https://opensees.berkeley.edu/community/
                # viewtopic.php?f=12&t=70795
                # To overcome this, we replace the reported
                # reactions corresponding to unrestrained DOFs
                # with zero in the following line
                local_reaction[~np.array(node.restraint)] = 0.00

                # transfer moments to the global coordinate system
                global_reaction: nparr = np.array(
                    [
                        local_reaction[0],
                        local_reaction[1],
                        local_reaction[2],
                        local_reaction[3]
                        + local_reaction[2] * y_coord
                        - local_reaction[1] * z_coord,
                        local_reaction[4]
                        + local_reaction[0] * z_coord
                        - local_reaction[2] * x_coord,
                        local_reaction[5]
                        + local_reaction[1] * x_coord
                        - local_reaction[0] * y_coord,
                    ]
                )
                # add to the global reactions
                reactions += global_reaction
        return reactions


@dataclass
class StaticAnalysis(Analysis):
    """
    Static analysis.  Stores all results (for each load case) in one
    single step.

    """

    def run(self, num_subdiv=1):
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
                elm
                for elm in self.mdl.list_of_specific_element(element.ElasticBeamColumn)
                if not elm.visibility.skip_opensees_definition
            ]
            disp_elems = [
                elm
                for elm in self.mdl.list_of_specific_element(element.DispBeamColumn)
                if not elm.visibility.skip_opensees_definition
            ]
            truss_elems = [
                elm
                for elm in self.mdl.list_of_specific_element(element.TrussBar)
                if not elm.visibility.skip_opensees_definition
            ]
            line_elems = elastic_elems + disp_elems + truss_elems
            zerolength_elems = self.mdl.list_of_specific_element(element.ZeroLength)
            step = 0
            ops.system(self.settings.solver)
            ops.numberer(NUMBERER)
            ops.constraints(*CONSTRAINTS)
            ops.test("EnergyIncr", 1.0e-8, 20, 3)
            ops.algorithm("KrylovNewton")
            ops.integrator("LoadControl", 1.00/num_subdiv)
            ops.analysis("Static")
            for _ in range(num_subdiv):
                ops.analyze(1)
            self._read_opensees_results(
                case_name,
                step,
                nodes,
                line_elems,
                zerolength_elems,
            )
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
                val = ops.nodeEigenvector(node.uid, i + 1)
                self.results[case_name].node_displacements[node.uid][i] = val

    def _read_frame_element_forces_modal(self, case_name, elems):
        # note: opensees does not output the element forces that correspond
        # to the displacement field of each obtained mode.
        # to overcome this, we run a separate analysis imposing those
        # displacements to get the element forces.

        for step in range(self.num_modes):
            for elm in elems:
                # displacements at the two ends (global system)
                u_i = np.array(self.results[case_name].node_displacements[
                    elm.nodes[0].uid
                ][step][0:3])
                r_i = np.array(self.results[case_name].node_displacements[
                    elm.nodes[0].uid
                ][step][3:6])
                u_j = np.array(self.results[case_name].node_displacements[
                    elm.nodes[1].uid
                ][step][0:3])
                r_j = np.array(self.results[case_name].node_displacements[
                    elm.nodes[1].uid
                ][step][3:6])
                if isinstance(elm, element.TrussBar):
                    offset_i = np.zeros(3)
                    offset_j = np.zeros(3)
                    u_i_o = u_i
                    u_j_o = u_j
                    x_vec, y_vec, z_vec = (
                        transformations.local_axes_from_points_and_angle(
                            np.array(elm.nodes[0].coords),
                            np.array(elm.nodes[1].coords), 0.00
                        ))
                else:
                    offset_i = elm.geomtransf.offset_i
                    offset_j = elm.geomtransf.offset_j
                    u_i_o = transformations.offset_transformation(
                        offset_i, np.array(u_i), np.array(r_i)
                    )
                    u_j_o = transformations.offset_transformation(
                        offset_j, np.array(u_j), np.array(r_j)
                    )
                    x_vec = elm.geomtransf.x_axis
                    y_vec = elm.geomtransf.y_axis
                    z_vec = np.cross(x_vec, y_vec)

                    # element UDL: shouldn't be there in a modal analysis
                    udl = self.load_cases[case_name].line_element_udl[elm.uid].val
                    # note: modal analysis doesn't account for applied loads.
                    # this will cause issues with plotting if loads
                    # have been applied.
                    if np.linalg.norm(udl) > common.EPSILON:
                        self.warning.issue("Loads were present in a modal load case. Ignoring them.")
                        udl = np.zeros(3)

                # global -> local transformation matrix
                transf_global2local = transformations.transformation_matrix(
                    x_vec, y_vec, z_vec
                )

                u_i_local = transf_global2local @ u_i_o
                r_i_local = transf_global2local @ r_i
                u_j_local = transf_global2local @ u_j_o
                r_j_local = transf_global2local @ r_j

                # stiffness matrix terms
                if isinstance(elm, element.TrussBar):
                    if isinstance(elm.mat, uniaxial_material.Elastic):
                        e_mod = elm.mat.e_mod
                    else:
                        self.print(
                            'Ignoring truss element with '
                            f'{elm.mat.__class__.__name__} material')
                        e_mod = 0.00
                    etimesa = e_mod * elm.area
                    etimesi_maj = 0.00
                    etimesi_min = 0.00
                    gtimesj = 0.00
                elif isinstance(elm, element.ElasticBeamColumn):
                    etimesa = elm.section.e_mod * elm.section.area
                    etimesi_maj = elm.section.e_mod * elm.section.i_x
                    etimesi_min = elm.section.e_mod * elm.section.i_y
                    gtimesj = elm.section.g_mod * elm.section.j_mod
                else:
                    raise ValueError(
                        'Oops! Need to extend the code '
                        'to support dispBeamColumn elements')

                length = elm.clear_length()

                deformation_vector = np.concatenate(
                    (u_i_local, r_i_local, u_j_local, r_j_local)
                )

                # axial load
                n_i = (
                    np.array([[etimesa / length, -etimesa / length]])
                    @ deformation_vector[[0, 6]]
                )

                # torsion
                t_i = (
                    np.array([[gtimesj / length, -gtimesj / length]])
                    @ deformation_vector[[3, 9]]
                )

                # major shear and minor bending
                f3_m2 = (
                    np.array(
                        [
                            [
                                12.00 * etimesi_min / length**3,
                                -6.00 * etimesi_min / length**2,
                                -12.00 * etimesi_min / length**3,
                                -6.00 * etimesi_min / length**2,
                            ],
                            [
                                -6.00 * etimesi_min / length**2,
                                4.00 * etimesi_min / length,
                                6.00 * etimesi_min / length**2,
                                2.00 * etimesi_min / length,
                            ],
                        ]
                    )
                    @ deformation_vector[[2, 4, 8, 10]]
                )

                # minor shear and major bending
                f2_m3 = (
                    np.array(
                        [
                            [
                                12.00 * etimesi_maj / length**3,
                                6.00 * etimesi_maj / length**2,
                                -12.00 * etimesi_maj / length**3,
                                6.00 * etimesi_maj / length**2,
                            ],
                            [
                                6.00 * etimesi_maj / length**2,
                                4.00 * etimesi_maj / length,
                                -6.00 * etimesi_maj / length**2,
                                2.00 * etimesi_maj / length,
                            ],
                        ]
                    )
                    @ deformation_vector[[1, 5, 7, 11]]
                )

                forces_vector_local = np.array(
                    (n_i[0], f2_m3[0], f3_m2[0], t_i[0], f3_m2[1], f2_m3[1])
                )

                # store results
                (
                    self.results[case_name].element_forces
                    # [elm.uid][step]) = \
                    #    np.array((n_i, qy_i, qzi, t_i, myi, mz_i))
                    [elm.uid][step]
                ) = forces_vector_local

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
            if self.settings.solver.lower() in (
                    'sparsesym', 'sparsespd'):
                raise ValueError(
                    f'{self.settings.solver} is unable '
                    'to run a modal analysis. Use UmfPack.')
            ops.system(self.settings.solver)
            # note: using SparseSYM results in wrong eigen decomposition
            num_inertial_nodes = 0
            ndtags = ops.getNodeTags()
            for node in ndtags:
                for j in range(6):
                    if ops.nodeMass(node, j + 1) > 0.00:
                        num_inertial_nodes += 1
            eigenvalues: nparr = np.array(ops.eigen(self.num_modes))
            self.results[case_name].periods = (
                2.00 * np.pi / np.sqrt(eigenvalues)
            )
            self._read_node_displacements_modal(case_name)
            if self.settings.store_forces:
                line_elements: list[Union[
                    element.TrussBar,
                    element.ElasticBeamColumn,
                    element.DispBeamColumn
                ]] = []
                line_elements.extend(
                    [elm for elm in self.mdl.list_of_specific_element(element.TrussBar)
                     if isinstance(elm, element.TrussBar)])
                line_elements.extend(
                    [elm for elm in self.mdl.list_of_specific_element(element.ElasticBeamColumn)
                     if isinstance(elm, element.ElasticBeamColumn)])
                line_elements.extend(
                    [elm for elm in self.mdl.list_of_specific_element(element.DispBeamColumn)
                     if isinstance(elm, element.DispBeamColumn)])
                self._read_frame_element_forces_modal(
                    case_name, line_elements
                )
        if self.settings.pickle_results:
            self._write_results_to_disk()

    def modal_participation_factors(self, case_name, direction):
        """
        Calculates modal participation factors

        """

        dof_dir = {"x": 0, "y": 1, "z": 2}
        ntgs = list(self.mdl.dict_of_all_nodes().keys())
        # if there is a rigid diaphragm, we also need to include the parent nodes
        loadcase = self.load_cases[case_name]
        if loadcase.parent_nodes:
            pnodes = [x.uid for x in loadcase.parent_nodes.values()]
            ntgs.extend(pnodes)
        gammas = np.zeros(self.num_modes)
        mstars = np.zeros(self.num_modes)
        mn_tot = 0.0
        for ntg in ntgs:
            node_mass = loadcase.node_mass[ntg].val
            mn_tot += node_mass[dof_dir[direction]]
        for mode in range(self.num_modes):
            l_n = 0.0
            m_n = 0.0
            for ntg in ntgs:
                node_mass = loadcase.node_mass[ntg].val
                node_phi = ops.nodeEigenvector(ntg, mode + 1)
                l_n += (
                    node_phi[dof_dir[direction]]
                    * node_mass[dof_dir[direction]]
                )
                for dof in range(6):
                    m_n += (node_phi[dof] ** 2) * node_mass[dof]
            gammas[mode] = l_n / m_n
            mstars[mode] = l_n**2 / m_n
        mstars /= mn_tot
        return (gammas, mstars, mn_tot)


@dataclass
class GravityPlusAnalysis(Analysis):
    """
    When performing nonlinear static or dynamic analysis, it is common
    to first apply gravitly loads on the model and then proceed with
    some other analysis, like static pushover or transient dynamic
    analysis. This parent class is used to define analysis objects
    that follow this practice.

    """

    def write_tcl_gravity_analysis(self, tcl_script, num_steps=1):

        system = self.settings.solver
        script_append(
            tcl_script,
            [
             'test EnergyIncr 1.0e-6 100 3',
             f'system {system}',
             f'numberer {NUMBERER}',
             'constraints '+' '.join([str(x) for x in CONSTRAINTS]),
             'algorithm KrylovNewton',
             'integrator LoadControl 1',
             'analysis Static',
             f'set check [analyze {num_steps}]'
            ]
        )
        script_append(
            tcl_script,
            [
                'if {$check != 0} {',
                '    puts "Gravity analysis failed. Unable to continue..."',
                '    error "Analysis Failed"',
                '}'
            ])

    def _run_gravity_analysis(self, num_steps=1):
        self.log("G: Setting test to ('EnergyIncr', 1.0e-6, 100, 3)")
        ops.test("EnergyIncr", 1.0e-6, 100, 3)
        system = self.settings.solver
        self.log(f"G: Setting system solver to {system}")
        ops.system(system)
        self.log(f"G: Setting numberer to {NUMBERER}")
        ops.numberer(NUMBERER)
        self.log(f"G: Setting constraints to {[*CONSTRAINTS]}")
        ops.constraints(*CONSTRAINTS)
        self.log("G: Setting algorithm to KrylovNewton")
        ops.algorithm("KrylovNewton")
        ops.integrator("LoadControl", 1)
        self.log("G: Setting analysis to Static")
        ops.analysis("Static")
        self.log("G: Analyzing now.")
        check = ops.analyze(num_steps)
        if check != 0:
            self.log("Gravity analysis failed. Unable to continue...")
            raise ValueError("Analysis Failed")

    def retrieve_node_displacement(self, uid, case_name):
        """
        Returns the displacement of a node for all analysis steps

        """

        if case_name not in self.results:
            raise ValueError(f"case_name {case_name} not found in results.")
        res = np.zeros((self.results[case_name].n_steps_success, 6))
        num = len(self.results[case_name].node_displacements[uid])
        for i in range(num):
            res[i] = self.results[case_name].node_displacements[uid][i]
        dframe = pd.DataFrame(
            res, columns=["ux", "uy", "uz", "urx", "ury", "urz"]
        )
        dframe.index.name = "step"
        return dframe

    def retrieve_node_acceleration(self, uid, case_name):
        """
        Returns the acceleration of a node for all analysis steps

        """

        res = np.zeros((self.results[case_name].n_steps_success, 6))
        num = len(self.results[case_name].node_accelerations[uid])
        for i in range(num):
            res[i] = self.results[case_name].node_accelerations[uid][i]
        dframe = pd.DataFrame(
            res, columns=["ax", "ay", "az", "arx", "ary", "arz"]
        )
        dframe.index.name = "step"
        return dframe

    def retrieve_node_velocity(self, uid, case_name):
        """
        Returns the velocity of a node for all analysis steps

        """

        res = np.zeros((self.results[case_name].n_steps_success, 6))
        num = len(self.results[case_name].node_velocities[uid])
        for i in range(num):
            res[i] = self.results[case_name].node_velocities[uid][i]
        dframe = pd.DataFrame(
            res, columns=["vx", "vy", "vz", "vrx", "vry", "vrz"]
        )
        dframe.index.name = "step"
        return dframe

    def retrieve_node_abs_acceleration(self, uid, case_name):
        """
        Returns the absolute acceleration of a node for all analysis
        steps

        """

        res = np.zeros((self.results[case_name].n_steps_success, 6))
        num = len(self.results[case_name].node_accelerations[uid])
        assert isinstance(self, THAnalysis)
        assert self.a_g is not None
        for i in range(num):
            res[i] = self.results[case_name].node_accelerations[uid][i]
        for j in range(3):
            if j in self.a_g:
                a_g = interp1d(
                    self.a_g[j][:, 0],
                    self.a_g[j][:, 1],
                    bounds_error=False,
                    fill_value=0.00,
                )
                res[:, j] += a_g(self.time_vector) * common.G_CONST_IMPERIAL
                # TODO: update to support SI
        dframe = pd.DataFrame(
            res,
            columns=[
                "abs ax",
                "abs ay",
                "abs az",
                "abs arx",
                "abs ary",
                "abs arz",
            ],
        )
        dframe.index.name = "step"
        return dframe

    def retrieve_node_abs_velocity(self, uid, case_name):
        """
        Returns the absolute velocity of a node for all analysis steps

        """

        res = np.zeros((self.results[case_name].n_steps_success, 6))
        num = len(self.results[case_name].node_velocities[uid])
        assert isinstance(self, THAnalysis)
        assert self.a_g is not None
        for i in range(num):
            res[i] = self.results[case_name].node_velocities[uid][i]
        for j in range(3):
            if j in self.a_g:
                a_g = interp1d(
                    self.a_g[j][:, 0],
                    self.a_g[j][:, 1],
                    bounds_error=False,
                    fill_value=0.00,
                )
                v_g = integrate.cumulative_trapezoid(
                    a_g(self.time_vector) * common.G_CONST_IMPERIAL,
                    self.time_vector,
                    initial=0,
                )
                res[:, j] = res[:, j] + v_g
        dfrmae = pd.DataFrame(
            res,
            columns=[
                "abs vx",
                "abs vy",
                "abs vz",
                "abs vrx",
                "abs vry",
                "abs vrz",
            ],
        )
        dfrmae.index.name = "step"
        return dfrmae

    def retrieve_base_shear(self, case_name):
        """
        Returns the base shear response history

        """
        base_shear_lst = []
        for step in range(
            self.results[case_name].n_steps_success
        ):  # type:ignore
            base_shear_lst.append(
                self.global_reactions(case_name, step)[0:3]
            )
        base_shear: nparr = np.array(base_shear_lst)
        return base_shear

    def retrieve_release_force_defo(self, uid, case_name):
        """
        Returns the force-deformation of a zerolength element for all
        analysis steps

        """

        num = len(self.results[case_name].release_force_defo[uid])
        res = np.zeros((num, 6))
        for i in range(num):
            res[i] = self.results[case_name].release_force_defo[uid][i]
        dframe = pd.DataFrame(
            res, columns=["u1", "u2", "u3", "q1", "q2", "q3"]
        )
        dframe.index.name = "step"
        return dframe


@dataclass
class PushoverAnalysis(GravityPlusAnalysis):
    """
    Pushover analysis

    """

    def _apply_lateral_load(
        self, case_name, direction, modeshape=None, node=None
    ):
        query = LoadCaseQuery(self.mdl, self.load_cases[case_name])
        distribution = query.level_masses()
        distribution = distribution / np.linalg.norm(distribution)

        # define the load pattern
        ops.timeSeries("Linear", 2)
        ops.pattern("Plain", 2, 2)

        if direction == "x":
            load_dir: nparr = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif direction == "y":
            load_dir = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        elif direction == "z":
            load_dir = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        else:
            raise ValueError("Invalid direction")

        if modeshape is not None:
            if direction not in ["x", "y"]:
                raise ValueError(
                    "Can't apply lateral loads based on the 1st "
                    + "mode shape in the z direction."
                )
            modeshape_ampl = modeshape / modeshape[-1]
        else:
            modeshape_ampl = np.ones(len(self.mdl.levels.values()))

        # if a node is given, apply the incremental load on that node
        if node:
            ops.load(node.uid, *(1.00 * load_dir))
        else:
            for i, lvl in enumerate(self.mdl.levels.values()):
                if lvl.uid == 0:
                    continue
                if self.load_cases[case_name].parent_nodes:
                    node_list = [
                        self.load_cases[case_name].parent_nodes[lvl.uid]
                    ]
                else:
                    node_list = list(lvl.nodes.values())
                masses: nparr = np.array(
                    [
                        self.load_cases[case_name].node_mass[n.uid].val[0]
                        for n in node_list
                    ]
                )
                masses = masses / np.linalg.norm(masses)
                for j, some_node in enumerate(node_list):
                    ops.load(
                        some_node.uid,
                        *(
                            distribution[i]
                            * masses[j]
                            * load_dir
                            * modeshape_ampl[i]
                        ),
                    )

    def run(
        self,
        direction,
        target_displacements,
        control_node,
        displ_incr,
        integrator='DisplacementControl',
        modeshape=None,
        loaded_node=None,
    ):
        """
        Run pushover analysis

        Arguments:
          direction: can be any of 'x', 'y', 'z'
          target_displacements: a list of target displacements.  each
            time a target is reached, the analysis continues until the
            next target is reached, flipping the direction as
            necessary.
          control_node: analysis control node (of which the direction
            is queried)
          displ_incr: initial displacement increment or arc length.
          integrator: Any of 'DisplacementControl' or 'ArcLength'
          mode shape: array containing a mode shape that is used to
            distribute the applied incremental loads. If no mode shape
            is specified, the distribution is uniform.
          loaded_node: if a loaded node is specified, all incremental
            load is applied entirely on that node.  Otherwise, the
            incremental loads are distributed to all nodes.

        """

        self.log(f"Direction: {direction}")
        if direction == "x":
            control_dof = 0
        elif direction == "y":
            control_dof = 1
        elif direction == "z":
            control_dof = 2
        else:
            raise ValueError("Direction can be 'x', 'y' or 'z'")

        self.log("Initializing containers")
        self._init_results()

        for case_name in self.load_cases:
            self.log(f"Load case: {case_name}")
            nodes = self.mdl.list_of_all_nodes()
            nodes.extend(self.load_cases[case_name].parent_nodes.values())
            elastic_elems = [
                elm
                for elm in self.mdl.list_of_specific_element(element.ElasticBeamColumn)
                if not elm.visibility.skip_opensees_definition
            ]
            disp_elems = [
                elm
                for elm in self.mdl.list_of_specific_element(element.DispBeamColumn)
                if not elm.visibility.skip_opensees_definition
            ]
            truss_elems = [
                elm
                for elm in self.mdl.list_of_specific_element(element.TrussBar)
                if not elm.visibility.skip_opensees_definition
            ]
            line_elems = elastic_elems + disp_elems + truss_elems
            zerolength_elems = self.mdl.list_of_specific_element(element.ZeroLength)

            self.log("Defining elements in OpenSees")
            self._to_opensees_domain(case_name)


            self.log("Defining loads")
            self._define_loads(case_name, 0.01)
            self.log("Running gravity analysis")
            self._run_gravity_analysis(100)

            curr_displ = ops.nodeDisp(control_node.uid, control_dof + 1)
            n_steps_success = 0
            self._read_opensees_results(
                case_name,
                n_steps_success,
                nodes,
                line_elems,
                zerolength_elems,
            )

            self.log("Starting pushover analysis")
            ops.wipeAnalysis()
            ops.loadConst("-time", 0.0)
            self._apply_lateral_load(
                case_name, direction, modeshape, loaded_node
            )
            ops.numberer(NUMBERER)
            ops.constraints(*CONSTRAINTS)

            total_fail = False
            num_subdiv = 0
            num_times = 0
            algorithm_idx = 0

            scale = [1.0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11]
            steps = [500]*12
            norm = [1.0e-15]*12
            algorithms = [("KrylovNewton", ), ("KrylovNewton", 'initial')]

            try:

                for i_loop, target_displacement \
                    in enumerate(target_displacements):

                    if total_fail:
                        break

                    # determine push direction
                    if curr_displ < target_displacement:
                        displ_incr = abs(displ_incr)
                        sign = +1.00
                    else:
                        displ_incr = -abs(displ_incr)
                        sign = -1.00

                    while curr_displ * sign < target_displacement * sign:

                        # determine increment
                        if (
                            abs(curr_displ - target_displacement)
                            < abs(displ_incr) * scale[num_subdiv]
                        ):
                            incr = sign * abs(curr_displ - target_displacement)
                        else:
                            incr = displ_incr * scale[num_subdiv]
                        ops.test(
                            "NormDispIncr",
                            norm[num_subdiv],
                            steps[num_subdiv],
                            0,
                        )
                        ops.algorithm(*algorithms[algorithm_idx])
                        if integrator == 'DisplacementControl':
                            ops.integrator(
                                "DisplacementControl",
                                int(control_node.uid),
                                control_dof + 1,
                                incr,
                            )
                        elif integrator == 'ArcLength':
                            ops.integrator("ArcLength", incr, 1e-7)
                        else:
                            raise ValueError(
                                f'Invalid integrator: {integrator}')
                        ops.system(self.settings.solver)
                        ops.analysis("Static")
                        flag = ops.analyze(1)
                        if flag != 0:
                            if num_subdiv == len(scale) - 1:
                                # can't refine further
                                self.print("===========================")
                                self.print("Analysis failed to converge")
                                self.print("===========================")
                                if self.logger:
                                    self.logger.warning(
                                        "Analysis failed"
                                        f" at disp {curr_displ:.5f}"
                                    )
                                total_fail = True
                                break
                            # can still reduce step size
                            if algorithm_idx != len(algorithms) - 1:
                                algorithm_idx += 1
                            else:
                                algorithm_idx = 0
                                num_subdiv += 1
                                # how many times to run with reduced step size
                                num_times = 50
                        else:
                            # analysis was successful
                            if num_times != 0:
                                num_times -= 1
                            n_steps_success += 1
                            self._read_opensees_results(
                                case_name,
                                n_steps_success,
                                nodes,
                                line_elems,
                                zerolength_elems
                            )

                            curr_displ = ops.nodeDisp(
                                int(control_node.uid), control_dof + 1
                            )
                            self.print(
                                f"Loop ({i_loop+1}/"
                                f"{len(target_displacements)}) | "
                                "Target displacement: "
                                f"{target_displacement:.2f}"
                                f" | Current: {curr_displ:.4f}",
                                end="\r",
                            )
                            algorithm_idx = 0
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
                line_elems,
                zerolength_elems)
            self.print(f"Number of saved analysis steps: {n_steps_success}")
            metadata: dict[str, object] = {"successful steps": n_steps_success}
            self.results[case_name].n_steps_success = n_steps_success
            self.results[case_name].metadata = metadata
        # done with all cases.
        if self.settings.pickle_results:
            self._write_results_to_disk()

    def table_pushover_curve(self, case_name, direction, node):
        """
        Returns the force deformation results

        """

        if direction == "x":
            control_dof = 0
        elif direction == "y":
            control_dof = 1
        elif direction == "z":
            control_dof = 2
        else:
            raise ValueError("Direction can be 'x', 'y' or 'z'")
        base_shear_lst = []
        displacement_lst = []
        for step in range(
            self.results[case_name].n_steps_success
        ):  # type:ignore
            base_shear_lst.append(
                self.global_reactions(case_name, step)[control_dof]
            )
            displacement_lst.append(
                self.results[case_name].node_displacements[node.uid][step][
                    control_dof
                ]
            )
        base_shear: nparr = -np.array(base_shear_lst)
        displacement: nparr = np.array(displacement_lst)
        return displacement, base_shear

    def plot_pushover_curve(self, case_name, direction, node):
        """
        Plots the pushover curve

        """

        # TODO: units
        displacement, base_shear = self.table_pushover_curve(
            case_name, direction, node
        )
        general_2d.line_plot_interactive(
            "Pushover Analysis Results<br>" + "Direction: " + direction,
            displacement,
            base_shear,
            "spline+markers",
            "Displacement",
            "in",
            ".0f",
            "Base Shear",
            "lb",
            ".0f",
        )

def define_lateral_load_pattern(
    ag_x, ag_y, ag_z, file_time_incr
):
    """
    Defines the load pattern for a time-history analysis from
    previously parsed files with a constant dt

    """

    error = True
    if ag_x is not None:
        error = False
        # define X-direction TH
        ops.timeSeries(
            "Path",
            2,
            "-dt",
            file_time_incr,
            "-values",
            *ag_x,
            "-factor",
            common.G_CONST_IMPERIAL,
        )
        # pattern, direction, time series tag
        ops.pattern("UniformExcitation", 2, 1, "-accel", 2)

    if ag_y is not None:
        error = False
        # define Y-direction TH
        ops.timeSeries(
            "Path",
            3,
            "-dt",
            file_time_incr,
            "-values",
            *ag_y,
            "-factor",
            common.G_CONST_IMPERIAL,
        )
        # pattern, direction, time series tag
        ops.pattern("UniformExcitation", 3, 2, "-accel", 3)

    if ag_z is not None:
        error = False
        # define Z-direction TH
        ops.timeSeries(
            "Path",
            4,
            "-dt",
            file_time_incr,
            "-values",
            *ag_z,
            "-factor",
            common.G_CONST_IMPERIAL,
        )
        # pattern, direction, time series tag
        ops.pattern("UniformExcitation", 4, 3, "-accel", 4)

    if error:
        raise ValueError("No input files specified.")


def plot_ground_motion(filename, file_time_incr, gmunit="g", plotly=False):
    """
    Plots a ground motion input file.

    """

    y_vals = np.loadtxt(filename)
    n_points = len(y_vals)
    x_vals = np.arange(0.00, n_points * file_time_incr, file_time_incr)
    if plotly:
        general_2d.line_plot_interactive(
            "Ground motion record<br>" + filename,
            x_vals,
            y_vals,
            "line",
            "Time",
            "s",
            ".3f",
            "Absolute Acceleration",
            gmunit,
            ".4f",
        )
    else:
        plt.figure()
        plt.plot(x_vals, y_vals, "k")
        plt.xlabel("Time (s)")
        plt.ylabel(f"Acceleration ({gmunit})")
        plt.show()


@dataclass
class THAnalysis(GravityPlusAnalysis):
    """
    Dynamic time-history analysis

    """

    time_vector: list[float] = field(default_factory=list, repr=False)
    a_g: dict[int, npt.NDArray[np.float64]] = field(
        default_factory=dict, repr=False
    )

    def write_tcl_th_pattern(self, tcl_script, ag_x, ag_y, ag_z, gm_dt, base_path):

        ag_dir = {'x': ag_x, 'y': ag_y, 'z': ag_z}
        ag_tag = {'x': '2', 'y': '3', 'z': '4'}
        ag_dof = {'x': '1', 'y': '2', 'z': '3'}

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        script_append(
            tcl_script,
            ['wipeAnalysis', 'loadConst -time 0.00']
        )

        for dr in ('x', 'y', 'z'):
            ag = ag_dir[dr]
            if ag is not None:
                np.savetxt(f'{base_path}/ag_{dr}', ag)
                script_append(
                    tcl_script,
                    [
                        f'set AccelSeries_{dr} "Series -dt {gm_dt} -filePath {base_path}/ag_{dr}"',
                        f'pattern UniformExcitation {ag_tag[dr]} {ag_dof[dr]} -accel $AccelSeries_{dr}'
                    ]
            )


    def write_tcl_th_analysis(
        self, tcl_script,
        finish_time, damping, time_increment, output_dir,
        drift_nodes, drift_heights,
        skip_steps=1, collapse_drift=0.10):

        damping_type = damping.get("type")

        target_timestamp = finish_time

        damping_code = ''

        if damping_type == "rayleigh":
            assert isinstance(damping["periods"], list)
            assert isinstance(damping["periods"][0], float)
            w_i = 2 * np.pi / damping["periods"][0]
            zeta_i = damping["ratio"]
            assert isinstance(damping["periods"][1], float)
            w_j = 2 * np.pi / damping["periods"][1]
            zeta_j = damping["ratio"]
            a_mat: nparr = np.array([[1 / w_i, w_i], [1 / w_j, w_j]])
            b_vec: nparr = np.array([zeta_i, zeta_j])
            x_sol: nparr = np.linalg.solve(a_mat, 2 * b_vec)
            damping_code += f'rayleigh {x_sol[0]} 0.0 0.0 {x_sol[1]}'

        if damping_type == "stiffness":
            assert isinstance(damping["ratio"], float)
            assert isinstance(damping["period"], float)
            damping_code += f'rayleigh 0.00 0.00 0.00 {damping["ratio"] * damping["period"] / np.pi}'

        if damping_type == "modal":
            num_modes = damping["num_modes"]
            damping_ratio = damping["ratio"]
            ops.eigen(num_modes)
            assert isinstance(damping_ratio, float)
            damping_code += f'modalDampingQ {damping_ratio}'

        if damping_type == "modal+stiffness":

            # stiffness-proportional contribution
            assert isinstance(damping["ratio_stiffness"], float)
            assert isinstance(damping["period"], float)
            alpha_1 = damping["ratio_stiffness"] * damping["period"] / np.pi
            damping_code += f'rayleigh 0.00 0.00 0.00 {alpha_1}'
            damping_code += '\n'

            num_modes = damping["num_modes"]
            damping_ratio = damping["ratio_modal"]
            damping_code += f'set ratio_modal {damping_ratio}'
            damping_code += '\n'
            damping_code += f'set num_modes {int(num_modes)}'
            damping_code += '\n'
            damping_code += f'set damping_ratio {damping_ratio}'
            damping_code += '\n'
            damping_code += 'set omega_squareds [eigen $num_modes]'
            damping_code += '\n'
            damping_code += f'set alpha_1 {alpha_1}'
            damping_code += '\n'
            damping_code += 'set values [list]'
            damping_code += '\n'
            damping_code += 'foreach omega_squared $omega_squareds {'
            damping_code += '\n'
            damping_code += '  set result [expr {$ratio_modal - $alpha_1 * $omega_squared / 2.00}]'
            damping_code += '\n'
            damping_code += '    if {$result < 0} {'
            damping_code += '\n'
            damping_code += '    lappend values 0.00'
            damping_code += '\n'
            damping_code += '    } else {'
            damping_code += '\n'
            damping_code += '      lappend values $result'
            damping_code += '\n'
            damping_code += '  }'
            damping_code += '\n'
            damping_code += '}'
            damping_code += '\n'
            damping_code += 'modalDampingQ {*}$values'
            damping_code += '\n'

        damping_code += '\n'

        data = pkgutil.get_data(__name__, 'th_analysis.tcl')
        assert data is not None
        contents = data.decode()

        contents = contents.replace('{%NUMBERER%}', str(NUMBERER))
        contents = contents.replace('{%CONSTRAINTS%}', ' '.join([str(x) for x in CONSTRAINTS]))
        contents = contents.replace('{%SYSTEM%}', self.settings.solver)
        contents = contents.replace('{%DAMPING CODE%}', damping_code)
        contents = contents.replace('{%time_increment%}', str(time_increment))
        contents = contents.replace('{%output_dir%}', output_dir)
        contents = contents.replace('{%skip_steps%}', str(skip_steps))
        contents = contents.replace('{%target_timestamp%}', str(target_timestamp))
        contents = contents.replace('{%collapse_drift%}', str(collapse_drift))
        contents = contents.replace('{%drift_nodes%}', ' '.join([str(x) for x in drift_nodes]))
        contents = contents.replace('{%drift_heights%}', ' '.join([str(x) for x in drift_heights]))
        # contents = contents.replace('{%%}', )
        # contents = contents.replace('{%%}', )
        # contents = contents.replace('{%%}', )

        contents_list = contents.splitlines()

        script_append(tcl_script, contents_list)
    
    def run(
            self,
            analysis_time_increment: float,
            ag_vec_x: Optional[nparr],
            ag_vec_y: Optional[nparr],
            ag_vec_z: Optional[nparr],
            ag_vec_time_incr: float,
            finish_time: float = 0.00,
            skip_steps: int = 1,
            damping: dict[str, Optional[Union[str, float, int, list[float]]]] = {"type": None},
            print_progress: bool = True,
            drift_check: float = 0.00,
            time_limit: Optional[float] = None
    ) -> dict[str, Union[int, str, float]]:
        """
        Run the time-history analysis

        Arguments:
            ag_vec_x, y, z: 1-D numpy arrays containing the fixed-step
                            ground acceleration records.
            ag_vec_time_incr:   The corresponding time increment
            finish_time: Specify a target time (s) to stop the analysis
                         the default value of 0.00 means that it will
                         run for the entire duration of the files.
            damping: Can be any of:
                     {'type': None},
                     {'type': 'rayleigh', 'ratio': r, 'periods': [t1, t2]},
                     {'type': 'stiffness', 'ratio': r, 'period': t1}
                     {'type': 'modal', 'num_modes': n, 'ratio': r}
                     {'type': 'modal+stiffness', 'num_modes': n,
                     'ratio_modal': r, 'period': t1,
                     'ratio_stiffness': r}
            print_progress: Controls whether the current time is printed out
            drift_check: If a value other than 0.00 is specified, the
              analysis stops if the drift ratio in each orthogonal
              direction exceeds the specified value. Levels that have
              no parent nodes are excempt from this check.
            time_limit: Maximum analysis time allowed, in hours.
              When reached, the anlysis is interrupted.

        """

        self._init_results()
        self.log("Running TH analysis")

        self.log(f'Model Name: {self.mdl.name}')

        nodes = self.mdl.list_of_all_nodes()
        # note: only runs the first load case provided.
        # th should not have load cases.
        # will be fixed in the future.
        case_name = list(self.load_cases.keys())[0]
        self.log(f'Case Name: {case_name}')
        pnodes = self.load_cases[case_name].parent_nodes
        nodes.extend(pnodes.values())
        elastic_elems = [
            elm
            for elm in self.mdl.list_of_specific_element(element.ElasticBeamColumn)
            if not elm.visibility.skip_opensees_definition
        ]
        disp_elems = [
            elm
            for elm in self.mdl.list_of_specific_element(element.DispBeamColumn)
            if not elm.visibility.skip_opensees_definition
        ]
        truss_elems = [
            elm
            for elm in self.mdl.list_of_specific_element(element.TrussBar)
            if not elm.visibility.skip_opensees_definition
        ]
        line_elems = elastic_elems + disp_elems + truss_elems
        zerolength_elems = self.mdl.list_of_specific_element(element.ZeroLength)

        damping_type = damping.get("type")
        self.log(f'Damping Type: {damping_type}')

        nss = []
        if ag_vec_x is not None:
            nss.append(len(ag_vec_x))
        if ag_vec_y is not None:
            nss.append(len(ag_vec_y))
        if ag_vec_z is not None:
            nss.append(len(ag_vec_z))

        num_gm_points = np.min(np.array(nss))
        duration = num_gm_points * ag_vec_time_incr

        self.log(f'Ground Motion Duration: {duration:.2f} s')

        t_vec = np.linspace(
            0.00, ag_vec_time_incr * num_gm_points, num_gm_points
        )
        if ag_vec_x is not None:
            self.a_g[0] = np.column_stack((t_vec, ag_vec_x[:num_gm_points]))  # type: ignore
        else:
            self.a_g[0] = np.column_stack((t_vec, np.zeros(len(t_vec))))
        if ag_vec_y is not None:
            self.a_g[1] = np.column_stack((t_vec, ag_vec_y[:num_gm_points]))  # type: ignore
        else:
            self.a_g[1] = np.column_stack((t_vec, np.zeros(len(t_vec))))
        if ag_vec_z is not None:
            self.a_g[2] = np.column_stack((t_vec, ag_vec_z[:num_gm_points]))  # type: ignore
        else:
            self.a_g[2] = np.column_stack((t_vec, np.zeros(len(t_vec))))

        if finish_time == 0.00:
            target_timestamp = duration
        else:
            target_timestamp = finish_time

        self.log('')
        self.log("Defining model in OpenSees")
        self._to_opensees_domain(case_name)

        # gravity analysis
        self.log("Defining loads")
        self._define_loads(case_name, factor=0.1)

        self.log("Starting gravity analysis (G)")
        self._run_gravity_analysis(num_steps=10)
        self.log("Gravity analysis finished successfully")
        n_steps_success = 0
        self._read_opensees_results(
            case_name,
            n_steps_success,
            nodes,
            line_elems,
            zerolength_elems,
        )

        self.log("")
        self.log("Starting transient analysis")

        ops.wipeAnalysis()
        ops.loadConst("-time", 0.0)
        curr_time = 0.00
        self.time_vector.append(curr_time)

        ops.numberer(NUMBERER)
        ops.constraints(*CONSTRAINTS)
        self.log(f"Setting system solver to {self.settings.solver}")
        ops.system(self.settings.solver)

        if damping_type == "rayleigh":
            self.log("Using Rayleigh damping")
            assert isinstance(damping["periods"], list)
            assert isinstance(damping["periods"][0], float)
            w_i = 2 * np.pi / damping["periods"][0]
            zeta_i = damping["ratio"]
            assert isinstance(damping["periods"][1], float)
            w_j = 2 * np.pi / damping["periods"][1]
            zeta_j = damping["ratio"]
            a_mat: nparr = np.array([[1 / w_i, w_i], [1 / w_j, w_j]])
            b_vec: nparr = np.array([zeta_i, zeta_j])
            x_sol: nparr = np.linalg.solve(a_mat, 2 * b_vec)
            ops.rayleigh(x_sol[0], 0.0, 0.0, x_sol[1])
            # https://portwooddigital.com/2020/11/08/rayleigh-damping-coefficients/
            # --thanks, prof. Scott

        if damping_type == "stiffness":
            self.log("Using stiffness proportional damping")
            assert isinstance(damping["ratio"], float)
            assert isinstance(damping["period"], float)
            ops.rayleigh(
                0.00, 0.0, 0.0, damping["ratio"] * damping["period"] / np.pi
            )

        if damping_type == "modal":
            # tags = ops.getNodeTags()
            # num_nodes = len(tags) - 4
            # num_modeshapes = 3*num_nodes
            # self.print(len(tags))

            self.log("Using modal damping")

            num_modes = damping["num_modes"]
            # num_modes = num_modeshapes
            damping_ratio = damping["ratio"]
            self.log("Running eigenvalue analysis" f" with {num_modes} modes")
            ops.eigen(num_modes)
            # ops.systemSize()
            self.log("Eigenvalue analysis finished")
            assert isinstance(damping_ratio, float)
            ops.modalDampingQ(damping_ratio)
            self.log(f"{damping_ratio*100.00:.2f}% " "modal damping defined")

        if damping_type == "modal+stiffness":

            self.log("Using modal+stiffness damping")

            # stiffness-proportional contribution
            assert isinstance(damping["ratio_stiffness"], float)
            assert isinstance(damping["period"], float)
            assert isinstance(damping["ratio_modal"], float)
            alpha_1 = damping["ratio_stiffness"] * damping["period"] / np.pi
            ops.rayleigh(
                0.00,
                0.0,
                0.0,
                alpha_1,
            )
            self.log("modal+stiffness damping defined")

            # modal damping
            num_modes = damping["num_modes"]
            assert isinstance(num_modes, int)
            damping_ratio = damping["ratio_modal"]
            assert isinstance(damping_ratio, float)
            self.log("Running eigenvalue analysis" f" with {num_modes} modes")
            omega_squareds = np.array(ops.eigen(num_modes))
            self.log("Eigenvalue analysis finished")
            damping_vals = damping["ratio_modal"] - alpha_1*omega_squareds/2.00
            damping_vals[damping_vals < 0.00] = 0.00
            ops.modalDampingQ(*damping_vals)

        ops.test("EnergyIncr", 1.0e-6, 100, 0)
        ops.integrator('TRBDF2')
        ops.algorithm("KrylovNewton")
        # ops.algorithm("KrylovNewton", 'initial', 'initial')
        ops.analysis("Transient")

        define_lateral_load_pattern(
            ag_vec_x, ag_vec_y, ag_vec_z, ag_vec_time_incr
        )

        num_subdiv = 0
        num_times = 0
        total_step_count = 0
        analysis_failed = False
        algorithm_idx = 0

        scale = (
            1.0, 1.0e-1, 1.0e-2, 1.0e-3,
            1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7,
            1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1e-12
        )
        tols = [1.0e-12]*13
        algorithms = (('KrylovNewton', ), ('KrylovNewton', 'initial'))

        # progress bar
        if print_progress:
            pbar = tqdm(
                total=target_timestamp,
                bar_format='{percentage:3.0f}%|{bar:25}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                )
            pbar.set_postfix({'time': f'{curr_time:.4f}/{target_timestamp:.2f}'})
            pbar.update(curr_time)
        else:
            pbar = None

        # store the start time. Used to add log entries on the status
        # of the analysis every 5 minutes.
        start_time = perf_counter()
        the_time = start_time

        try:

            while curr_time + common.EPSILON < target_timestamp:

                if analysis_failed:
                    break

                ops.test(
                    "EnergyIncr", tols[num_subdiv], 200, 3, 2)
                ops.algorithm(*algorithms[algorithm_idx])
                check = ops.analyze(
                    1, analysis_time_increment * scale[num_subdiv]
                )
                total_step_count += 1

                if check != 0:
                    # analysis failed
                    if num_subdiv == len(scale) - 1:
                        # can't subdivide any further
                        self.print("===========================")
                        self.print("Analysis failed to converge")
                        self.print("===========================")
                        if self.logger:
                            self.logger.warning(
                                f"Analysis failed at time {curr_time:.5f}"
                                " and cannot continue."
                            )
                        analysis_failed = True
                        break

                    # otherwise, we can still try
                    if algorithm_idx != len(algorithms) - 1:
                        algorithm_idx += 1
                    else:
                        algorithm_idx = 0
                        num_subdiv += 1
                        # how many times to run with reduced step size
                        num_times = 50
                else:
                    # analysis was successful
                    prev_time = curr_time
                    curr_time = float(ops.getTime())

                    # progress bar
                    if pbar is not None:
                        pbar.set_postfix({'time': f'{curr_time:.4f}/{target_timestamp:.2f}'})
                        pbar.update(curr_time - prev_time)
                    # log entry for analysis status
                    if perf_counter() - the_time > 5.00*60.00:  # 5 min
                        the_time = perf_counter()
                        # total time running
                        running_time = the_time - start_time
                        # th seconds ran is `curr_time`
                        remaining_time = target_timestamp - curr_time
                        average_speed = curr_time / running_time  # th [s] / real [s]
                        # estimated remaining real time to finish [s]
                        est_remaining_dur = remaining_time / average_speed
                        self.log(f'Analysis status: {{curr: {curr_time:.2f}, '
                                 f'target: {target_timestamp:.2f}, '
                                 f'num_subdiv: {num_subdiv}, '
                                 f'~ {est_remaining_dur:.0f} s to finish}}')

                    if num_times != 0:
                        num_times -= 1

                    if total_step_count % skip_steps == 0:
                        n_steps_success += 1
                        self._read_opensees_results(
                            case_name,
                            n_steps_success,
                            nodes,
                            line_elems,
                            zerolength_elems,
                        )
                        self.time_vector.append(curr_time)

                    algorithm_idx = 0
                    if num_subdiv != 0:
                        if num_times == 0:
                            num_subdiv -= 1
                            num_times = 50
                    if drift_check > 0.00 and pnodes:
                        peak_drift = 0.00
                        for lvl_idx, pnode in pnodes.items():
                            if lvl_idx == 1:
                                drift = np.max(np.abs(
                                    np.array(
                                        self.results[case_name]
                                        .node_displacements[
                                            pnode.uid][
                                                n_steps_success][0:2])
                                    /pnode.coords[2]))
                            else:
                                drift = np.max(np.abs(
                                    (
                                        np.array(
                                            self.results[case_name]
                                            .node_displacements[
                                                pnodes[lvl_idx].uid][
                                                    n_steps_success][0:2])
                                        - np.array(
                                            self.results[case_name]
                                            .node_displacements[
                                                pnodes[lvl_idx-1].uid][
                                                    n_steps_success][0:2]))
                                    /(pnodes[lvl_idx].coords[2]
                                      - pnodes[lvl_idx-1].coords[2])))
                            if drift > peak_drift:
                                peak_drift = drift
                        if peak_drift > drift_check:
                            # terminate analysis
                            if self.logger:
                                self.logger.warning(
                                    "Analysis stopped at time"
                                    f" {curr_time:.5f}"
                                    " due to excessive drift."
                                )
                            analysis_failed = True
                            break
                        if time_limit and (perf_counter() - start_time) > time_limit*3600.00:
                            self.print(
                                "Analysis interrupted due to time limit.")
                            if self.logger:
                                self.logger.warning(
                                    "Analysis interrupted at time"
                                    f" {curr_time:.5f}"
                                    " because the time limit of"
                                    f" {time_limit}h was reached."
                                )
                            analysis_failed = True
                            # self.save_state()
                            break

        except KeyboardInterrupt:
            self.print("Analysis interrupted")
            if self.logger:
                self.logger.warning("Analysis interrupted")

        # remove the progress bar
        if pbar is not None:
            pbar.close()

        self.log("Analysis finished")
        metadata: dict[str, Union[int, str, float]] = {
            "successful steps": n_steps_success,
            "analysis_finished_successfully": not analysis_failed,
        }
        self.results[case_name].n_steps_success = len(self.time_vector)
        if self.settings.pickle_results:
            self._write_results_to_disk()

        return metadata

    # def save_state(self):
    #     if self.settings.output_directory:
    #         # remove earlier saved state if it exists
    #         path = os.path.join(self.settings.output_directory, 'saved_state')
    #         if os.path.isdir(path):
    #             shutil.rmtree(path)
    #         # create the directory to save the state
    #         os.makedirs(path)
    #         ops.database('File', '/tmp/')
    #         ops.save(0)

    def plot_node_displacement_history(
        self, case_name, node, direction, plotly=False
    ):
        """
        Plots the displacement history of the specified node.

        """

        time_vec = self.time_vector
        uid = node.uid
        results = []
        for k in range(self.results[case_name].n_steps_success):  # type:ignore
            results.append(
                self.results[case_name].node_displacements[uid][k][direction]
            )
        vals: nparr = np.array(results)
        if plotly:
            general_2d.line_plot_interactive(
                f"Node {uid} displacement history",
                time_vec,
                vals,
                "line",
                "Time",
                "s",
                ".3f",
                "Rel. Displacement",
                "in",
                ".4f",
            )
        else:
            plt.figure()
            plt.plot(time_vec, vals, "k")
            plt.xlabel("Time (s)")
            plt.ylabel("Displacement (in)")
            plt.show()


@dataclass
class ModalResponseSpectrumAnalysis:
    """
    Modal response spectrum analysis.

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
        Run the modal response spectrum analysis.

        """

        spectrum_ifun = interp1d(self.periods, self.spectrum, kind="linear")
        anl = ModalAnalysis(
            self.mdl,
            {self.load_case.name: self.load_case},
            num_modes=self.num_modes,
        )
        anl.settings.pickle_results = False
        anl.settings.store_fiber = False
        anl.settings.store_forces = True
        anl.settings.store_reactions = False
        anl.run()
        case_name = self.load_case.name
        gammas, mstars, mtot = anl.modal_participation_factors(
            case_name, self.direction
        )
        periods = anl.results[case_name].periods
        if self.mdl.settings.imperial_units:
            g_const = common.G_CONST_IMPERIAL
        else:
            g_const = common.G_CONST_SI
        vb_modal = np.zeros(self.num_modes)
        modal_q0 = np.zeros(self.num_modes)
        for i in range(self.num_modes):
            assert periods is not None
            vb_modal[i] = (
                (spectrum_ifun(periods[i])) * mstars[i] * mtot * g_const
            )
            modal_q0[i] = gammas[i] * (
                spectrum_ifun(periods[i])
                / (2.0 * np.pi / periods[i]) ** 2
                * g_const
            )
        self.modal_q0 = modal_q0
        self.vb_modal = vb_modal
        self.anl = anl

    def combined_node_disp(self, node_uid):
        """
        Returns the SRSS-combined node displacement of a node.

        """

        all_vals = []
        assert self.anl is not None
        for i in range(self.num_modes):
            assert self.modal_q0 is not None
            vals = (
                np.array(
                    self.anl.results[self.load_case.name].node_displacements[
                        node_uid
                    ][i]
                )
                * self.modal_q0[i]
            )
            all_vals.append(vals)
        all_vals_np: nparr = np.column_stack(all_vals)
        return np.sqrt(np.sum(all_vals_np**2, axis=1))

    def combined_node_disp_diff(self, node_i_uid, node_j_uid):
        """
        Returns the SRSS-combined displacement difference between two
        nodes.

        """

        all_vals = []
        assert self.anl is not None
        for i in range(self.num_modes):
            assert self.modal_q0 is not None
            vals_i = (
                np.array(
                    self.anl.results[self.load_case.name].node_displacements[
                        node_i_uid
                    ][i]
                )
                * self.modal_q0[i]
            )
            assert self.modal_q0 is not None
            vals_j = (
                np.array(
                    self.anl.results[self.load_case.name].node_displacements[
                        node_j_uid
                    ][i]
                )
                * self.modal_q0[i]
            )
            vals = vals_i - vals_j
            all_vals.append(vals)
        all_vals_np: nparr = np.column_stack(all_vals)
        return np.sqrt(np.sum(all_vals_np**2, axis=1))

    def combined_basic_forces(self, element_uid):
        """
        Returns the SRSS-combined basic forces of a line element.

        """

        all_vals = []
        assert self.anl is not None
        for i in range(self.num_modes):
            assert self.modal_q0 is not None
            vals = (
                np.array(
                    self.anl.results[self.load_case.name].element_forces[
                        element_uid
                    ][i]
                )
                * self.modal_q0[i]
            )
            all_vals.append(vals)
        all_vals_np: nparr = np.column_stack(all_vals)
        return np.sqrt(np.sum(all_vals_np**2, axis=1))
