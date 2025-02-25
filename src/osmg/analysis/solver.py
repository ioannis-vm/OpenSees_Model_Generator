"""Defines Analysis objects."""

from __future__ import annotations

import logging
import platform
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from osmg.analysis.recorders import ElementRecorder, NodeRecorder
from osmg.core.common import NDF, NDM
from osmg.core.osmg_collections import BeamColumnAssembly
from osmg.model_objects.element import (
    Bar,
    DispBeamColumn,
    ElasticBeamColumn,
    TwoNodeLink,
    ZeroLength,
)

try:
    import opensees.openseespy as ops
except (ImportError, ModuleNotFoundError):
    import openseespy.opensees as ops

if TYPE_CHECKING:
    from osmg.analysis.load_case import LoadCase
    from osmg.analysis.recorders import Recorder
    from osmg.core.model import Model
    from osmg.model_objects.uniaxial_material import UniaxialMaterial


@dataclass()
class AnalysisSettings:
    """Analysis settings object."""

    result_directory: str | None = field(default=None)
    log_file_name: str = field(default='log.txt')
    log_level: int = field(default=logging.DEBUG)
    solver: str = field(default='UmfPack')
    restrict_dof: tuple[bool, ...] | None = field(default=None)
    disable_default_recorders: bool = field(default=False)
    constraints: tuple[str] = ('Transformation',)
    numberer: Literal['RCM'] = 'RCM'
    system: Literal['Umfpack'] = 'Umfpack'
    ignore_by_tag: set[str] = field(default_factory=set)


@dataclass()
class StaticAnalysisSettings(AnalysisSettings):
    """Static analysis settings."""

    num_steps: int = field(default=1)


@dataclass(repr=False)
class Analysis:
    """Parent analysis class."""

    settings: AnalysisSettings = field(default_factory=AnalysisSettings)
    _logger: logging.Logger = field(init=False)
    recorders: dict[str, Recorder] = field(default_factory=dict)
    _defined_materials: list[int] = field(default_factory=list)

    def initialize_logger(self) -> None:
        """
        Initialize the logger of the analysis.

        Analysis objects are instantiated with a default configuration
        to make it possible for each load case to initialize a
        corresponding analysis. Their configuration needs to be
        updated before running. Part of that is to set the log file
        path.

        Raises:
          ValueError: If the log file path is still None.
        """
        if (
            self.settings.log_file_name is None
            or self.settings.result_directory is None
        ):
            msg = 'Analysis log file is required.'
            raise ValueError(msg)
        logging.basicConfig(
            filename=Path(self.settings.result_directory)
            / self.settings.log_file_name,
            filemode='w',
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
        )
        self._logger: logging.Logger = logging.getLogger('OpenSees_Model_Generator')
        self._logger.setLevel(self.settings.log_level)

        # Log system info
        self._logger.info('Analysis initialized.')
        os_system = platform.system()
        self._logger.info(f'Platform: {os_system}.')
        if os_system == 'Linux':
            self._logger.info(f'Hostname: {socket.gethostname()}.')
        self._logger.info(f'Python Version: {sys.version}.')

    def log(
        self,
        msg: str,
        level: Literal['debug', 'info', 'warning', 'error', 'critical'] = 'info',
    ) -> None:
        """
        Add a message to the log file.

        Raises:
          ValueError: If an invalid logging level is specified.
        """
        if level == 'info':
            self._logger.info(msg)
        elif level == 'debug':
            self._logger.debug(msg)
        elif level == 'warning':
            self._logger.warning(msg)
        elif level == 'error':
            self._logger.error(msg)
        elif level == 'critical':
            self._logger.critical(msg)
        else:
            msg = f'Invalid logging level: {level}'
            raise ValueError(msg)

    @staticmethod
    def opensees_instantiate(model: Model) -> None:
        """Instantiate the model in OpenSees."""
        ops.wipe()
        ops.model(
            'basic',
            '-ndm',
            NDM[model.dimensionality],
            '-ndf',
            NDF[model.dimensionality],
        )

    def opensees_define_material(self, material: UniaxialMaterial) -> None:
        """Recursively define materials with predecessors."""
        if material.uid not in self._defined_materials:
            while (
                hasattr(material, 'predecessor')
                and material.predecessor.uid not in self._defined_materials
            ):
                self.opensees_define_material(material.predecessor)
            ops.uniaxialMaterial(*material.ops_args())
            self._defined_materials.append(material.uid)

    def opensees_define_nodes(self, model: Model) -> None:
        """Define the nodes of the model in OpenSees."""
        for uid, node in model.get_all_nodes(self.settings.ignore_by_tag).items():
            ops.node(uid, *node.coordinates)

    def opensees_define_elements(
        self,
        model: Model,
    ) -> None:
        """Define elements."""
        elastic_beamcolumn_elements: list[ElasticBeamColumn] = []
        bar_elements: list[Bar] = []
        two_node_link_elements: list[TwoNodeLink] = []
        zerolength_elements: list[ZeroLength] = []
        unsupported_element_types: list[str] = []

        # Note: Materials are defined on an as-needed basis.  We keep
        # track of defined materials in `self._defined_materials` to
        # avoid trying to define the same material twice. After
        # elements are defined, we clear that list, to start with an
        # empty list in subsequent model definitions.

        components = model.components.values()
        for component in components:
            if component.tags & self.settings.ignore_by_tag:
                continue
            elements = component.elements
            for element in elements.values():
                if isinstance(element, ElasticBeamColumn):
                    elastic_beamcolumn_elements.append(element)
                elif isinstance(element, Bar):
                    bar_elements.append(element)
                elif isinstance(element, TwoNodeLink):
                    two_node_link_elements.append(element)
                elif isinstance(element, ZeroLength):
                    zerolength_elements.append(element)
                else:
                    unsupported_element_types.append(element.__class__.__name__)

        if unsupported_element_types:
            print(  # noqa: T201
                f'WARNING: Unsupported element types found: {set(unsupported_element_types)}'
            )

        self.opensees_define_elastic_beamcolumn_elements(
            model, elastic_beamcolumn_elements
        )
        self.opensees_define_bar_elements(bar_elements)
        self.opensees_define_two_node_link_elements(two_node_link_elements)
        self.opensees_define_zerolength_elements(zerolength_elements)

        # clear defined materials
        self._defined_materials = []

    @staticmethod
    def opensees_define_elastic_beamcolumn_elements(
        model: Model,
        elements: list[ElasticBeamColumn],
    ) -> None:
        """
        Define elastic beamcolumn elements.

        Raises:
          TypeError: If the model type is invalid.
        """
        for element in elements:
            if element.visibility.skip_opensees_definition:
                continue
            ops.geomTransf(*element.geomtransf.ops_args())
            if model.dimensionality == '2D Frame':
                ops.element(*element.ops_args_2d())
            elif model.dimensionality == '3D Frame':
                ops.element(*element.ops_args())
            else:
                msg = f'Invalid model dimensionality: `{model.dimensionality}`.'
                raise TypeError(msg)

    def opensees_define_bar_elements(self, elements: list[Bar]) -> None:
        """Define bar elements."""
        for element in elements:
            self.opensees_define_material(element.material)
            ops.element(*element.ops_args())

    def opensees_define_two_node_link_elements(
        self, elements: list[TwoNodeLink]
    ) -> None:
        """Define TwoNodeLink elements."""
        for element in elements:
            for material in element.materials:
                self.opensees_define_material(material)
            ops.element(*element.ops_args())

    def opensees_define_zerolength_elements(
        self, elements: list[ZeroLength]
    ) -> None:
        """Define Zerolength elements."""
        for element in elements:
            for material in element.materials:
                self.opensees_define_material(material)
            ops.element(*element.ops_args())

    def opensees_define_node_restraints(
        self, model: Model, load_case: LoadCase
    ) -> None:
        """Define node restraints."""
        ndf = NDF[model.dimensionality]

        for uid, support in load_case.fixed_supports.items():
            fix = []
            for i in range(ndf):
                if support[i] is True or (
                    self.settings.restrict_dof
                    and self.settings.restrict_dof[i] is True
                ):
                    fix.append(True)
                else:
                    fix.append(False)
            if True in fix:
                ops.fix(uid, *[int(x) for x in fix])

        nodes = model.get_all_nodes()
        elastic_materials = {}
        for uid, support in load_case.elastic_supports.items():
            assert len(support) == ndf
            node = nodes[uid]
            # for each direction.
            material_uids_for_this_support = []
            for value in support:
                # define material if needed.
                if value not in elastic_materials:
                    material_uid = next(model.uid_generator.MATERIAL)
                    elastic_materials[value] = material_uid
                    ops.uniaxialMaterial('Elastic', material_uid, value)
                else:
                    material_uid = elastic_materials[value]
                material_uids_for_this_support.append(material_uid)
            # define a node at the same location.
            new_node_uid = next(model.uid_generator.NODE)
            ops.node(new_node_uid, *node.coordinates)
            # fix that node.
            ops.fix(new_node_uid, *([1] * ndf))
            # define a zerolength element connecting the two nodes.
            ops.element(
                'zeroLength',
                next(model.uid_generator.ELEMENT),
                uid,
                new_node_uid,
                '-mat',
                *material_uids_for_this_support,
                '-dir',
                *range(1, ndf + 1),
            )

    @staticmethod
    def opensees_define_node_constraints(model: Model, load_case: LoadCase) -> None:
        """
        Define node constraints.

        Raises:
          ValueError: If the model dimensionality is not supported.
        """
        if not load_case.rigid_diaphragm:
            return

        for parent_node_uid, children_node_uids in load_case.rigid_diaphragm.items():
            if model.dimensionality in {'3D Frame', '3D Truss'}:
                ops.rigidDiaphragm(3, parent_node_uid, *children_node_uids)
            elif model.dimensionality in {'2D Frame', '2D Truss'}:
                for child_node_uid in children_node_uids:
                    ops.equalDOF(parent_node_uid, child_node_uid, 1)
            else:
                msg = 'Unsupported model dimensionality: {model.dimensionality}'
                raise ValueError(msg)

    def opensees_define_model(self, model: Model, load_case: LoadCase) -> None:
        """Define the model in OpenSees."""
        self.opensees_instantiate(model)
        self.opensees_define_nodes(model)
        self.opensees_define_elements(model)
        self.opensees_define_node_restraints(model, load_case)
        self.opensees_define_node_constraints(model, load_case)
        if not self.settings.disable_default_recorders:
            self.define_default_recorders(model)
        self.opensees_define_recorders()

    def opensees_define_loads(
        self,
        model: Model,
        load_case: LoadCase,
        amplification_factor: float = 1.00,
        time_series_tag: int = 1,
        pattern_tag: int = 1,
    ) -> None:
        """
        Define loads in OpenSees.

        Defines a `Linear` `timeSeries` with a `Plain` `pattern` and
        assigns all loads in the `laodase`.

        Raises:
          TypeError: If the model type is invalid.
        """
        ops.timeSeries('Linear', time_series_tag)
        ops.pattern('Plain', pattern_tag, time_series_tag)

        # Point load on nodes
        for node_uid, point_load in load_case.load_registry.nodal_loads.items():  # type: ignore
            ops.load(node_uid, *(v * amplification_factor for v in point_load))

        # UDL on components
        for (
            component_uid,
            global_udl,
        ) in load_case.load_registry.component_udl.items():  # type: ignore
            component = model.components[component_uid]
            if component.tags & self.settings.ignore_by_tag:
                continue
            assert isinstance(component, BeamColumnAssembly)
            local_udls = component.calculate_element_udl(global_udl)
            for beamcolumn_element_uid, local_udl in local_udls.items():
                if model.dimensionality == '3D Frame':
                    ops.eleLoad(
                        '-ele',
                        beamcolumn_element_uid,
                        '-type',
                        '-beamUniform',
                        local_udl[1] * amplification_factor,
                        local_udl[2] * amplification_factor,
                        local_udl[0] * amplification_factor,
                    )
                elif model.dimensionality == '2D Frame':
                    ops.eleLoad(
                        '-ele',
                        beamcolumn_element_uid,
                        '-type',
                        '-beamUniform',
                        local_udl[1] * amplification_factor,
                        local_udl[0] * amplification_factor,
                    )
                else:
                    msg = f'Invalid model dimensionality: `{model.dimensionality}`.'
                    raise TypeError(msg)

    @staticmethod
    def opensees_define_mass(
        load_case: LoadCase, amplification_factor: float = 1.00
    ) -> None:
        """Define mass in OpenSees."""
        for node_uid, point_mass in load_case.mass_registry.items():  # type: ignore
            ops.mass(node_uid, *(v * amplification_factor for v in point_mass))

    def define_default_recorders(self, model: Model) -> None:
        """
        Create a set of default recorders.

        Does not define them in  OpenSees.

        Raises:
          ValueError: If the results directory is unspecified.
        """
        ndf = NDF[model.dimensionality]
        store_dir = self.settings.result_directory
        if store_dir is None:
            msg = 'Please specify a result directory in the analysis options.'
            raise ValueError(msg)
        node_recorder = NodeRecorder(
            uid_generator=model.uid_generator,
            file_name='node_displacements',
            recorder_type='Node',
            nodes=tuple(model.get_all_nodes().keys()),
            dofs=tuple(v + 1 for v in range(ndf)),
            response_type='disp',
            number_of_significant_digits=6,
            output_time=True,
        )
        self.recorders['default_node'] = node_recorder
        node_reaction_recorder = NodeRecorder(
            uid_generator=model.uid_generator,
            file_name='node_reactions',
            recorder_type='Node',
            nodes=tuple(model.get_all_nodes().keys()),
            dofs=tuple(v + 1 for v in range(ndf)),
            response_type='reaction',
            number_of_significant_digits=6,
            output_time=True,
        )
        self.recorders['default_node_reaction'] = node_reaction_recorder

        applicable_elements = []
        components = model.components.values()
        for component in components:
            if component.tags & self.settings.ignore_by_tag:
                continue
            elements = component.elements
            for element in elements.values():
                if isinstance(element, (ElasticBeamColumn, DispBeamColumn, Bar)):
                    if element.visibility.skip_opensees_definition:
                        continue
                    if (
                        isinstance(element, Bar)
                        and element.transf_type == 'Corotational'
                    ):
                        # Crurently `localForce` is not a valid
                        # recorder argument for corotational truss
                        # elements.
                        continue
                    applicable_elements.append(element.uid)

        element_force_recorder = ElementRecorder(
            uid_generator=model.uid_generator,
            file_name='basic_forces',
            recorder_type='Element',
            elements=tuple(applicable_elements),
            element_arguments=('localForce',),
            number_of_significant_digits=6,
            output_time=True,
        )
        self.recorders['default_basic_force'] = element_force_recorder

    def opensees_define_recorders(self) -> None:
        """Define recorders in OpenSees."""
        assert self.settings.result_directory is not None
        for recorder in self.recorders.values():
            # Update file name to include the base directory
            recorder.file_name = str(
                (Path(self.settings.result_directory) / recorder.file_name).resolve()
            )
            ops.recorder(*recorder.ops_args())

    def run(self, model: Model, load_case: LoadCase) -> None:  # noqa: PLR6301
        """Run the analysis."""
        msg = 'Subclasses should implement this.'
        raise NotImplementedError(msg)


@dataclass(repr=False)
class StaticAnalysis(Analysis):
    """Static analysis."""

    settings: StaticAnalysisSettings = field(default_factory=StaticAnalysisSettings)

    def run(self, model: Model, load_case: LoadCase) -> None:
        """Run the analysis."""
        self.initialize_logger()

        self.log('Running a static analysis.')

        self.log('Defining model in OpenSees.')
        self.opensees_define_model(model, load_case)
        self.log('Defining loads in OpenSees.')
        self.opensees_define_loads(model, load_case)

        self.log('Setting up analysis.')
        ops.system(self.settings.system)
        ops.numberer(self.settings.numberer)
        ops.constraints(*self.settings.constraints)
        ops.test('EnergyIncr', 1.0e-8, 20, 3)
        ops.algorithm('Linear')
        ops.integrator('LoadControl', 1.00 / self.settings.num_steps)
        ops.analysis('Static')

        self.log('Analyzing.')
        out = ops.analyze(self.settings.num_steps)
        assert out == 0, 'Analysis failed.'

        ops.wipe()
        self.log('Analysis finished.')


@dataclass()
class ModalAnalysisSettings(AnalysisSettings):
    """Modal analysis settings."""

    num_modes: int = field(default=3)


@dataclass(repr=False)
class ModalAnalysis(Analysis):
    """Modal analysis."""

    settings: ModalAnalysisSettings = field(default_factory=ModalAnalysisSettings)
    periods: list[float] = field(default_factory=list)

    def run(self, model: Model, load_case: LoadCase) -> None:  # noqa: C901  # type: ignore
        """
        Run the modal analysis.

        Raises:
          ValueError: If a solver known to not work with `eigen` is
           specified.
        """
        self.initialize_logger()
        self.log('Running a modal analysis.')

        ndf = NDF[model.dimensionality]

        if self.settings.disable_default_recorders is True:
            self.settings.disable_default_recorders = False
            msg = (
                'Default recorders are required for '
                'modal analysis and were force-enabled.'
            )
            self.log(msg)
            print(msg)  # noqa: T201
            # TODO(JVM): turn into a warning.

        self.log('Defining model in OpenSees.')
        self.opensees_define_model(model, load_case)
        self.log('Defining mass in OpenSees.')
        self.opensees_define_mass(load_case)

        self.log('Setting up analysis.')
        if self.settings.system.lower() in {'sparsesym', 'sparsespd'}:
            msg = (
                f'{self.settings.solver} is unable '
                'to run a modal analysis. Use UmfPack.'
            )
            self.log(msg)
            raise ValueError(msg)

        ops.system(self.settings.system)
        ops.numberer(self.settings.numberer)
        ops.constraints(*self.settings.constraints)

        self.log('Analyzing.')
        lambda_values = np.array(ops.eigen(self.settings.num_modes))

        self.log('Calculating periods.')
        omega_values = np.sqrt(lambda_values)
        periods = 2.0 * np.pi / omega_values
        self.periods = periods
        self.log(f'Periods: {periods}')

        self.log('Retrieving node eigenvectors.')

        node_recorder = self.recorders['default_node']

        assert isinstance(node_recorder, NodeRecorder)
        nodes = node_recorder.nodes
        data = {}
        for mode in range(1, self.settings.num_modes + 1):
            for node in nodes:
                disp = ops.nodeEigenvector(node, mode)
                data[node, mode] = disp
        eigenvectors = pd.DataFrame(
            data.values(),
            index=pd.MultiIndex.from_tuples(data.keys(), names=['node', 'mode']),
            columns=range(1, ndf + 1),
        )
        eigenvectors.columns.name = 'dof'
        eigenvectors = eigenvectors.unstack(level='node')  # noqa: PD010
        eigenvectors.columns = eigenvectors.columns.reorder_levels(['node', 'dof'])
        node_order = eigenvectors.columns.get_level_values('node').unique()
        eigenvectors = eigenvectors.loc[:, node_order]
        self.log('Retrieved node eigenvectors.')

        basic_force_data = {}

        self.log('Obtaining basic forces for each mode.')
        # Recover basic forces
        self.log('   Wiping OpenSees domain.')
        ops.wipe()
        progress_bar = tqdm(
            range(1, self.settings.num_modes + 1),
            ncols=80,
            desc='Processing modes',
            unit='mode',
            position=0,
            leave=False,
        )
        for mode in range(1, self.settings.num_modes + 1):
            self.log(f'  Working on mode {mode}.')
            self.log('  Defining model in OpenSees.')
            self.opensees_define_model(model, load_case)
            mode_eigenvectors = eigenvectors.loc[mode, :].copy()
            # ignore the node-dof pairs with a fixed constraint.
            to_drop = []
            for uid, support in load_case.fixed_supports.items():
                for i, dof in enumerate(support):
                    if dof:
                        to_drop.append((uid, i + 1))
            mode_eigenvectors = mode_eigenvectors.drop(to_drop)
            if (
                bool((mode_eigenvectors.isna() | np.isinf(mode_eigenvectors)).any())
                is True
            ):
                self.log(
                    f'  NaNs or infs present in displacement data, skipping mode {mode}.'
                )
                continue

            self.log('  Setting up Sp constraints.')
            ops.timeSeries('Constant', 1)
            ops.pattern('Plain', 1, 1)

            for key, displacement in mode_eigenvectors.items():
                node, dof = key
                ops.sp(node, dof, displacement)

            self.log('  Setting up analysis.')
            ops.integrator('LoadControl', 0.0)
            ops.constraints('Transformation')
            ops.algorithm('Linear')
            ops.numberer(self.settings.numberer)
            ops.system(self.settings.system)
            ops.analysis('Static')

            self.log('  Analyzing.')
            out = ops.analyze(1)
            assert out == 0, 'Analysis failed.'
            # The recorders should have captured the results here.
            self.log('  Retrieving data from recorder output.')
            # TODO(JVM): read basic_force_data from the recorder, add
            # in the dict.

            # Verifying displacements are correct.
            for key, value in mode_eigenvectors.items():
                node_uid, dof = key
                assert np.allclose(ops.nodeDisp(node_uid, dof), value)

            self.log('   Wiping OpenSees domain.')
            # Doing this before reading the basic force recorder data
            # ensures the recorder's buffer will have been flushed.
            ops.wipe()

            basic_force_data[mode] = self.recorders['default_basic_force'].get_data()
            basic_force_data[mode].index.name = 'mode'
            basic_force_data[mode].index = [mode]

            progress_bar.update(1)

        progress_bar.close()

        self.log('Obtained basic forces for each mode.')
        self.log('Wiping OpenSees domain for the last time.')
        ops.wipe()
        self.log('Storing data.')
        self.recorders['default_node'].set_data(eigenvectors)
        self.recorders['default_basic_force'].set_data(
            pd.concat(basic_force_data.values(), axis=0)
        )
        self.log('Analysis finished.')
