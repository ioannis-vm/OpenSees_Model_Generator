"""Defines Analysis objects."""

from __future__ import annotations

import logging
import platform
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from osmg.analysis.recorders import NodeRecorder
from osmg.core.model import Model2D, Model3D
from osmg.core.osmg_collections import BeamColumnAssembly
from osmg.model_objects.element import ElasticBeamColumn

try:
    import opensees.openseespy as ops
except (ImportError, ModuleNotFoundError):
    import openseespy.opensees as ops

if TYPE_CHECKING:
    from osmg.analysis.load_case import LoadCase


@dataclass()
class AnalysisSettings:
    """Analysis settings object."""

    result_directory: str | None = field(default=None)
    log_file_name: str | None = field(default=None)
    log_level: int = field(default=logging.DEBUG)
    solver: str = field(default='UmfPack')
    restrict_dof: tuple[bool, ...] | None = field(default=None)
    disable_default_recorders: bool = field(default=False)


@dataclass(repr=False)
class Analysis:
    """Parent analysis class."""

    settings: AnalysisSettings = field(default_factory=AnalysisSettings)
    _logger: logging.Logger = field(init=False)

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
        self._logger.info(f'Platform: {os_system}')
        if os_system == 'Linux':
            self._logger.info(f'Hostname: {socket.gethostname()}')
        self._logger.info(f'Python Version: {sys.version}')

    def log(self, msg: str) -> None:
        """Add a message to the log file."""
        self._logger.info(msg)

    @staticmethod
    def opensees_instantiate(model: Model2D | Model3D) -> None:
        """
        Instantiate the model in OpenSees.

        Raises:
          TypeError: If the model object has an invalid type.
        """
        if isinstance(model, Model2D):
            num_dimensions = 2
        elif isinstance(model, Model3D):
            num_dimensions = 3
        else:
            msg = f'Invalid model type: {type(model)}'
            raise TypeError(msg)

        ops.wipe()
        ops.model('basic', '-ndm', num_dimensions, '-ndf', model.ndf)

    @staticmethod
    def opensees_define_nodes(model: Model2D | Model3D) -> None:
        """Define the nodes of the model in OpenSees."""
        for uid, node in model.get_all_nodes().items():
            ops.node(uid, *node.coordinates)

    @staticmethod
    def opensees_define_elastic_beamcolumn_elements(
        model: Model2D | Model3D,
    ) -> None:
        """
        Define elastic beamcolumn elements.

        Raises:
          TypeError: If the model type is invalid.
        """
        components = model.components.values()
        for component in components:
            elements = component.elements
            for element in elements.values():
                if isinstance(element, ElasticBeamColumn):
                    # Define it here
                    if element.visibility.skip_opensees_definition:
                        continue
                    ops.geomTransf(*element.geomtransf.ops_args())
                    if isinstance(model, Model2D):
                        ops.element(*element.ops_args_2d())
                    elif isinstance(model, Model3D):
                        ops.element(*element.ops_args())
                    else:
                        msg = f'Invalid model type: {type(model)}'
                        raise TypeError(msg)

    def opensees_define_node_restraints(
        self, model: Model2D | Model3D, load_case: LoadCase
    ) -> None:
        """
        Define node restraints.

        Raises:
          NotImplementedError: If elastic supports exist.
        """
        if load_case.elastic_supports:
            msg = 'Elastic supports not implemented yet.'
            raise NotImplementedError(msg)

        for uid, support in load_case.fixed_supports.items():
            fix = []
            for i in range(model.ndf):
                if support[i] is True or (
                    self.settings.restrict_dof
                    and self.settings.restrict_dof[i] is True
                ):
                    fix.append(True)
                else:
                    fix.append(False)
            if True in fix:
                ops.fix(uid, *[int(x) for x in fix])

    # @staticmethod
    # def opensees_define_node_mass(load_case: LoadCase) -> None:
    #     """Define node mass."""
    #     for uid, point_mass in load_case.mass_registry:
    #         ops.mass(uid, *point_mass)

    def define_model_in_opensees(
        self, model: Model2D | Model3D, load_case: LoadCase
    ) -> None:
        """Define the model in OpenSees."""
        self.opensees_instantiate(model)
        self.opensees_define_nodes(model)
        self.opensees_define_elastic_beamcolumn_elements(model)
        self.opensees_define_node_restraints(model, load_case)
        self.define_default_recorders(model)

        # TODO(JVM): debugging
        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('BandGeneral')
        ops.test('NormDispIncr', 1.0e-6, 6, 2)
        ops.algorithm('Linear')
        ops.integrator('LoadControl', 1)
        ops.analysis('Static')
        ops.analyze(1)

        # if isinstance(load_case, HasMass):
        #     self.opensees_define_node_mass(load_case)

    @staticmethod
    def define_loads_in_opensees(
        model: Model2D | Model3D,
        loadcase: LoadCase,
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
        for node_uid, point_load in loadcase.load_registry.nodal_loads.items():
            ops.load(node_uid, *(v * amplification_factor for v in point_load))

        # UDL on components
        for component_uid, global_udl in loadcase.load_registry.element_udl.items():
            component = model.components[component_uid]
            assert isinstance(component, BeamColumnAssembly)
            local_udls = component.calculate_element_udl(global_udl)
            for beamcolumn_element_uid, local_udl in local_udls.items():
                if isinstance(model, Model3D):
                    ops.eleLoad(
                        '-ele',
                        beamcolumn_element_uid,
                        '-type',
                        '-beamUniform',
                        local_udl[1] * amplification_factor,
                        local_udl[2] * amplification_factor,
                        local_udl[0] * amplification_factor,
                    )
                elif isinstance(model, Model2D):
                    ops.eleLoad(
                        '-ele',
                        beamcolumn_element_uid,
                        '-type',
                        '-beamUniform',
                        local_udl[1] * amplification_factor,
                        local_udl[0] * amplification_factor,
                    )
                else:
                    msg = f'Invalid model type: {type(model)}.'
                    raise TypeError(msg)

    def define_default_recorders(self, model: Model2D | Model3D) -> None:
        """
        Define a default set of recorders.

        Raises:
          ValueError: If the results directory is unspecified.
        """
        store_dir = self.settings.result_directory
        if store_dir is None:
            msg = 'Please specify a result directory in the analysis options.'
            raise ValueError(msg)
        recorder = NodeRecorder(
            uid_generator=model.uid_generator,
            recorder_type='Node',
            nodes=tuple(model.get_all_nodes().keys()),
            dofs=tuple(v + 1 for v in range(model.ndf)),
            response_type='disp',
            file_name=str((Path(store_dir) / 'disp').resolve()),
            number_of_significant_digits=6,
            output_time=True,
        )
        ops.recorder(*recorder.ops_args())


@dataclass(repr=False)
class StaticAnalysis(Analysis):
    """Static analysis."""
