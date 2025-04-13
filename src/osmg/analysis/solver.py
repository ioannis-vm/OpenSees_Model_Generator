"""Defines Analysis objects."""

from __future__ import annotations

from time import perf_counter
from osmg.core import common
import logging
import contextlib
import platform
import socket
import sys
import tempfile
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from osmg.analysis.load_case import HasLoads, ModalLoadCase, StaticLoadCase
from osmg.analysis.recorders import ElementRecorder, NodeRecorder
from osmg.core.common import NDF, NDM, THREE_DIMENSIONAL, TWO_DIMENSIONAL
from osmg.core.model import Model
from osmg.core.osmg_collections import BarAssembly, BeamColumnAssembly
from osmg.model_objects.element import (
    Bar,
    BeamColumnElement,
    DispBeamColumn,
    ElasticBeamColumn,
    TwoNodeLink,
    ZeroLength,
    LeadRubberX,
    TripleFrictionPendulum,
)

try:
    import opensees.openseespy as ops
except (ImportError, ModuleNotFoundError):
    import openseespy.opensees as ops

if TYPE_CHECKING:
    from osmg.analysis.common import UDL, PointLoad, PointMass
    from osmg.analysis.load_case import LoadCaseRegistry
    from osmg.analysis.recorders import Recorder
    from osmg.analysis.supports import ElasticSupport, FixedSupport
    from osmg.core.model import Model
    from osmg.core.osmg_collections import ComponentAssembly
    from osmg.model_objects.uniaxial_material import UniaxialMaterial
    from osmg.model_objects.friction_model import FrictionModel


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
class TransientDriftCheckSetup:
    """Transient analysis drift check setup."""

    node_uids: list[int]
    drift_limit: float
    elevation_dof: int = field(default=3)


@dataclass(repr=False)  # noqa: PLR0904
class Analysis:
    """Parent analysis class."""

    model: Model = field(default_factory=Model)
    settings: AnalysisSettings = field(default_factory=AnalysisSettings)
    _logger: logging.Logger = field(init=False)
    recorders: dict[str, Recorder] = field(default_factory=dict)
    _defined_materials: list[int] = field(default_factory=list)
    _defined_friction_models: list[int] = field(default_factory=list)
    _basic_force_cache: dict = field(default_factory=dict, init=False)
    _time_series_tags: list = field(default_factory=list)
    _yielded_elements: list = field(default_factory=list)

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
            filename=(
                Path(self.settings.result_directory) / self.settings.log_file_name
            ),
            filemode='w',
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            force=True,
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

    def opensees_instantiate(self) -> None:
        """Instantiate the model in OpenSees."""
        ops.wipe()
        ops.model(
            'basic',
            '-ndm',
            NDM[self.model.dimensionality],
            '-ndf',
            NDF[self.model.dimensionality],
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

    def opensees_define_friction_model(self, friction_model: FrictionModel) -> None:
        """Define friction models."""
        if friction_model.uid not in self._defined_friction_models:
            ops.frictionModel(*friction_model.ops_args())
            self._defined_friction_models.append(friction_model.uid)

    def opensees_define_nodes(self) -> None:
        """Define the nodes of the model in OpenSees."""
        for uid, node in self.model.get_all_nodes(
            self.settings.ignore_by_tag
        ).items():
            ops.node(uid, *node.coordinates)

    def opensees_define_elements(self) -> None:
        """Define elements."""
        elastic_beamcolumn_elements: list[ElasticBeamColumn] = []
        bar_elements: list[Bar] = []
        two_node_link_elements: list[TwoNodeLink] = []
        zerolength_elements: list[ZeroLength] = []
        lead_rubber_x_elements: list[LeadRubberX] = []
        triple_friction_pendulum_elements: list[TripleFrictionPendulum] = []
        unsupported_element_types: list[str] = []

        # Note: Materials are defined on an as-needed basis.  We keep
        # track of defined materials in `self._defined_materials` to
        # avoid trying to define the same material twice. After
        # elements are defined, we clear that list, to start with an
        # empty list in subsequent model definitions.

        components = self.model.components.values()
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
                elif isinstance(element, LeadRubberX):
                    lead_rubber_x_elements.append(element)
                elif isinstance(element, TripleFrictionPendulum):
                    triple_friction_pendulum_elements.append(element)
                else:
                    unsupported_element_types.append(element.__class__.__name__)

        if unsupported_element_types:
            print(  # noqa: T201
                f'WARNING: Unsupported element types found: {set(unsupported_element_types)}'
            )

        self.opensees_define_elastic_beamcolumn_elements(elastic_beamcolumn_elements)
        self.opensees_define_bar_elements(bar_elements)
        self.opensees_define_two_node_link_elements(two_node_link_elements)
        self.opensees_define_zerolength_elements(zerolength_elements)
        self.opensees_define_lead_rubber_x_elements(lead_rubber_x_elements)
        self.opensees_define_triple_friction_pendulum_elements(
            triple_friction_pendulum_elements
        )

        # clear defined materials
        self._defined_materials = []
        self._defined_friction_models = []

    def opensees_define_lead_rubber_x_elements(
        self, elements: list[LeadRubberX]
    ) -> None:
        """
        Define LeadRubberX elements.

        Raises:
          ValueError: If the analysis is not 3D.
        """
        ndm = NDM[self.model.dimensionality]
        ndf = NDF[self.model.dimensionality]
        if not (ndm == 3 and ndf == 6):
            msg = 'LeadRubberX elements only work with ndm=3 and ndf=6.'
            raise ValueError(msg)
        for element in elements:
            ops.element(*element.ops_args())

    def opensees_define_triple_friction_pendulum_elements(
        self, elements: list[TripleFrictionPendulum]
    ) -> None:
        """
        Define TripleFrictionPendulum elements.

        Raises:
          ValueError: If the analysis is not 3D.
        """
        ndm = NDM[self.model.dimensionality]
        ndf = NDF[self.model.dimensionality]
        if not (ndm == 3 and ndf == 6):
            msg = 'LeadRubberX elements only work with ndm=3 and ndf=6.'
            raise ValueError(msg)
        for element in elements:
            for friction_model in (
                element.friction_model_1,
                element.friction_model_2,
                element.friction_model_3,
            ):
                self.opensees_define_friction_model(friction_model)
            for material in (
                element.vertical_material,
                element.rot_z_material,
                element.rot_x_material,
                element.rot_y_material,
            ):
                self.opensees_define_material(material)
            ops.element(*element.ops_args())

    def opensees_define_elastic_beamcolumn_elements(
        self, elements: list[ElasticBeamColumn]
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
            if self.model.dimensionality == '2D Frame':
                ops.element(*element.ops_args_2d())
            elif self.model.dimensionality == '3D Frame':
                ops.element(*element.ops_args())
            else:
                msg = f'Invalid model dimensionality: `{self.model.dimensionality}`.'
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
        self,
        fixed_supports: dict[int, FixedSupport],
        elastic_supports: dict[int, ElasticSupport],
    ) -> None:
        """Define node restraints."""
        ndf = NDF[self.model.dimensionality]

        for uid, support in fixed_supports.items():
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

        nodes = self.model.get_all_nodes()
        elastic_materials = {}
        for uid, support in elastic_supports.items():
            assert len(support) == ndf
            node = nodes[uid]
            # for each direction.
            material_uids_for_this_support = []
            for value in support:
                # define material if needed.
                if value not in elastic_materials:
                    material_uid = next(self.model.uid_generator.MATERIAL)
                    elastic_materials[value] = material_uid
                    ops.uniaxialMaterial('Elastic', material_uid, value)
                else:
                    material_uid = elastic_materials[value]
                material_uids_for_this_support.append(material_uid)
            # define a node at the same location.
            new_node_uid = next(self.model.uid_generator.NODE)
            ops.node(new_node_uid, *node.coordinates)
            # fix that node.
            ops.fix(new_node_uid, *([1] * ndf))
            # define a zerolength element connecting the two nodes.
            ops.element(
                'zeroLength',
                next(self.model.uid_generator.ELEMENT),
                uid,
                new_node_uid,
                '-mat',
                *material_uids_for_this_support,
                '-dir',
                *range(1, ndf + 1),
            )

    def opensees_define_node_constraints(
        self, rigid_diaphragm: dict[int, tuple[int, ...]]
    ) -> None:
        """
        Define node constraints.

        Raises:
          ValueError: If the model dimensionality is not supported.
        """
        if not rigid_diaphragm:
            return

        for (
            parent_node_uid,
            children_node_uids,
        ) in rigid_diaphragm.items():
            if self.model.dimensionality in {'3D Frame', '3D Truss'}:
                ops.rigidDiaphragm(3, parent_node_uid, *children_node_uids)
            elif self.model.dimensionality in {'2D Frame', '2D Truss'}:
                for child_node_uid in children_node_uids:
                    ops.equalDOF(parent_node_uid, child_node_uid, 1)
            else:
                msg = 'Unsupported model dimensionality: {model.dimensionality}'
                raise ValueError(msg)

    def opensees_define_model(self) -> None:
        """Define the model in OpenSees."""
        self.opensees_instantiate()
        self.opensees_define_nodes()
        self.opensees_define_elements()
        if not self.settings.disable_default_recorders:
            self.define_default_recorders()

    def opensees_define_loads(
        self,
        nodal_loads: dict[int, PointLoad],
        component_udl: dict[int, UDL],
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
        if time_series_tag not in self._time_series_tags:
            ops.timeSeries('Linear', time_series_tag)
            self._time_series_tags.append(time_series_tag)

        ops.pattern('Plain', pattern_tag, time_series_tag)

        # Point load on nodes
        for node_uid, point_load in nodal_loads.items():  # type: ignore
            ops.load(node_uid, *(v * amplification_factor for v in point_load))

        # UDL on components
        for component_uid, global_udl in component_udl.items():  # type: ignore
            component = self.model.components[component_uid]
            if component.tags & self.settings.ignore_by_tag:
                continue
            assert isinstance(component, BeamColumnAssembly)
            local_udls = component.calculate_element_udl(global_udl)
            for beamcolumn_element_uid, local_udl in local_udls.items():
                if self.model.dimensionality == '3D Frame':
                    ops.eleLoad(
                        '-ele',
                        beamcolumn_element_uid,
                        '-type',
                        '-beamUniform',
                        local_udl[1] * amplification_factor,
                        local_udl[2] * amplification_factor,
                        local_udl[0] * amplification_factor,
                    )
                elif self.model.dimensionality == '2D Frame':
                    ops.eleLoad(
                        '-ele',
                        beamcolumn_element_uid,
                        '-type',
                        '-beamUniform',
                        local_udl[1] * amplification_factor,
                        local_udl[0] * amplification_factor,
                    )
                else:
                    msg = f'Invalid model dimensionality: `{self.model.dimensionality}`.'
                    raise TypeError(msg)

    @staticmethod
    def opensees_define_mass(
        mass_registry: dict[int, PointMass], amplification_factor: float = 1.00
    ) -> None:
        """Define mass in OpenSees."""
        for node_uid, point_mass in mass_registry.items():  # type: ignore
            amplified_mass = [float(v) * amplification_factor for v in point_mass]
            if any(amplified_mass):  # Skip if all elements are zero
                ops.mass(node_uid, *amplified_mass)

    def define_default_recorders(self) -> None:
        """
        Create a set of default recorders.

        Does not define them in  OpenSees.

        Raises:
          ValueError: If the results directory is unspecified.
        """
        ndf = NDF[self.model.dimensionality]
        store_dir = self.settings.result_directory
        if store_dir is None:
            msg = 'Please specify a result directory in the analysis options.'
            raise ValueError(msg)
        node_recorder = NodeRecorder(
            uid_generator=self.model.uid_generator,
            file_name='node_displacements',
            recorder_type='Node',
            nodes=tuple(self.model.get_all_nodes().keys()),
            dofs=tuple(v + 1 for v in range(ndf)),
            response_type='disp',
            number_of_significant_digits=6,
            output_time=True,
        )
        self.recorders['default_node'] = node_recorder
        node_reaction_recorder = NodeRecorder(
            uid_generator=self.model.uid_generator,
            file_name='node_reactions',
            recorder_type='Node',
            nodes=tuple(self.model.get_all_nodes().keys()),
            dofs=tuple(v + 1 for v in range(ndf)),
            response_type='reaction',
            number_of_significant_digits=6,
            output_time=True,
        )
        self.recorders['default_node_reaction'] = node_reaction_recorder

        applicable_elements = []
        components = self.model.components.values()
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
            uid_generator=self.model.uid_generator,
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

    def opensees_mck(self) -> None:
        """Get the mass, damping, and stiffness matrices."""
        ops.wipeAnalysis()
        ops.system('FullGeneral')
        ops.numberer(self.settings.numberer)
        ops.constraints(*self.settings.constraints)
        ops.analysis('Transient')
        # Mass
        ops.integrator('GimmeMCK', 1.0, 0.0, 0.0)
        ops.analyze(1, 0.0)
        # Number of equations in the model
        num = ops.systemSize()  # Has to be done after analyze
        # Convert to np array and reshape to NxN matrix
        mass_mat = np.array(ops.printA('-ret')).reshape((num, num))
        # Damping
        ops.integrator('GimmeMCK', 0.0, 1.0, 0.0)
        ops.analyze(1, 0.0)
        damping_mat = np.array(ops.printA('-ret')).reshape((num, num))
        # Stiffness
        ops.integrator('GimmeMCK', 0.0, 0.0, 1.0)
        ops.analyze(1, 0.0)
        stiffness_mat = np.array(ops.printA('-ret')).reshape((num, num))
        return {
            'number_of_equations': num,
            'stiffness_matrix': stiffness_mat,
            'damping_matrix': damping_mat,
            'mass_matrix': mass_mat,
        }

    def calculate_basic_forces(  # noqa: C901
        self,
        recorder_name: str,
        components: dict[int, ComponentAssembly],
        *,
        ndm: int,
        num_stations: int = 12,
    ) -> tuple[
        pd.DataFrame,  # Axial forces
        pd.DataFrame,  # Shear forces (Y)
        pd.DataFrame,  # Shear forces (Z)
        pd.DataFrame,  # Torsion
        pd.DataFrame,  # Bending moments (Y)
        pd.DataFrame,  # Bending moments (Z)
    ]:
        """
        Calculate basic forces at intermediate locations.

        This function calculates axial forces, shear forces (in Y and
        Z directions), torsion, and bending moments (in Y and Z
        directions) at multiple stations along each element based on
        the provided recorder data and element lengths.  The results
        are computed for either 2D or 3D elements and returned as a
        tuple of Pandas DataFrames.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
            pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                A tuple containing six DataFrames in the following
                order:
                - Axial forces (`axial_df`)
                - Shear forces in the Y direction (`shear_y_df`)
                - Shear forces in the Z direction (`shear_z_df`) or
                  zeros for 2D
                - Torsion (`torsion_df`) or zeros for 2D
                - Bending moments in the Y direction (`moment_y_df`)
                  or zeros for 2D
                - Bending moments in the Z direction (`moment_z_df`)
                Each DataFrame is multi-indexed by 'element' and
                'station', with forces computed at evenly spaced
                stations along each element.

        Raises:
            ValueError: If the specified recorder is not found.
            TypeError: If the specified recorder is not an Element
              recorder.
            ValueError: If `ndf` (number of degrees of freedom) is not
              2 or 3.
            ValueError: If the recorder data do not have required
              column levels 'dof' or 'station'.
            ValueError: If element length information is missing for
              any element.
        """
        cache_key = (recorder_name, tuple(components.keys()), ndm, num_stations)

        # Check if the result is already cached
        if cache_key in self._basic_force_cache:
            return self._basic_force_cache[cache_key]

        recorder = self.recorders.get(recorder_name)
        if recorder is None:
            msg = f'Specified recorder not available: {recorder_name}.'
            raise ValueError(msg)
        if not isinstance(recorder, ElementRecorder):
            msg = f'The specified recorder (`{recorder_name}`) is not an Element recorder.'
            raise TypeError(msg)

        if isinstance(self, HasLoads):
            udls_global = self.load_registry.component_udl
        else:
            udls_global = {}

        udls_local = {}
        elements = {}
        for component_uid, component in components.items():
            if not isinstance(component, BeamColumnAssembly):
                if isinstance(component, BarAssembly):
                    elements.update(component.elements)
                continue
            for element in component.elements.values():
                if isinstance(element, (BeamColumnElement, Bar)):
                    elements[element.uid] = element
            elements.update(component.elements)
            component_global_udl = udls_global.get(component_uid)
            if component_global_udl:
                component_local_udls = component.calculate_element_udl(
                    component_global_udl
                )
                udls_local.update(component_local_udls)

        data = recorder.get_data()

        required_levels = {'dof', 'station'}
        if not required_levels.issubset(data.columns.names):
            msg = f'Data must have levels: {required_levels}'
            raise ValueError(msg)

        ndf = data.columns.get_level_values('dof').max()
        if ndf not in {3, 6}:
            msg = 'Must be either 2D or 3D Frame.'
            raise ValueError(msg)
        dof_data = {
            dof: data.xs(dof, level='dof', axis=1)
            for dof in data.columns.get_level_values('dof').unique()
        }
        dof_data_i = {
            dof: df.xs(0.0, level='station', axis=1) for dof, df in dof_data.items()
        }

        # dof_data_j = {
        #     dof: df.xs(1.0, level='station', axis=1) for dof, df in dof_data.items()
        # }

        missing_elements = [e for e in recorder.elements if e not in elements]
        if missing_elements:
            msg = f'Missing elements: {missing_elements}'
            raise ValueError(msg)

        element_lengths: dict[int, float] = {
            element.uid: element.clear_length()
            for element in elements.values()
            if isinstance(element, (BeamColumnElement, Bar))
        }

        locations = np.array(
            [
                np.linspace(0.00, element_lengths[element], num=num_stations)
                for element in recorder.elements
            ]
        )
        locations_expanded = locations[np.newaxis, :, :]

        columns = pd.MultiIndex.from_tuples(
            [  # noqa: C416
                (element, loc)
                for element, loc in product(
                    recorder.elements,
                    [
                        float(f'{v:.2f}')
                        for v in np.linspace(0.00, 1.00, num=num_stations)
                    ],
                )
            ],
            names=['element', 'station'],
        )

        # ~~~ Local X axis axial force ~~~

        data_i_axial = dof_data_i[1].to_numpy()[:, :, np.newaxis]
        # data_j_axial = dof_data_j[1]
        w_data_axial = np.array(
            [
                udls_local[element_uid][0]
                if udls_local.get(element_uid) is not None
                else 0.00
                for element_uid in recorder.elements
            ]
        )[np.newaxis, :, np.newaxis]

        result_axial = -(data_i_axial + w_data_axial * locations_expanded)
        result_axial_flattened = result_axial.reshape(result_axial.shape[0], -1)
        axial_df = pd.DataFrame(
            result_axial_flattened, index=dof_data_i[1].index, columns=columns
        )

        # # check: results at station '1.00' should be equal to `data_j`.
        # pd.testing.assert_frame_equal(
        #     axial_df.xs(1.0, level='station', axis=1),
        #     data_j_axial,
        #     check_exact=False,
        #     atol=1e-6,
        # )

        # ~~~ Local Y axis shear force ~~~

        data_i_shear_y = dof_data_i[2].to_numpy()[:, :, np.newaxis]
        # data_j_shear_y = dof_data_j[2]
        w_data_shear_y = np.array(
            [
                udls_local[element_uid][1]
                if udls_local.get(element_uid) is not None
                else 0.00
                for element_uid in recorder.elements
            ]
        )[np.newaxis, :, np.newaxis]

        result_shear_y = data_i_shear_y + w_data_shear_y * locations_expanded
        result_shear_y_flattened = result_shear_y.reshape(
            result_shear_y.shape[0], -1
        )
        shear_y_df = pd.DataFrame(
            result_shear_y_flattened, index=dof_data_i[1].index, columns=columns
        )

        # # check: results at station '1.00' should be equal to `data_j`.
        # pd.testing.assert_frame_equal(
        #     shear_y_df.xs(1.0, level='station', axis=1),
        #     - data_j_shear_y,
        #     check_exact=False,
        #     atol=1e-6,
        # )

        # ~~~ Local Z axis shear force ~~~

        # In 2D this is zero.
        if ndm == TWO_DIMENSIONAL:
            shear_z_df = pd.DataFrame(
                0.00, index=axial_df.index, columns=shear_y_df.columns
            )
        elif ndm == THREE_DIMENSIONAL:
            data_i_shear_z = dof_data_i[3].to_numpy()[:, :, np.newaxis]
            # data_j_shear_z = dof_data_j[3]
            w_data_shear_z = np.array(
                [
                    udls_local[element_uid][2]
                    if udls_local.get(element_uid) is not None
                    else 0.00
                    for element_uid in recorder.elements
                ]
            )[np.newaxis, :, np.newaxis]

            result_shear_z = data_i_shear_z + w_data_shear_z * locations_expanded
            result_shear_z_flattened = result_shear_z.reshape(
                result_shear_z.shape[0], -1
            )
            shear_z_df = pd.DataFrame(
                result_shear_z_flattened, index=dof_data_i[1].index, columns=columns
            )

            # # check: results at station '1.00' should be equal to `data_j`.
            # pd.testing.assert_frame_equal(
            #     shear_z_df.xs(1.0, level='station', axis=1),
            #     data_j_shear_z,
            #     check_exact=False,
            #     atol=1e-6,
            # )

        # ~~~ Local X axis torsional moment ~~~

        # In 2D this is zero.
        if ndm == TWO_DIMENSIONAL:
            torsion_df = pd.DataFrame(
                0.00, index=axial_df.index, columns=shear_y_df.columns
            )
        elif ndm == THREE_DIMENSIONAL:
            data_i_torsion = dof_data_i[4].to_numpy()[:, :, np.newaxis]
            # data_j_torsion = dof_data_j[4]

            result_torsion = -data_i_torsion * np.ones_like(locations_expanded)
            result_torsion_flattened = result_torsion.reshape(
                result_torsion.shape[0], -1
            )
            torsion_df = pd.DataFrame(
                result_torsion_flattened, index=dof_data_i[1].index, columns=columns
            )

            # # check: results at station '1.00' should be equal to `data_j`.
            # pd.testing.assert_frame_equal(
            #     torsion_df.xs(1.0, level='station', axis=1),
            #     data_j_torsion,
            #     check_exact=False,
            #     atol=1e-6,
            # )

        # ~~~ Local Y axis bending moment (typically the weak axis) ~~~
        if ndm == TWO_DIMENSIONAL:
            moment_y_df = pd.DataFrame(
                0.00, index=axial_df.index, columns=shear_y_df.columns
            )
        elif ndm == THREE_DIMENSIONAL:
            data_i_moment_y = dof_data_i[5].to_numpy()[:, :, np.newaxis]
            # data_j_moment_y = dof_data_j[5]

            # Already obtained: {w_data_shear_z, data_i_shear_z, data_j_shear_z}

            result_moment_y = (
                locations_expanded**2 * 0.50 * w_data_shear_z
                + locations_expanded * data_i_shear_z
                + data_i_moment_y
            )
            result_moment_y_flattened = result_moment_y.reshape(
                result_moment_y.shape[0], -1
            )
            moment_y_df = pd.DataFrame(
                result_moment_y_flattened, index=dof_data_i[1].index, columns=columns
            )

            # # check: results at station '1.00' should be equal to `data_j`.
            # pd.testing.assert_frame_equal(
            #     moment_y_df.xs(1.0, level='station', axis=1),
            #     data_j_moment_y,
            #     check_exact=False,
            #     atol=1e-6,
            # )

        # ~~~ Local Z axis bending moment (typically the strong axis) ~~~
        if ndm == TWO_DIMENSIONAL:
            data_i_moment_z = dof_data_i[3].to_numpy()[:, :, np.newaxis]
            # data_j_moment_z = dof_data_j[3]

        elif ndm == THREE_DIMENSIONAL:
            data_i_moment_z = dof_data_i[6].to_numpy()[:, :, np.newaxis]
            # data_j_moment_z = dof_data_j[6]

        # Already obtained: {w_data_shear_y, data_i_shear_y, data_j_shear_y}

        result_moment_z = (
            locations_expanded**2 * 0.50 * w_data_shear_y
            + locations_expanded * data_i_shear_y
            - data_i_moment_z
        )
        result_moment_z_flattened = result_moment_z.reshape(
            result_moment_z.shape[0], -1
        )
        moment_z_df = pd.DataFrame(
            result_moment_z_flattened, index=dof_data_i[1].index, columns=columns
        )

        # # check: results at station '1.00' should be equal to `data_j`.
        # pd.testing.assert_frame_equal(
        #     moment_z_df.xs(1.0, level='station', axis=1),
        #     data_j_moment_z,
        #     check_exact=False,
        #     atol=1e-6,
        # )

        result = (
            axial_df,
            shear_y_df,
            shear_z_df,
            torsion_df,
            moment_y_df,
            moment_z_df,
        )
        self._basic_force_cache[cache_key] = result

        return result

    def run_static(
        self, load_case: StaticLoadCase, num_steps: int = 1, *, wipe: bool = True
    ) -> None:
        """Run a static analysis."""
        self.initialize_logger()
        self.log('Running a static analysis.')
        self.log('Defining model in OpenSees.')
        self.opensees_define_model()
        self.opensees_define_recorders()
        self.opensees_define_node_restraints(
            load_case.fixed_supports, load_case.elastic_supports
        )
        self.opensees_define_node_constraints(load_case.rigid_diaphragm)
        self.log('Defining loads in OpenSees.')
        self.opensees_define_loads(
            load_case.load_registry.nodal_loads,
            load_case.load_registry.component_udl,
        )
        self.log('Setting up analysis.')
        self.log(f'Setting system solver to {self.settings.system}')
        ops.system(self.settings.system)
        self.log(f'Setting numberer to {self.settings.numberer}')
        ops.numberer(self.settings.numberer)
        self.log(f'Setting constraints to {self.settings.constraints}')
        ops.constraints(*self.settings.constraints)
        self.log("Setting test to ('EnergyIncr', 1.0e-8, 20, 3)")
        ops.test('EnergyIncr', 1.0e-8, 20, 3)
        self.log('Setting algorithm to KrylovNewton')
        ops.algorithm('KrylovNewton')
        self.log(f'Setting integrator to LoadControl with {num_steps} steps')
        ops.integrator('LoadControl', 1.00 / num_steps)
        self.log('Setting analysis to Static')
        ops.analysis('Static')
        self.log('Analyzing.')
        out = ops.analyze(num_steps)
        assert out == 0, 'Analysis failed.'
        if wipe:
            ops.wipe()
        self.log('Analysis finished.')

    def run_pushover(
        self,
        target_displacements: list[float | None],
        control_node_uid: int,
        dof: int,
        displ_incr: float,
        *,
        pattern_tag: int,
        max_steps: int = 50,
        norm: float = 1e-8,
    ) -> None:
        """
        Run a pushover analysis.

        Raises:
          ValueError: If it fails to unload.
        """
        curr_displ = ops.nodeDisp(control_node_uid, dof)

        self.log('Starting pushover analysis')
        ops.system(self.settings.system)
        ops.numberer(self.settings.numberer)
        ops.constraints(*self.settings.constraints)

        total_fail = False
        num_subdiv = 0
        num_times = 0
        algorithm_idx = 0

        scale = [1.0, 1.0e-1, 1.0e-2, 1.0e-3]
        algorithms = [('KrylovNewton',), ('KrylovNewton', 'initial')]

        for i_loop, target_displacement in enumerate(target_displacements):
            if total_fail:
                break
            if target_displacement is not None:
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

                    ops.test('NormDispIncr', norm, max_steps, 0)
                    ops.algorithm(*algorithms[algorithm_idx])
                    ops.system(self.settings.solver)
                    ops.integrator(
                        'DisplacementControl', control_node_uid, dof, incr
                    )
                    ops.analysis('Static')
                    flag = ops.analyze(1)
                    if flag != 0:
                        if num_subdiv == len(scale) - 1:
                            # can't refine further
                            print('Analysis failed to converge')
                            self.log(f'Analysis failed at disp {curr_displ:.5f}')
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

                        curr_displ = ops.nodeDisp(control_node_uid, dof)
                        print(
                            f'Loop ({i_loop + 1}/'
                            f'{len(target_displacements)}) | '
                            'Target displacement: '
                            f'{target_displacement:.2f}'
                            f' | Current: {curr_displ:.4f}',
                            end='\r',
                        )
                        algorithm_idx = 0
                        if num_subdiv != 0:
                            if num_times == 0:
                                num_subdiv -= 1
                                num_times = 10

            else:
                # Need to unload
                ops.test('NormDispIncr', norm[num_subdiv], max_steps, 0)
                ops.algorithm(*algorithms[algorithm_idx])
                current_load = ops.getLoadFactor(pattern_tag)
                load_threshold = 1e-4
                while current_load > load_threshold:
                    increment = -current_load / 10.00
                    ops.integrator('LoadControl', increment)
                    flag = ops.analyze(1)
                    if flag != 0:
                        msg = 'Failed to unload.'
                        raise ValueError(msg)
                    current_load = ops.getLoadFactor(pattern_tag)
                    curr_displ = ops.nodeDisp(control_node_uid, dof)
                    print(
                        f'Loop ({i_loop + 1}/'
                        f'{len(target_displacements)}) | '
                        'Target displacement: '
                        f'(Unloading)'
                        f' | Current: {curr_displ:.4f}',
                        end='\r',
                    )

        print('Analysis finished.')

    @staticmethod
    def opensees_define_transient_time_series_and_pattern(
        direction: int,
        ag_vec: np.ndarray,
        dt: float,
        factor: float,
        time_series_tag: int,
        pattern_tag: int,
    ) -> None:
        """Define a time series and pattern for a transient analysis."""
        ops.timeSeries(
            'Path',
            time_series_tag,
            '-dt',
            dt,
            '-values',
            *ag_vec,
            '-factor',
            factor,
        )
        ops.pattern(
            'UniformExcitation', pattern_tag, direction, '-accel', time_series_tag
        )

    @staticmethod
    def opensees_define_viscous_damping_rayleigh(
        period_1: float, period_2: float, ratio: float
    ) -> None:
        """
        Assign Rayleigh damping.

        Assigns Rayleigh damping given two periods and the damping
        ratio.
        """
        w_i = 2 * np.pi / period_1
        w_j = 2 * np.pi / period_2
        zeta_i = ratio
        zeta_j = ratio
        a_mat = np.array([[1 / w_i, w_i], [1 / w_j, w_j]])
        b_vec = np.array([zeta_i, zeta_j])
        x_sol = np.linalg.solve(a_mat, 2 * b_vec)
        ops.rayleigh(x_sol[0], 0.0, 0.0, x_sol[1])
        # https://portwooddigital.com/2020/11/08/rayleigh-damping-coefficients/
        # --thanks, prof. Scott

    @staticmethod
    def opensees_define_viscous_damping_stiffness(
        period: float, ratio: float
    ) -> None:
        """Assign stiffness-proportional damping."""
        ops.rayleigh(0.00, 0.0, 0.0, ratio * period / np.pi)

    @staticmethod
    def opensees_define_viscous_damping_modal(num_modes: int, ratio: float) -> None:
        """Assign modal damping."""
        ops.eigen(num_modes)
        ops.modalDampingQ(ratio)

    @staticmethod
    def opensees_define_viscious_damping_modal_and_stiffness(
        ratio_stiffness: float, period: float, ratio_modal: float, num_modes: int
    ) -> None:
        """
        Assign modal + stiffness-proportional damping.

        Assigns modal combined with stiffness proportional damping.
        """
        alpha_1 = ratio_stiffness * period / np.pi
        ops.rayleigh(
            0.00,
            0.0,
            0.0,
            alpha_1,
        )
        omega_squareds = np.array(ops.eigen(num_modes))
        damping_vals = ratio_modal - alpha_1 * np.sqrt(omega_squareds) / 2.00
        assert np.min(damping_vals) > 0.00
        ops.modalDampingQ(*damping_vals)

    def run_transient(
        self,
        *,
        analysis_time_increment: float,
        finish_time: float,
        print_progress: bool = True,
        time_limit: float | None = None,
        transient_drift_check_setup: TransientDriftCheckSetup | None = None,
    ) -> None:
        """
        Run a transient analysis.

        Arguments:
          analysis_time_increment: Time increment.
          finish_time: Specify a target time (s) to stop the analysis.
          print_progress: Controls whether the current time is printed out.
          time_limit: Maximum analysis time allowed, in hours.
          When reached, the analysis is interrupted.

        Raises:
          ValueError: If no mass is assigned.
        """
        # Check if there is any mass
        total_mass = 0.00
        node_tags = ops.getNodeTags()
        ndf = NDF[self.model.dimensionality]
        for node in node_tags:
            total_mass += np.sum([ops.nodeMass(node, i + 1) for i in range(ndf)])
        if total_mass == 0.00:
            msg = 'No mass!'
            raise ValueError(msg)

        # Constants
        tolerance: float = 1e-9
        max_iterations: int = 100
        time_limit_seconds: float | None = (
            time_limit * 3600.0 if time_limit else None
        )

        self.log('Starting transient analysis')

        # Set up OpenSees parameters
        ops.numberer(self.settings.numberer)
        ops.constraints(*self.settings.constraints)
        ops.system(self.settings.solver)
        ops.test('EnergyIncr', tolerance, max_iterations, 0)
        ops.integrator('Newmark', 0.5, 0.25)
        ops.algorithm('Newton')
        ops.analysis('Transient')

        # Progress bar
        pbar = self._initialize_progress_bar(
            print_progress=print_progress, finish_time=finish_time
        )

        start_time: float = perf_counter()
        last_log_time: float = start_time
        curr_time: float = 0.0

        self.log('Initiating time traversal')
        self.log(f'Max time step: {analysis_time_increment}')

        try:
            self._perform_analysis_loop(
                curr_time,
                finish_time,
                analysis_time_increment,
                start_time,
                last_log_time,
                pbar,
                time_limit_seconds,
                transient_drift_check_setup,
            )
        except KeyboardInterrupt:
            self.log('Analysis interrupted')
            self._logger.warning('Analysis interrupted')

        self.log('Analysis finished.')
        if pbar is not None:
            pbar.close()

    def run_transient_dampen_residual_response(
        self,
        prior_load_patterns: list,
        analysis_time_increment: float,
        *,
        print_norm: bool = True,
        velocity_threshold: float = 1e-2,
        rayleigh_factor: float = 1.00,
        absolute_tolerance: float = 1e-4,
    ) -> None:
        """Dampen out any residual motion."""
        for load_pattern in prior_load_patterns:
            ops.remove('loadPattern', load_pattern)

        ops.rayleigh(rayleigh_factor, 0.00, 0.0, 0.00)

        vel_norm = np.inf
        while vel_norm > velocity_threshold:
            check = ops.analyze(1, analysis_time_increment)

            if check != 0:
                self.log('Failed to dampen out residual motion: Analysis fails.')
                break

            vel = np.zeros(6)
            node_tags = ops.getNodeTags()
            num_nodes = len(node_tags)
            for ntag in node_tags:
                vel += [x / num_nodes for x in ops.nodeVel(ntag)]
            previous_norm = vel_norm
            vel_norm = np.sqrt(vel @ vel)

            if print_norm:
                print(f'{vel_norm:5.3e} > {velocity_threshold:5.3e}', end='\r')

            if np.abs(vel_norm - previous_norm) < absolute_tolerance:
                self.log(
                    'Failed to dampen out residual motion: '
                    'Velocity norm stopped decreasing.'
                )
                break

        print()

    @staticmethod
    def _initialize_progress_bar(
        *, print_progress: bool, finish_time: float
    ) -> tqdm.std.tqdm | None:
        """Initialize a progress bar if requested."""
        if not print_progress:
            return None

        pbar: tqdm.std.tqdm = tqdm(
            total=finish_time,
            bar_format='{percentage:3.0f}%|{bar:25}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
        )
        return pbar

    def _perform_analysis_loop(
        self,
        curr_time: float,
        finish_time: float,
        analysis_time_increment: float,
        start_time: float,
        last_log_time: float,
        pbar: tqdm.std.tqdm | None,
        time_limit_seconds: float | None,
        transient_drift_check_setup: TransientDriftCheckSetup | None,
    ) -> None:
        """Perform the main loop of a transient analysis."""
        scale: tuple[float, float, float, float] = (1.0, 1.0e-1, 1.0e-2, 1.0e-3)
        algorithms: tuple[tuple[str], tuple[str, str, str], tuple[str]] = (
            ('KrylovNewton',),
            ('KrylovNewton', 'initial', 'initial'),
            ('NewtonLineSearch',),
        )
        num_subdiv: int = 0
        num_times: int = 0
        algorithm_idx: int = 0
        analysis_failed: bool = False

        while curr_time + common.EPSILON < finish_time:
            if analysis_failed:
                break

            ops.test('EnergyIncr', 1e-9, 100, 0)
            ops.algorithm(*algorithms[algorithm_idx])
            check: int = ops.analyze(1, analysis_time_increment * scale[num_subdiv])

            if check != 0:
                num_subdiv, num_times, algorithm_idx, should_stop = (
                    self._handle_analysis_failure(
                        curr_time,
                        num_subdiv,
                        algorithm_idx,
                        num_times,
                        scale,
                        algorithms,
                    )
                )
                if should_stop:
                    break
            else:
                curr_time, last_log_time, should_stop = (
                    self._handle_successful_analysis(
                        curr_time,
                        finish_time,
                        pbar,
                        start_time,
                        last_log_time,
                        num_subdiv,
                        num_times,
                        time_limit_seconds,
                        transient_drift_check_setup,
                    )
                )
                if should_stop:
                    break  # Stop the loop if the time limit is reached

    def _handle_analysis_failure(
        self,
        curr_time: float,
        num_subdiv: int,
        algorithm_idx: int,
        num_times: int,
        scale: tuple[float, float, float, float],
        algorithms: tuple[tuple[str], tuple[str, str, str], tuple[str]],
    ) -> bool:
        """Handle an analysis failure by adjusting parameters or logging failure."""
        if num_subdiv == len(scale) - 1:
            self.log('Analysis failed to converge.')
            self._logger.warning(
                f'Analysis failed at time {curr_time:.5f} and cannot continue.'
            )
            return num_subdiv, num_times, algorithm_idx, True  # Analysis failed

        if algorithm_idx < len(algorithms) - 1:
            # Try another algorithm.
            algorithm_idx += 1
            print(f'{num_subdiv=}\t{num_times=}\t{algorithm_idx=}')
            return (
                num_subdiv,
                num_times,
                algorithm_idx,
                False,
            )

        # Increase subdivisions if all algorithms have been attempted
        algorithm_idx = 0
        num_subdiv += 1
        num_times = 50  # Reset retries

        return num_subdiv, num_times, algorithm_idx, False

    def _handle_successful_analysis(
        self,
        curr_time: float,
        finish_time: float,
        pbar: tqdm.std.tqdm | None,
        start_time: float,
        last_log_time: float,
        num_subdiv: int,
        num_times: int,
        time_limit_seconds: float | None,
        transient_drift_check_setup: TransientDriftCheckSetup | None,
    ) -> tuple[float, float, bool]:
        """Handle successful analysis updates."""
        prev_time: float = curr_time
        curr_time = float(ops.getTime())
        test_iter: int = ops.testIter()

        # Update progress bar
        if pbar is not None:
            pbar.set_postfix(
                {'time': f'{curr_time:.4f}/{finish_time:.2f} [{test_iter}]'}
            )
            pbar.update(curr_time - prev_time)

        # Periodic logging
        if perf_counter() - last_log_time > 5.00 * 60.00:  # 5 min
            last_log_time = perf_counter()
            running_time: float = last_log_time - start_time
            remaining_time: float = finish_time - curr_time
            average_speed: float = curr_time / running_time
            est_remaining_dur: float = remaining_time / average_speed
            self.log(
                f'Analysis status: {{curr: {curr_time:.2f}, target: {finish_time:.2f}, '
                f'num_subdiv: {num_subdiv}, ~ {est_remaining_dur:.0f} s to finish}}'
            )

        # Time limit check
        if time_limit_seconds and (perf_counter() - start_time) > time_limit_seconds:
            self._logger.warning(
                f'Analysis interrupted at time {curr_time:.5f} because the time limit was reached.'
            )
            return curr_time, last_log_time, True  # Return flag to stop the loop

        # Maximum drift check
        if transient_drift_check_setup is not None:
            node_uids = transient_drift_check_setup.node_uids
            drift_limit = transient_drift_check_setup.drift_limit
            elevation_dof = transient_drift_check_setup.elevation_dof
            # We need at least two nodes
            min_nodes_required = 2
            assert len(node_uids) > min_nodes_required
            assert elevation_dof in {1, 2, 3}
            if elevation_dof == 1:
                other_dofs = (2, 3)
            elif elevation_dof == 2:  # noqa: PLR2004
                other_dofs = (1, 3)
            else:
                other_dofs = (1, 2)
            node_pairs = list(zip(node_uids, node_uids[1:]))
            for bottom_node, top_node in node_pairs:
                bottom_elev = ops.nodeCoord(bottom_node)[elevation_dof - 1]
                top_elev = ops.nodeCoord(top_node)[elevation_dof - 1]
                for other_dof in other_dofs:
                    top_disp = ops.nodeDisp(top_node, other_dof)
                    bottom_disp = ops.nodeDisp(bottom_node, other_dof)
                    drift = (top_disp - bottom_disp) / (top_elev - bottom_elev)
                    if drift > drift_limit:
                        self.log(
                            f'Drift limit reached: {drift * 100:.2}% at dofs ({top_node}, {bottom_node}). '
                            f'Time: {curr_time:.3f} s.'
                            f'Stopping analysis.'
                        )
                        return curr_time, last_log_time, True  # Stop

        if num_times != 0:
            num_times -= 1
        elif num_subdiv != 0:
            num_subdiv -= 1
            num_times = 50

        return curr_time, last_log_time, False  # Continue analysis


@dataclass()
class ModalAnalysisSettings(AnalysisSettings):
    """Modal analysis settings."""

    num_modes: int = field(default=3)
    retrieve_basic_forces: bool = field(default=True)


@dataclass(repr=False)
class ModalAnalysis(Analysis):
    """Modal analysis."""

    load_case: ModalLoadCase = field(default_factory=ModalLoadCase)
    settings: ModalAnalysisSettings = field(default_factory=ModalAnalysisSettings)
    periods: list[float] = field(default_factory=list)
    _is_executed: bool = field(default=False, init=False)

    def run_modal(self) -> None:  # noqa: C901  # type: ignore
        """
        Run the modal analysis.

        Raises:
          ValueError: If a solver known to not work with `eigen` is
           specified.
        """
        self.initialize_logger()
        self.log('Running a modal analysis.')

        ndf = NDF[self.model.dimensionality]

        if self.settings.disable_default_recorders is True:
            self.settings.disable_default_recorders = False
            msg = (
                'Default recorders are required for '
                'modal analysis and were force-enabled.'
            )
            self.log(msg)
            print(msg)
            # TODO(JVM): turn into a warning.

        self.log('Defining model in OpenSees.')
        self.opensees_define_model()
        self.opensees_define_recorders()
        self.opensees_define_node_restraints(
            self.load_case.fixed_supports, self.load_case.elastic_supports
        )
        self.opensees_define_node_constraints(self.load_case.rigid_diaphragm)
        self.log('Defining mass in OpenSees.')
        self.opensees_define_mass(self.load_case.mass_registry)

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
        self.recorders['default_node'].set_data(eigenvectors)
        self.log('Retrieved node eigenvectors.')

        basic_force_data = {}

        if self.settings.retrieve_basic_forces:
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
                self.opensees_define_model()
                self.opensees_define_recorders()
                self.opensees_define_node_restraints(
                    self.load_case.fixed_supports, self.load_case.elastic_supports
                )
                self.opensees_define_node_constraints(self.load_case.rigid_diaphragm)
                mode_eigenvectors = eigenvectors.loc[mode, :].copy()
                # ignore the node-dof pairs with a fixed constraint.
                to_drop = []
                for uid, support in self.load_case.fixed_supports.items():
                    for i, dof in enumerate(support):
                        if dof:
                            to_drop.append((uid, i + 1))
                mode_eigenvectors = mode_eigenvectors.drop(to_drop)
                if (
                    bool(
                        (
                            mode_eigenvectors.isna() | np.isinf(mode_eigenvectors)
                        ).any()
                    )
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

                basic_force_data[mode] = self.recorders[
                    'default_basic_force'
                ].get_data()
                basic_force_data[mode].index.name = 'mode'
                basic_force_data[mode].index = [mode]

                progress_bar.update(1)

            progress_bar.close()
            self.log('Obtained basic forces for each mode.')
            self.recorders['default_basic_force'].set_data(
                pd.concat(basic_force_data.values(), axis=0)
            )

        self.log('Analysis finished.')
        ops.wipe()
        self._is_executed = True


def ensure_minmax_level_exists_or_add(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'min/max' column level if it doesn't exist.

    Assigns everything to 'max', and duplicates columns for 'min'.

    Args:
        data: Input DataFrame with MultiIndex columns.

    Returns:
        Updated DataFrame with 'min/max' as the outermost column
        level.
    """
    columns = data.columns

    if 'min/max' not in columns.names:
        new_columns = pd.MultiIndex.from_tuples(
            [(*col, minmax) for col in columns for minmax in ('max', 'min')],
            names=[*list(columns.names), 'min/max'],
        )
        # Repeat the data for 'max' and 'min'
        repeated_data = pd.concat([data, data], axis=1)
        repeated_data.columns = new_columns
        return repeated_data

    return data


def combine_single(
    df1: pd.DataFrame, df2: pd.DataFrame, action: Literal['add', 'envelope']
) -> pd.DataFrame:
    """
    Combine two DataFrames based on the specified action.

    Args:
        df1: First DataFrame.
        df2: Second DataFrame.
        action: Action to perform:
            - 'add': Element-wise addition of the DataFrames. Missing values are treated as 0.
            - 'envelope': Take the largest of the maxes and the
              smallest of the mins.

    Returns:
        Combined DataFrame based on the action.

    Raises:
      ValueError: If an unknown action is specified.
    """
    # Reindex to ensure alignment and fill missing data with 0
    df1_aligned = df1.reindex(columns=df2.columns.union(df1.columns), fill_value=0)
    df2_aligned = df2.reindex(columns=df2.columns.union(df1.columns), fill_value=0)

    if action == 'add':
        combined = df1_aligned + df2_aligned
    elif action == 'envelope':
        df1 = ensure_minmax_level_exists_or_add(df1)
        df2 = ensure_minmax_level_exists_or_add(df2)
        if not np.all(
            df1.columns.names == df2.columns.names
        ) or not df1.columns.equals(df2.columns):
            msg = 'Cannot align DataFrames with different columns'
            raise ValueError(msg)
        max_df = pd.DataFrame(
            np.maximum(
                df1.xs('max', level='min/max', axis=1),
                df2.xs('max', level='min/max', axis=1),
            ),
            index=df1.index,
            columns=df1.xs('max', level='min/max', axis=1).columns,
        )
        min_df = pd.DataFrame(
            np.minimum(
                df1.xs('min', level='min/max', axis=1),
                df2.xs('min', level='min/max', axis=1),
            ),
            index=df1.index,
            columns=df1.xs('min', level='min/max', axis=1).columns,
        )
        combined = pd.concat([max_df, min_df], axis=1)
        combined.columns = pd.MultiIndex.from_product(
            [max_df.columns.levels[0], max_df.columns.levels[1], ['max', 'min']],
            names=[*max_df.columns.names, 'min/max'],
        )

    else:
        msg = 'Action must be one of `add` or `envelope`.'
        raise ValueError(msg)

    return combined


def combine(
    dfs: list[pd.DataFrame], action: Literal['add', 'envelope']
) -> pd.DataFrame:
    """
    Combine multiple DataFrames sequentially based on the specified action.

    Args:
        dfs: List of DataFrames to combine.
        action: Action to perform:
            - 'add': Element-wise addition of the DataFrames.
            - 'envelope': Take the largest of the maxes and the
              smallest of the mins.

    Returns:
        Combined DataFrame based on the action.

    Raises:
        ValueError: If less than two DataFrames are provided.
    """
    min_df_count = 2
    if len(dfs) < min_df_count:
        msg = 'At least two DataFrames are required to combine.'
        raise ValueError(msg)

    # Combine DataFrames sequentially
    combined_df = dfs[0]
    for df in dfs[1:]:
        combined_df = combine_single(combined_df, df, action)

    return combined_df


@dataclass(repr=False)
class AnalysisRegistry:
    """Analysis registry."""

    load_case_registry: LoadCaseRegistry
    analysis_objects: dict[Analysis] = field(default_factory=dict)

    def run_static_batch(self) -> None:
        """Run a batch of static analyses."""
        # Determine the base directory for results
        base_dir = (
            Path(self.load_case_registry.result_setup.directory)
            if self.load_case_registry.result_setup.directory
            else Path(tempfile.mkdtemp())
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        self.load_case_registry.result_setup.directory = str(base_dir.resolve())

        cases_dict = self.load_case_registry.get_load_cases()
        num_cases = len(cases_dict)
        progress_bar = tqdm(
            total=num_cases,
            ncols=80,
            desc='Processing cases',
            unit='case',
            leave=False,
        )
        for load_case_name, load_case in cases_dict.items():
            case_type = load_case.get_load_case_type()
            progress_bar.set_description(f'Processing {case_type}: {load_case_name}')
            # Create a subdirectory for each load case
            case_dir = base_dir / f'{case_type}_{load_case_name}'
            case_dir.mkdir(parents=True, exist_ok=True)
            analysis = Analysis(
                self.load_case_registry.model,
                AnalysisSettings(num_steps=1, result_directory=str(case_dir)),
            )
            analysis.run_static(self.load_case)
            self.analysis_objects[load_case_name] = analysis
            progress_bar.update(1)
        progress_bar.close()

    def get_combined_results(
        self, recorder_name: str, combination: dict[str, float]
    ) -> pd.DataFrame:
        """
        Get results for a specific load combination.

        Raises:
          ValueError: If the recorder is not found for some load case.

        Returns:
          The results.
        """
        associated_load_cases = set(combination.keys())
        load_case_objects = {
            load_case: self.load_case_registry.find_load_case_by_name(load_case)
            for load_case in associated_load_cases
        }
        all_data = []
        for load_case_name, scale_factor in combination.items():
            load_case = load_case_objects[load_case_name]
            if recorder_name not in self.analysis_objects[load_case_name].recorders:
                msg = (
                    f'Recorder not found: {recorder_name} '
                    f'for load case {load_case}.'
                )
                raise ValueError(msg)
            data = (
                self.analysis_objects[load_case_name]
                .recorders[recorder_name]
                .get_data()
            )
            if isinstance(data, list):
                scaled_data = []
                for element in data:
                    assert isinstance(element, pd.DataFrame)
                    scaled_data.append(element * scale_factor)
                all_data.append(scaled_data)
                lists = True
            else:
                all_data.append(data * scale_factor)
                lists = False
        if lists:
            return [combine(item, action='add') for item in all_data]
        return combine(all_data, action='add')

    def get_combined_basic_forces(
        self,
        recorder_name: str,
        combination: dict[str, float],
        live_load_reduction_factors: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Get combined basic forces for a specific load combination.

        Raises:
          ValueError: If the recorder is not found for some load case.

        Returns:
          The results.
        """
        associated_load_cases = set(combination.keys())
        load_case_objects = {
            load_case: self.load_case_registry.find_load_case_by_name(load_case)
            for load_case in associated_load_cases
        }
        all_data = []
        for load_case_name, scale_factor in combination.items():
            load_case = load_case_objects[load_case_name]
            if recorder_name not in self.analysis_objects[load_case_name].recorders:
                msg = (
                    f'Recorder not found: {recorder_name} '
                    f'for load case {load_case}.'
                )
                raise ValueError(msg)
            data = self.analysis_objects[load_case_name].calculate_basic_forces(
                recorder_name,
                self.load_case_registry.model.components,
                ndm=NDM[self.load_case_registry.model.dimensionality],
                num_stations=12,
            )
            scaled_data = []
            for element in data:
                assert isinstance(element, pd.DataFrame)
                if (
                    '_live' in load_case_name
                    and live_load_reduction_factors is not None
                ):
                    element_names = element.columns.get_level_values(0)
                    reduction_factors = element_names.map(
                        live_load_reduction_factors.to_dict()
                    ).fillna(1.0)
                    scaled_data.append(element * scale_factor * reduction_factors)
                else:
                    scaled_data.append(element * scale_factor)
            all_data.append(scaled_data)
        return [combine([x[i] for x in all_data], action='add') for i in range(6)]
