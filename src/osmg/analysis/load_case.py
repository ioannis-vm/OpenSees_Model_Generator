"""Load cases."""

from __future__ import annotations

import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd

from osmg.analysis.common import UDL, PointMass
from osmg.analysis.recorders import ElementRecorder
from osmg.analysis.solver import Analysis, ModalAnalysis, StaticAnalysis
from osmg.analysis.supports import ElasticSupport, FixedSupport
from osmg.core.common import EPSILON, THREE_DIMENSIONAL, TWO_DIMENSIONAL
from osmg.core.osmg_collections import BeamColumnAssembly
from osmg.model_objects.element import BeamColumnElement

from tqdm import tqdm

if TYPE_CHECKING:
    from osmg.analysis.common import PointLoad
    from osmg.core.model import Model, Model2D, Model3D
    from osmg.model_objects.node import Node


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
            - 'add': Element-wise addition of the DataFrames.
            - 'envelope': Take the largest of the maxes and the
              smallest of the mins.

    Returns:
        Combined DataFrame based on the action.

    Raises:
      ValueError: If an unknown action is specified.
    """
    # Validate column compatibility
    if action == 'add':
        if not np.all(
            df1.columns.names == df2.columns.names
        ) or not df1.columns.equals(df2.columns):
            msg = 'Cannot align DataFrames with different columns'
            raise ValueError(msg)

        combined = df1 + df2

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
class LoadRegistry:
    """Load registry."""

    nodal_loads: dict[int, PointLoad] = field(default_factory=dict)
    element_udl: dict[int, UDL] = field(default_factory=dict)


@dataclass(repr=False)
class LoadCase:
    """Load case."""

    fixed_supports: dict[int, FixedSupport] = field(default_factory=dict)
    elastic_supports: dict[int, ElasticSupport] = field(default_factory=dict)
    analysis: Analysis = field(default_factory=Analysis)
    rigid_diaphragm: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def add_supports_at_level(
        self,
        model: Model2D | Model3D,
        support: ElasticSupport | FixedSupport,
        level_tag: str,
    ) -> None:
        """
        Add the given support at the specified level.

        Determines all primary nodes that have an elevation equal to
        the specified level's elevation and assigns the specified
        support to them.

        Assumes that the last coordinate of the nodes corresponds to
        elevation.

        Raises:
          TypeError: If the provided support is not a known support
            type.
        """
        nodes = list(model.nodes.values())
        level_elevation = model.grid_system.get_level(level_tag).elevation()
        for node in nodes:
            if np.abs(node.coordinates[-1] - level_elevation) < EPSILON:
                if isinstance(support, FixedSupport):
                    self.fixed_supports[node.uid] = support
                elif isinstance(support, ElasticSupport):
                    self.elastic_supports[node.uid] = support
                else:
                    msg = f'Unsupported object type: {type(support)}'
                    raise TypeError(msg)

    def define_rigid_diaphragm(
        self,
        model: Model2D | Model3D,
        primary_node: Node,
    ) -> None:
        """
        Define a rigid diaphragm using a specified parent node.

        Raises:
          ValueError: If the model dimensionality is not supported.
        """
        elevation = primary_node.coordinates[-1]
        self.rigid_diaphragm[primary_node.uid] = tuple(
            node.uid
            for node in model.nodes.values()
            if np.abs(node.coordinates[-1] - elevation) < EPSILON
            and node.uid != primary_node.uid
        )
        if model.dimensionality == '3D Frame':
            self.fixed_supports[primary_node.uid] = FixedSupport(
                (False, False, True, True, True, False)
            )
        elif model.dimensionality == '3D Truss':
            self.fixed_supports[primary_node.uid] = FixedSupport(
                (False, False, True)
            )
        elif model.dimensionality == '2D Frame':
            self.fixed_supports[primary_node.uid] = FixedSupport((False, True, True))
        elif model.dimensionality == '2D Truss':
            self.fixed_supports[primary_node.uid] = FixedSupport((False, True))
        else:
            msg = 'Unsupported model dimensionality: {model.dimensionality}'
            raise ValueError(msg)

    def calculate_basic_forces(  # noqa: C901
        self,
        recorder_name: str,
        element_lengths: dict[int, float],
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
        recorder = self.analysis.recorders.get(recorder_name)
        if recorder is None:
            msg = f'Specified recorder not available: {recorder_name}.'
            raise ValueError(msg)
        if not isinstance(recorder, ElementRecorder):
            msg = f'The specified recorder (`{recorder_name}`) is not an Element recorder.'
            raise TypeError(msg)

        if isinstance(self, HasLoads):
            udls = self.load_registry.element_udl
        else:
            udls = {}

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

        missing_elements = [e for e in recorder.elements if e not in element_lengths]
        if missing_elements:
            msg = f'Missing lengths for elements: {missing_elements}'
            raise ValueError(msg)

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
                udls[element_uid][0] if udls.get(element_uid) is not None else 0.00
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
                udls[element_uid][1] if udls.get(element_uid) is not None else 0.00
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
                    udls[element_uid][2]
                    if udls.get(element_uid) is not None
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

        return axial_df, shear_y_df, shear_z_df, torsion_df, moment_y_df, moment_z_df


@dataclass(repr=False)
class HasMass:
    """Parent class for load cases that have a mass registry."""

    mass_registry: dict[int, PointMass] = field(default_factory=dict)


@dataclass(repr=False)
class HasLoads:
    """Parent class for load cases that have a load registry."""

    load_registry: LoadRegistry = field(default_factory=LoadRegistry)


@dataclass(repr=False)
class DeadLoadCase(LoadCase, HasLoads):
    """Dead load case."""

    analysis: StaticAnalysis = field(default_factory=StaticAnalysis)


@dataclass(repr=False)
class LiveLoadCase(LoadCase, HasLoads):
    """Live load case."""

    analysis: StaticAnalysis = field(default_factory=StaticAnalysis)


@dataclass(repr=False)
class ModalLoadCase(LoadCase, HasMass):
    """Modal load case."""

    analysis: ModalAnalysis = field(default_factory=ModalAnalysis)


@dataclass(repr=False)
class SeismicLoadCase(LoadCase, HasLoads):
    """Seismic load case base class."""


@dataclass(repr=False)
class SeismicELFLoadCase(SeismicLoadCase, HasLoads):
    """Seismic ELF load case."""

    analysis: StaticAnalysis = field(default_factory=StaticAnalysis)


@dataclass(repr=False)
class SeismicRSLoadCase(SeismicLoadCase, HasLoads, HasMass):
    """Seismic RS load case."""


@dataclass(repr=False)
class SeismicTransientLoadCase(SeismicLoadCase, HasLoads, HasMass):
    """Seismic transient load case."""


@dataclass(repr=False)
class OtherLoadCase(LoadCase, HasLoads, HasMass):
    """Other load case."""


@dataclass(repr=False)
class AnalysisResultSetup:
    """
    Analysis result setup.

    Configures the analysis result storage setup.
    """

    directory: str | None = field(default=None)


@dataclass(repr=False)
class LoadCaseRegistry:
    """
    Load case registry.

    A load case registry is an organized collection of load cases.
    Load cases are categorized based on the nature of the loads, such
    as `dead`, `live`, or `seismic_elf`. Each type of loading
    necessitates a specific type of analysis needed to estimate the
    structural response, with most being static. Custom analyses which
    don't need to conform to load type classification can use the
    `other` load case.

    The load case registry can be used to orchestrate all analyses,
    retrieve, and post-process results.

    """

    model: Model
    result_setup: AnalysisResultSetup = field(default_factory=AnalysisResultSetup)
    dead: defaultdict[str, DeadLoadCase] = field(
        default_factory=lambda: defaultdict(DeadLoadCase)
    )
    live: defaultdict[str, LiveLoadCase] = field(
        default_factory=lambda: defaultdict(LiveLoadCase)
    )
    modal: defaultdict[str, ModalLoadCase] = field(
        default_factory=lambda: defaultdict(ModalLoadCase)
    )
    seismic_elf: defaultdict[str, SeismicELFLoadCase] = field(
        default_factory=lambda: defaultdict(SeismicELFLoadCase)
    )
    seismic_rs: defaultdict[str, SeismicRSLoadCase] = field(
        default_factory=lambda: defaultdict(SeismicRSLoadCase)
    )
    seismic_transient: defaultdict[str, SeismicTransientLoadCase] = field(
        default_factory=lambda: defaultdict(SeismicTransientLoadCase)
    )
    other: defaultdict[str, OtherLoadCase] = field(
        default_factory=lambda: defaultdict(OtherLoadCase)
    )

    def get_cases_list(self) -> list[tuple[str, defaultdict[str, LoadCase]]]:
        """
        Get a list of load case types.

        Returns:
          List of load case types
        """
        # Iterate over each category of load cases
        return [
            ('dead', cast(defaultdict[str, LoadCase], self.dead)),
            ('live', cast(defaultdict[str, LoadCase], self.live)),
            ('modal', cast(defaultdict[str, LoadCase], self.modal)),
            ('seismic_elf', cast(defaultdict[str, LoadCase], self.seismic_elf)),
            ('seismic_rs', cast(defaultdict[str, LoadCase], self.seismic_rs)),
            (
                'seismic_transient',
                cast(defaultdict[str, LoadCase], self.seismic_transient),
            ),
            ('other', cast(defaultdict[str, LoadCase], self.other)),
        ]

    def self_weight(self, case_name: str, scaling_factor: float = 1.0) -> None:
        """
        Define self weight.

        Define self weight based on the properties of the sections of
        BeamColumn elements.

        Params:
          case_name: Name of the load case to be created.
          scaling_factor: Self-weight scaling factor to use.
        """
        # get all beamcolumn elements
        elements = [
            element
            for component in self.model.components.values()
            if isinstance(component, BeamColumnAssembly)
            for element in component.elements.values()
            if isinstance(element, BeamColumnElement)
        ]
        for element in elements:
            weight_per_length = element.section.sec_w * scaling_factor
            udl = UDL((0.00, 0.00, -weight_per_length))
            self.dead[case_name].load_registry.element_udl[element.uid] = udl

    def self_mass(
        self,
        target_load_case: HasMass,
        source_load_cases: list[tuple[HasLoads, float]],
        g_constant: float,
    ) -> None:
        """
        Define self mass.

        Define self weight based on the properties of the sections of
        BeamColumn elements.

        Params:
          target_load_case: Load case to assign mass to.
          source_load_cases: Load cases to consider when calculating
                             self-mass. The float in the tuple acts as
                             a scaling coefficient.
          g_constant: Factor to convert loads to mass.
        """
        # get all beamcolumn elements
        elements = {
            element.uid: element
            for component in self.model.components.values()
            if isinstance(component, BeamColumnAssembly)
            for element in component.elements.values()
            if isinstance(element, BeamColumnElement)
        }
        for source_load_case, factor in source_load_cases:
            # Convert UDL to mass
            for uid, udl in source_load_case.load_registry.element_udl.items():
                element = elements[uid]
                length = element.clear_length()
                weight = np.abs(udl[-1] * length)
                mass = weight * factor / g_constant / 2.0
                # TODO(JVM): separate cases for other ndm/ndf
                # configurations.
                point_mass = PointMass((mass, mass, mass, 0.00, 0.00, 0.00))
                for node in element.nodes:
                    if node.uid not in target_load_case.mass_registry:
                        target_load_case.mass_registry[node.uid] = point_mass
                    else:
                        existing_mass = target_load_case.mass_registry[node.uid]
                        target_load_case.mass_registry[node.uid] = PointMass(
                            (*(e + p for e, p in zip(existing_mass, point_mass)),)
                        )

            # Convert point loads to mass
            for (
                uid,
                point_load,
            ) in source_load_case.load_registry.nodal_loads.items():
                mass = np.abs(point_load[-1] * factor / g_constant)
                point_mass = PointMass((mass, mass, mass, 0.00, 0.00, 0.00))
                if uid not in target_load_case.mass_registry:
                    target_load_case.mass_registry[uid] = point_mass
                else:
                    existing_mass = target_load_case.mass_registry[uid]
                    target_load_case.mass_registry[uid] = PointMass(
                        (*(e + p for e, p in zip(existing_mass, point_mass)),)
                    )

    def run(self) -> None:
        """
        Run all analyses.

        This function organizes analyses by load case type and assigns
        a results directory for each load case. If no results directory
        is specified, a temporary directory is created.
        """
        # Determine the base directory for results
        base_dir = (
            Path(self.result_setup.directory)
            if self.result_setup.directory
            else Path(tempfile.mkdtemp())
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        self.result_setup.directory = str(base_dir.resolve())

        cases_list = self.get_cases_list()
        num_cases = 0
        for _, cases in cases_list:
            num_cases += len(cases)
        progress_bar = tqdm(
            total=num_cases,
            ncols=80,
            desc='Processing cases',
            unit='case',
            leave=False,
        )
        for case_type, cases in cases_list:
            for key, load_case in cases.items():
                progress_bar.set_description(f'Processing {case_type}: {key}')
                # Create a subdirectory for each load case
                case_dir = base_dir / f'{case_type}_{key}'
                case_dir.mkdir(parents=True, exist_ok=True)

                load_case.analysis.settings.result_directory = str(case_dir)
                load_case.analysis.run(self.model, load_case)

                progress_bar.update(1)
        progress_bar.close()

    def combine_recorder(self, recorder_name: str) -> pd.DataFrame:
        """
        Combine results of a recorder across cases.

        Returns:
          Combined results.

        Raises:
          ValueError: If the specified recorder does not exist in some
            load case.
        """
        # TODO(JVM): in progress.
        cases_list = [
            ('dead', cast(defaultdict[str, LoadCase], self.dead)),
            ('live', cast(defaultdict[str, LoadCase], self.live)),
        ]
        all_data: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
        case_type_data = {}
        for case_type, cases in cases_list:
            for key, load_case in cases.items():
                if recorder_name not in load_case.analysis.recorders:
                    msg = (
                        f'Recorder `{recorder_name}` not '
                        f'found in `{case_type}` `{key}`.'
                    )
                    raise ValueError(msg)
                all_data[case_type][key] = load_case.analysis.recorders[
                    recorder_name
                ].get_data()
        # we have all data here.
        for case_type, dataframes in all_data.items():
            case_type_data[case_type] = combine(list(dataframes.values()), 'add')
        # Hard-coded factors for now.
        return case_type_data['dead']
