"""Load cases."""

from __future__ import annotations

import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from osmg.analysis.recorders import ElementRecorder
from osmg.analysis.solver import Analysis, StaticAnalysis
from osmg.analysis.supports import ElasticSupport, FixedSupport
from osmg.core.common import EPSILON, THREE_DIMENSIONAL, TWO_DIMENSIONAL

if TYPE_CHECKING:
    from osmg.analysis.common import UDL, PointLoad, PointMass
    from osmg.core.model import Model, Model2D, Model3D


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
    load_registry: LoadRegistry = field(default_factory=LoadRegistry)
    analysis: Analysis = field(default_factory=Analysis)

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
        level_elevation = model.grid_system.get_level_elevation(level_tag)
        for node in nodes:
            if np.abs(node.coordinates[-1] - level_elevation) < EPSILON:
                if isinstance(support, FixedSupport):
                    self.fixed_supports[node.uid] = support
                elif isinstance(support, ElasticSupport):
                    self.elastic_supports[node.uid] = support
                else:
                    msg = f'Unsupported object type: {type(support)}'
                    raise TypeError(msg)

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

        udls = self.load_registry.element_udl
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
                for element in range(len(recorder.elements))
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
class DeadLoadCase(LoadCase):
    """Dead load case."""

    analysis: StaticAnalysis = field(default_factory=StaticAnalysis)


@dataclass(repr=False)
class LiveLoadCase(LoadCase):
    """Live load case."""

    analysis: StaticAnalysis = field(default_factory=StaticAnalysis)


@dataclass(repr=False)
class ModalLoadCase(LoadCase):
    """Modal load case."""


@dataclass(repr=False)
class SeismicLoadCase(LoadCase):
    """Seismic load case base class."""


@dataclass(repr=False)
class SeismicELFLoadCase(SeismicLoadCase):
    """Seismic ELF load case."""

    analysis: StaticAnalysis = field(default_factory=StaticAnalysis)


@dataclass(repr=False)
class SeismicRSLoadCase(SeismicLoadCase, HasMass):
    """Seismic RS load case."""


@dataclass(repr=False)
class SeismicTransientLoadCase(SeismicLoadCase, HasMass):
    """Seismic transient load case."""


@dataclass(repr=False)
class OtherLoadCase(LoadCase, HasMass):
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

        # Iterate over each category of load cases
        cases_list: list[tuple[str, defaultdict[str, LoadCase]]] = [
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
        for case_type, cases in cases_list:
            for key, load_case in cases.items():
                # Create a subdirectory for each load case
                case_dir = base_dir / f'{case_type}_{key}'
                case_dir.mkdir(parents=True, exist_ok=True)

                load_case.analysis.settings.result_directory = str(case_dir)
                load_case.analysis.run(self.model, load_case)
