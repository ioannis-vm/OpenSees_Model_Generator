"""Load cases."""

from __future__ import annotations

import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from osmg.analysis.solver import Analysis, StaticAnalysis
from osmg.analysis.supports import ElasticSupport, FixedSupport
from osmg.core.common import EPSILON

if TYPE_CHECKING:
    from osmg.analysis.common import UDL, PointLoad, PointMass
    from osmg.core.model import Model2D, Model3D


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

    model: Model2D | Model3D
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

                # Update the result directory of the analysis
                load_case.analysis.settings.result_directory = str(case_dir)
                # Define the model
                load_case.analysis.define_model_in_opensees(self.model, load_case)
