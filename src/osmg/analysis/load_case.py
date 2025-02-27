"""Load cases."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import scipy as sp

from osmg.analysis.common import UDL, PointLoad, PointMass
from osmg.analysis.supports import ElasticSupport, FixedSupport
from osmg.core.common import EPSILON, NDF
from osmg.core.osmg_collections import BeamColumnAssembly

if TYPE_CHECKING:
    from osmg.core.model import Model, Model2D, Model3D
    from osmg.model_objects.node import Node


@dataclass(repr=False)
class LoadRegistry:
    """Load registry."""

    nodal_loads: dict[int, PointLoad] = field(default_factory=dict)
    component_udl: dict[int, UDL] = field(default_factory=dict)


@dataclass(repr=False)
class HasModel:
    """Has a model object."""

    model: Model | None = field(default=None)

    def __post_init__(self) -> None:
        """
        Post-initialization.

        Raises:
          ValueError: If the `model` attribute is set to None after
          initialization.
        """
        if self.model is None:
            msg = 'Model is a required attribute.'
            raise ValueError(msg)


@dataclass(repr=False)
class LoadCase(HasModel):
    """Load case."""

    model: Model | None = field(default=None)
    fixed_supports: dict[int, FixedSupport] = field(default_factory=dict)
    elastic_supports: dict[int, ElasticSupport] = field(default_factory=dict)
    rigid_diaphragm: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-initialization."""
        super().__init__()
        self._case_type = 'Undefined'

    def get_load_case_type(self) -> str:
        """
        Get the case type.

        Returns:
          The case type.
        """
        return self._case_type

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


@dataclass(repr=False)
class HasMass:
    """Parent class for load cases that have a mass registry."""

    mass_registry: dict[int, PointMass] = field(default_factory=dict)


@dataclass(repr=False)
class HasLoads:
    """Parent class for load cases that have a load registry."""

    load_registry: LoadRegistry = field(default_factory=LoadRegistry)


@dataclass(repr=False)
class StaticLoadCase(LoadCase, HasLoads):
    """Static load case."""

    def __post_init__(self) -> None:
        """Post-initialization."""
        self._case_type = 'Static'


@dataclass(repr=False)
class ModalLoadCase(LoadCase, HasMass):
    """Modal load case."""

    def __post_init__(self) -> None:
        """Post-initialization."""
        self._case_type = 'Modal'


@dataclass(repr=False)
class SeismicLoadCase(LoadCase, HasLoads):
    """Seismic load case base class."""

    def __post_init__(self) -> None:
        """Post-initialization."""
        self._case_type = 'Seismic'


@dataclass(repr=False)
class SpectrumLoadCase:
    """Involves a response spectrum."""

    _design_spectrum: pd.DataFrame | None = field(default=None)

    def define_design_spectrum_from_csv(self, filepath: str) -> None:
        """Load a design spectrum from a CSV file."""
        self._design_spectrum = pd.read_csv(filepath, index_col=0, header=0)

    def interpolate_spectrum(self, period: float) -> float:
        """
        Obtain Sa for a given T.

        Interpolates the design spectrum to obtain the spectral
        acceleration (Sa) for a given period.

        Args:
            period (float): The period T for which to calculate the
            spectral acceleration.

        Returns:
            float: The interpolated spectral acceleration (Sa) at the
            given period T.
        """
        periods = self._design_spectrum.index.to_numpy()
        spectral_accelerations = self._design_spectrum['Sa(g)'].to_numpy()

        return float(np.interp(period, periods, spectral_accelerations))


@dataclass(repr=False)
class SeismicELFLoadCase(SeismicLoadCase, HasLoads, SpectrumLoadCase):
    """Seismic ELF load case."""

    _seismic_weight: dict[int, float] = field(default_factory=dict)
    _metadata: list[str] = field(default_factory=list)

    def extract_seismic_weight(
        self, modal_load_case: ModalLoadCase, g_constant: float
    ) -> None:
        """Extract seismic weight from a modal load case."""
        mass_registry = modal_load_case.mass_registry
        for node_uid, point_mass in mass_registry.items():
            mass_value = point_mass[0]
            self._seismic_weight[node_uid] = mass_value * g_constant

    def define_loads(
        self,
        response_modification_factor: float,
        importance_factor: float,
        first_mode_period: float,
        sd1: float,
        structural_height: float,
        approximate_period_parameters: tuple[float, float],
        direction: tuple[float, ...],
        base_elevation: float = 0.00,
        length_to_feet_factor: float = 1.00,
    ) -> None:
        """
        Calculate and distribute equivalent lateral forces.

        Params:
            response_modification_factor: $R$, Table 12.2-1.
            overstrength_factor: $Omega_0$, Table 12.2-1.
            deflection_amplification_factor: $C_d$, Table 12.2-1.
            importance_factor: $I_e$, Table 1.5-2.
            approximate_period_parameters: $C_t$ and $x$ from Table 12.8-2.
            first_mode_period: From modal analysis.
            sd1: From site-specific hazard.
            structural_height: Height in ft.
            direction: Vector (as tuple) defining the direction of
              loading. It should be a normal vector.
            base_elevation: Elevation of the base level. Nodes below
              don't get loaded.
            length_to_feet_factor: What to multiply to convert length
              unit used by model to feet.
        """
        all_nodes = self.model.get_all_nodes()
        c_t, x_param = approximate_period_parameters
        # Equation 12.8-8
        approximate_period = c_t * structural_height**x_param
        self._metadata.append(f'Approx. period = {approximate_period:.2f} s.')

        cu_ifun = sp.interpolate.interp1d(
            np.array((0.4, 0.3, 0.2, 0.15, 0.1)),
            np.array((1.4, 1.4, 1.5, 1.6, 1.7)),
            kind='linear',
            fill_value='extrapolate',
        )
        cu_value = float(cu_ifun(sd1))
        self._metadata.append(f'Cu = {cu_value:.2f}.')
        max_period = cu_value * approximate_period
        self._metadata.append(f'Tmax = {max_period:.2f} s.')
        self._metadata.append(f'T1 = {first_mode_period:.2f} s.')
        controling_period = np.minimum(first_mode_period, max_period)
        s_a = self.interpolate_spectrum(controling_period)
        self._metadata.append(f'Sa(T) = {s_a:.2f} g.')
        c_s = s_a / (response_modification_factor / importance_factor)
        weight = np.sum(list(self._seismic_weight.values()))
        self._metadata.append(f'W = {weight:.0f}.')
        v_b = c_s * weight
        self._metadata.append(f'Vb = {v_b:.0f}.')
        exponent_ifun = sp.interpolate.interp1d(
            np.array((0.5, 2.5)),
            np.array((1.0, 2.0)),
            kind='linear',
            fill_value='extrapolate',
        )
        exponent_value = float(exponent_ifun(controling_period))
        nodal_cvx_value: dict[int, float] = {}
        for node_uid, weight_value in self._seismic_weight.items():
            node_elevation = all_nodes[node_uid].coordinates[-1] - base_elevation
            if node_elevation < 0.00:
                continue
            nodal_cvx_value[node_uid] = (
                weight_value
                * (node_elevation * length_to_feet_factor) ** exponent_value
            )
        total_cvx = np.sum(list(nodal_cvx_value.values()))
        for key in nodal_cvx_value:
            nodal_cvx_value[key] *= v_b / total_cvx
        # Now nodal_cvx holds the absolute nodal forces.
        for node_uid, nodal_force in nodal_cvx_value.items():
            self.load_registry.nodal_loads[node_uid] = PointLoad(
                v * nodal_force for v in direction
            )

    def get_metadata(self) -> None:
        """Get the metadata."""
        return self._metadata


@dataclass
class SeismicRSAnalysisResults:
    """Stores Seismic RS related results."""

    gamma_n: tuple[float, ...]
    m_star: tuple[float, ...]
    vb_modal: tuple[float, ...]
    modal_q: tuple[float, ...]
    total_mass: float


@dataclass(repr=False)
class SeismicRSLoadCase(SeismicLoadCase, SpectrumLoadCase):
    """Seismic response spectrum load case."""

    _direction: int | None = field(default=None)
    _g_constant: float | None = field(default=None)
    _linked_modal_load_case: ModalLoadCase | None = field(default=None)
    _results: SeismicRSAnalysisResults | None = field(default=None)

    def configure(
        self,
        *,
        direction: Literal[0, 1, 2],
        g_constant: float,
        linked_modal_load_case: ModalLoadCase,
    ) -> None:
        """Define the excitation direction."""
        assert direction in {0, 1, 2}, f'Invalid direction: {direction}.'
        self._direction = direction
        self._g_constant = g_constant
        self._link_modal_load_case(linked_modal_load_case)

    def _link_modal_load_case(self, modal_load_case: ModalLoadCase) -> None:
        """Link a modal load case."""
        self.analysis = modal_load_case.analysis
        self.fixed_supports = modal_load_case.fixed_supports
        self.elastic_supports = modal_load_case.elastic_supports
        self.analysis = modal_load_case.analysis
        self.rigid_diaphragm = modal_load_case.rigid_diaphragm
        self.mass_registry = modal_load_case.mass_registry
        self._linked_modal_load_case = modal_load_case

    def calculate_modal_participation_factors(self) -> None:
        """
        Calculate modal participation factors.

        Code adapted from - https://portwooddigital.com/
          2020/11/01/modal-participation-factors/
        - Thanks

        Raises:
          ValueError: If the modal analysis does not exist or has not
            been executed yet.
          ValueError: If no spectrum is set.
          ValueError: If no direction is set.
        """
        if self._linked_modal_load_case is None:
            msg = (
                'Seismic RS analysis requires linking '
                'to an existing modal load case.'
            )
            raise ValueError(msg)

        if self._design_spectrum is None:
            msg = 'Seismic RS analysis requires a spectrum.'
            raise ValueError(msg)

        if self._direction is None:
            msg = 'Seismic RS analysis requires an excitation direction to be set.'
            raise ValueError(msg)

        if self._g_constant is None:
            msg = 'Seismic RS analysis requires G to be set (`g_constant`).'
            raise ValueError(msg)

        periods = self._linked_modal_load_case.analysis.periods
        num_modes = len(periods)
        if num_modes == 0:
            msg = 'Modal analysis has not been executed yet.'
            raise ValueError(msg)
        nodes = self._linked_modal_load_case.model.get_all_nodes()
        node_displacements = self.analysis.recorders['default_node'].get_data()
        ndf = NDF[self._linked_modal_load_case.model.dimensionality]

        node_displacements_ordered = node_displacements[nodes.keys()]
        displacements = node_displacements_ordered.to_numpy()

        num_nodes = len(nodes)
        mass_matrix = np.zeros((num_nodes, ndf))

        for i, node_uid in enumerate(nodes):
            node_mass = self.mass_registry.get(node_uid)
            if node_mass:
                mass_matrix[i, :] = node_mass

        total_mass = mass_matrix[:, self._direction].sum()

        displacements = displacements.reshape(
            (displacements.shape[0], num_nodes, ndf)
        )

        g_constant = self._g_constant
        m_stars = []
        gamma_ns = []
        vb_modal = []
        modal_q = []

        for n_mode in range(num_modes):
            mode_displacements = displacements[n_mode]
            l_n = (
                mode_displacements[:, self._direction]
                * mass_matrix[:, self._direction]
            ).sum()
            m_n = (mode_displacements**2 * mass_matrix).sum()

            gamma_n = l_n / m_n
            m_star = l_n**2 / m_n
            gamma_ns.append(float(gamma_n))
            m_stars.append(float(m_star))
            sa_t = self.interpolate_spectrum(periods[n_mode])
            vb_modal.append(float(sa_t * m_star * g_constant))
            modal_q.append(
                float(
                    gamma_n
                    * sa_t
                    / (2.0 * np.pi / periods[n_mode]) ** 2
                    * g_constant
                )
            )

        self._results = SeismicRSAnalysisResults(
            gamma_n=tuple(gamma_ns),
            m_star=tuple(m_stars),
            vb_modal=tuple(vb_modal),
            modal_q=tuple(modal_q),
            total_mass=total_mass,
        )

    def run(self) -> None:
        """Run associated analysis."""
        if not self._is_executed:
            self.calculate_modal_participation_factors()
            self._is_executed = True


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


class LoadCaseRegistry:
    """
    Load case registry.

    A load case registry is an organized collection of load cases.
    Load cases are categorized based on the type of analysis required,
    such as `static`, or `modal`. Custom analyses which don't need to
    conform to load type classification can use the `other` load case.
    """

    def __init__(
        self, model: Model, result_setup: AnalysisResultSetup = None
    ) -> None:
        """Instantiate a LoadCaseRegistry."""
        self.model = model
        self.result_setup = result_setup or AnalysisResultSetup()

        # Initialize defaultdicts with factory functions that include the model
        self.static = defaultdict(lambda: StaticLoadCase(model=self.model))
        self.modal = defaultdict(lambda: ModalLoadCase(model=self.model))
        self.seismic_elf = defaultdict(lambda: SeismicELFLoadCase(model=self.model))
        self.seismic_rs = defaultdict(lambda: SeismicRSLoadCase(model=self.model))
        self.seismic_transient = defaultdict(
            lambda: SeismicTransientLoadCase(model=self.model)
        )
        self.other = defaultdict(lambda: OtherLoadCase(model=self.model))

    def self_weight(self, case_name: str, scaling_factor: float = 1.0) -> None:
        """
        Define self weight.

        Define self weight based on the properties of the sections of
        BeamColumn elements.

        Params:
          case_name: Name of the load case to be created.
          scaling_factor: Self-weight scaling factor to use.
        """
        # get all beamcolumn assemblies
        components = [
            component
            for component in self.model.components.values()
            if isinstance(component, BeamColumnAssembly)
        ]
        for component in components:
            weight_per_length = component.get_section().sec_w * scaling_factor
            udl = UDL((0.00, 0.00, -weight_per_length))
            self.static[case_name].load_registry.component_udl[component.uid] = udl

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
        # get all BeamColumn assemblies
        components = {
            component.uid: component
            for component in self.model.components.values()
            if isinstance(component, BeamColumnAssembly)
        }
        for source_load_case, factor in source_load_cases:
            # Convert UDL to mass
            for uid, udl in source_load_case.load_registry.component_udl.items():
                component = components[uid]
                length = component.clear_length()
                weight = np.abs(udl[-1] * length)
                num_nodes = len(component.external_nodes)
                assert num_nodes > 0, 'Invalid component: no external nodes.'
                mass = weight * factor / g_constant / num_nodes
                # TODO(JVM): separate cases for other ndm/ndf
                # configurations.
                point_mass = PointMass((mass, mass, mass, 0.00, 0.00, 0.00))
                for node_uid in component.external_nodes:
                    if node_uid not in target_load_case.mass_registry:
                        target_load_case.mass_registry[node_uid] = point_mass
                    else:
                        existing_mass = target_load_case.mass_registry[node_uid]
                        target_load_case.mass_registry[node_uid] = PointMass(
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

    def get_load_cases(self) -> dict[str, LoadCase]:
        """
        Get a dictionary of load cases.

        Returns:
          Dictionary of load cases.
        """
        return (
            self.static
            | self.modal
            | self.seismic_elf
            | self.seismic_rs
            | self.seismic_transient
            | self.other
        )

    def get_load_case_list(self) -> list[LoadCase]:
        """
        Get a list of load cases.

        Returns:
          List of load cases.
        """
        return list(self.get_load_cases().values())

    def find_load_case_by_name(self, load_case: str) -> LoadCase | SeismicRSLoadCase:
        """
        Find a load case by name.

        Returns:
          The load case.

        Raises:
          ValueError: If the load case is not found.
        """
        load_cases = self.get_load_cases()
        if load_case not in load_cases:
            msg = f'Load case not found: {load_case}.'
            raise ValueError(msg)
        return (load_cases | self.seismic_rs)[load_case]
