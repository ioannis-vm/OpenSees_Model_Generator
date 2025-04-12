"""Defines OpenSees Element interfrace objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from osmg.core.uid_object import UIDObject
from osmg.graphics.visibility import ElementVisibility

if TYPE_CHECKING:
    from osmg.core.common import numpy_array
    from osmg.mesh import Mesh
    from osmg.model_objects.node import Node
    from osmg.model_objects.section import ElasticSection, FiberSection
    from osmg.model_objects.uniaxial_material import UniaxialMaterial
    from osmg.model_objects.friction_model import Coulomb


@dataclass(repr=False)
class Element(UIDObject):
    """
    OpenSees element.

    https://openseespydoc.readthedocs.io/en/latest/src/element.html

    Attributes:
    ----------
        parent_component:
          the parent component assembly that this element belongs to.
        uid: the unique identifier of this element.
        nodes: the list of nodes that this element connects.

    """

    nodes: list[Node]

    def __post_init__(self) -> None:
        """Post-initialization."""
        self.visibility = ElementVisibility()
        super().__post_init__()


@dataclass(repr=False)
class ZeroLength(Element):
    """
    OpenSees ZeroLength element.

    https://openseespydoc.readthedocs.io/en/latest/src/ZeroLength.html

    """

    materials: list[UniaxialMaterial]
    directions: list[int]
    vecx: numpy_array | None
    vecyp: numpy_array | None
    enable_rayleigh: bool = field(default=True)

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        output = [
            'zeroLength',
            self.uid,
            *[n.uid for n in self.nodes],
            '-mat',
            *[m.uid for m in self.materials],
            '-dir',
            *self.directions,
        ]
        if self.enable_rayleigh:
            output.extend(
                [
                    '-doRayleigh',
                    1,
                ]
            )
        else:
            output.extend(
                [
                    '-doRayleigh',
                    0,
                ]
            )
        if self.vecx is not None or self.vecyp is not None:
            assert self.vecx is not None
            assert self.vecyp is not None
            output.extend(
                [
                    '-orient',
                    *self.vecx,
                    *self.vecyp,
                ]
            )

        return output

    def __repr__(self) -> str:
        """
        Get string representation.

        Returns:
          The string representation of the object.
        """
        res = ''
        res += 'ZeroLength element object\n'
        res += f'uid: {self.uid}'
        res += 'Materials:'
        for mat, direction in zip(self.materials, self.directions):
            res += f'  {direction}: {mat.name}\n'
        res += f'vecx: {self.vecx}\n'
        res += f'vecyp: {self.vecyp}\n'
        return res


@dataclass(repr=False)
class TwoNodeLink(Element):
    """
    OpenSees TwoNodeLink element.

    https://openseespydoc.readthedocs.io/en/latest/src/twoNodeLink.html

    """

    materials: list[UniaxialMaterial]
    directions: list[int]
    vecyp: numpy_array | None = None

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        args = [
            'twoNodeLink',
            self.uid,
            *[n.uid for n in self.nodes],
            '-mat',
            *[m.uid for m in self.materials],
            '-dir',
            *self.directions,
        ]
        if self.vecyp is not None:
            args.extend(
                [
                    '-orient',
                    *self.vecyp,
                ]
            )
        return args

    def __repr__(self) -> str:
        """
        Get string representation.

        Returns:
          The string representation of the object.
        """
        res = ''
        res += 'TwoNodeLink element object\n'
        res += f'uid: {self.uid}'
        res += 'Materials:'
        for mat, direction in zip(self.materials, self.directions):
            res += f'  {direction}: {mat.name}\n'
        if self.vecyp:
            res += f'vecyp: {self.vecyp}\n'
        return res


@dataclass(repr=False)
class Bar(Element):
    """
    Truss and Corotational Truss.

    Implements both of the following:
    OpenSees Truss Element
    https://openseespydoc.readthedocs.io/en/latest/src/trussEle.html
    OpenSees Corotational Truss Element
    https://openseespydoc.readthedocs.io/en/latest/src/corotTruss.html

    Params:
      transf_type: Linear or Corotational, determines what type of truss to use.
      area: Cross-sectional area.
      mat: Uniaxial material (for stress-strain relationship).
      outside_shape: Mesh defining the cross-section shape, used only for plots.
      weight_per_length: Weight per unit length, used when assigning self-weight.
      rho: Mass per unit length.
      cFlag: 0: lumped mass, 1: consistent mass.
      rFlag: 0: No Rayleigh damping, 1: Include Rayleigh damping.
    """

    transf_type: Literal['Linear', 'Corotational']
    area: float
    material: UniaxialMaterial
    outside_shape: Mesh | None = field(default=None)
    weight_per_length: float = field(default=0.00)
    rho: float = field(default=0.00)
    cflag: int = field(default=0)
    rflag: int = field(default=0)

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        elm_name = {'Linear': 'Truss', 'Corotational': 'corotTruss'}

        return [
            elm_name[self.transf_type],
            self.uid,
            self.nodes[0].uid,
            self.nodes[1].uid,
            self.area,
            self.material.uid,
            '-rho',
            self.rho,
            '-cMass',
            self.cflag,
            '-doRayleigh',
            self.rflag,
        ]

    def clear_length(self) -> float:
        """
        Clear length.

        Returns the clear length of the element (without the rigid
        offsets)

        Returns:
          The clear length.
        """
        p_i = np.array(self.nodes[0].coordinates)
        p_j = np.array(self.nodes[1].coordinates)
        return float(np.linalg.norm(p_i - p_j))

    def __repr__(self) -> str:
        """
        Get string representation.

        Returns:
          The string representation of the object.
        """
        elm_name = {'Linear': 'Truss', 'Corotational': 'corotTruss'}
        res = ''
        res += f'{elm_name[self.transf_type]} element object\n'
        res += f'uid: {self.uid}\n'
        res += f'node_i.uid: {self.nodes[0].uid}\n'
        res += f'node_j.uid: {self.nodes[1].uid}\n'
        res += f'node_i.coordinates: {self.nodes[0].coordinates}\n'
        res += f'node_j.coordinates: {self.nodes[1].coordinates}\n'
        return res


@dataclass(repr=False)
class GeomTransf(UIDObject):
    """
    Geometric transformation.

    `y_axis` = None implies 2D space.

    https://openseespydoc.readthedocs.io/en/latest/src/ZeroLength.html
    https://openseespydoc.readthedocs.io/en/latest/src/geomTransf.html?highlight=geometric%20transformation

    """

    transf_type: Literal['Linear', 'Corotational', 'PDelta']
    offset_i: numpy_array
    offset_j: numpy_array
    x_axis: numpy_array
    y_axis: numpy_array | None
    z_axis: numpy_array

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        if self.y_axis is not None:
            # 3D case
            return [
                self.transf_type,
                self.uid,
                *self.z_axis,
                '-jntOffset',
                *self.offset_i,
                *self.offset_j,
            ]
        # 2D case
        return [
            self.transf_type,
            self.uid,
            '-jntOffset',
            *self.offset_i,
            *self.offset_j,
        ]


@dataclass(repr=False)
class ModifiedStiffnessParameterConfig:
    """Configuration parameters for ModifiedElasticBeam elements."""

    n_x: float | None
    n_y: float | None


@dataclass(repr=False)
class BeamColumnElement(Element):
    """Beamcolumn element."""

    section: ElasticSection
    geomtransf: GeomTransf

    def clear_length(self) -> float:
        """
        Clear length.

        Returns the clear length of the element (without the rigid
        offsets)

        Returns:
          The clear length.
        """
        p_i = np.array(self.nodes[0].coordinates) + self.geomtransf.offset_i
        p_j = np.array(self.nodes[1].coordinates) + self.geomtransf.offset_j
        return float(np.linalg.norm(p_i - p_j))


@dataclass(repr=False)
class ElasticBeamColumn(BeamColumnElement):
    """
    OpenSees Elastic Beam Column Element.

    https://openseespydoc.readthedocs.io/en/latest/src/elasticBeamColumn.html
    """

    modified_stiffness_config: ModifiedStiffnessParameterConfig | None = field(
        default=None
    )

    @staticmethod
    def _calculate_stiffness(n: float) -> tuple[float, float, float]:
        """
        Calculate stiffness parameters based on the given n value.

        Args:
            n: The stiffness parameter (n_x or n_y).

        Returns:
            A tuple containing (k11, k33, k44).
        """
        k44 = 6.0 * (1.0 + n) / (2.0 + 3.0 * n)
        k11 = (1.0 + 2.0 * n) * k44 / (1.0 + n)
        k33 = k11
        return k11, k33, k44

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
            The OpenSees arguments.
        """
        mod_params = self.modified_stiffness_config
        if mod_params:
            stiffness_x = (
                self._calculate_stiffness(mod_params.n_x)
                if mod_params.n_x is not None
                else None
            )
            stiffness_y = (
                self._calculate_stiffness(mod_params.n_y)
                if mod_params.n_y is not None
                else None
            )

            args = [
                'ModElasticBeam3d',
                self.uid,
                self.nodes[0].uid,
                self.nodes[1].uid,
                self.section.area,
                self.section.e_mod,
                self.section.g_mod,
                self.section.j_mod,
                self.section.i_y,
                self.section.i_x,
            ]

            # Append stiffness parameters based on availability
            if stiffness_x and not stiffness_y:
                args.extend([4.00, 4.00, 2.00, *stiffness_x, self.geomtransf.uid])
            elif stiffness_y and not stiffness_x:
                args.extend([*stiffness_y, 4.00, 4.00, 2.00, self.geomtransf.uid])
            elif stiffness_x and stiffness_y:
                args.extend([*stiffness_y, *stiffness_x, self.geomtransf.uid])

            return args

        # Default elastic beam column configuration if no modified stiffness
        return [
            'elasticBeamColumn',
            self.uid,
            self.nodes[0].uid,
            self.nodes[1].uid,
            self.section.area,
            self.section.e_mod,
            self.section.g_mod,
            self.section.j_mod,
            self.section.i_y,
            self.section.i_x,
            self.geomtransf.uid,
        ]

    def ops_args_2d(self) -> list[object]:
        """
        Obtain the OpenSees arguments for a 2D model.

        Returns:
            The OpenSees arguments.
        """
        mod_params = self.modified_stiffness_config
        if mod_params:
            msg = 'Not implemented yet.'
            raise NotImplementedError(msg)

        # Default elastic beam column configuration if no modified stiffness
        return [
            'elasticBeamColumn',
            self.uid,
            self.nodes[0].uid,
            self.nodes[1].uid,
            self.section.area,
            self.section.e_mod,
            self.section.i_x,
            self.geomtransf.uid,
        ]

    def __repr__(self) -> str:
        """
        Get string representation.

        Returns:
          The string representation of the object.
        """
        res = ''
        res += 'elasticBeamColumn element object\n'
        res += f'uid: {self.uid}\n'
        res += f'node_i.uid: {self.nodes[0].uid}\n'
        res += f'node_j.uid: {self.nodes[1].uid}\n'
        res += f'node_i.coordinates: {self.nodes[0].coordinates}\n'
        res += f'node_j.coordinates: {self.nodes[1].coordinates}\n'
        res += f'offset_i: {self.geomtransf.offset_i}\n'
        res += f'offset_j: {self.geomtransf.offset_j}\n'
        res += f'x_axis: {self.geomtransf.x_axis}\n'
        res += f'y_axis: {self.geomtransf.y_axis}\n'
        res += f'z_axis: {self.geomtransf.z_axis}\n'
        res += f'section.name: {self.section.name}\n'
        return res


@dataclass
class BeamIntegration(UIDObject):
    """
    OpenSees beamIntegration parent class.

    https://openseespydoc.readthedocs.io/en/latest/src/beamIntegration.html?highlight=beamintegration

    """

    parent_section: ElasticSection | FiberSection = field(repr=False)


@dataclass
class Lobatto(BeamIntegration):
    """
    OpenSees Lobatto beam integration.

    https://openseespydoc.readthedocs.io/en/latest/src/Lobatto.html

    """

    n_p: int

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return ['Lobatto', self.uid, self.parent_section.uid, self.n_p]


@dataclass
class DispBeamColumn(BeamColumnElement):
    """
    OpenSees dispBeamColumn element.

    https://openseespydoc.readthedocs.io/en/latest/src/ForceBeamColumn.html

    """

    integration: BeamIntegration

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return [
            'dispBeamColumn',
            self.uid,
            self.nodes[0].uid,
            self.nodes[1].uid,
            self.geomtransf.uid,
            self.integration.uid,
            # '-iter',  # can change it to forceBeamColumn here for testing
            # 50,
            # 1e-1
        ]

    def __repr__(self) -> str:
        """
        Get string representation.

        Returns:
          The string representation of the object.
        """
        res = ''
        res += 'dispBeamColumn element object\n'
        res += f'uid: {self.uid}\n'
        res += f'node_i.uid: {self.nodes[0].uid}\n'
        res += f'node_j.uid: {self.nodes[1].uid}\n'
        res += f'node_i.coordinates: {self.nodes[0].coordinates}\n'
        res += f'node_j.coordinates: {self.nodes[1].coordinates}\n'
        res += f'offset_i: {self.geomtransf.offset_i}\n'
        res += f'offset_j: {self.geomtransf.offset_j}\n'
        res += f'x_axis: {self.geomtransf.x_axis}\n'
        res += f'y_axis: {self.geomtransf.y_axis}\n'
        res += f'z_axis: {self.geomtransf.z_axis}\n'
        res += f'section.name: {self.section.name}\n'
        return res


@dataclass(repr=False)
class LeadRubberX(Element):
    """
    OpenSees LeadRubberX element.

    https://openseespydoc.readthedocs.io/en/latest/src/LeadRubberX.html
    """

    f_y: float  # yield strength
    alpha: float  # post-yield stiffness ratio
    g_r: float  # shear modulus of bearing
    k_bulk: float  # bulk modulus of rubber
    d_1: float  # internal diameter
    d_2: float  # outer diameter
    t_s: float  # single steel shim layer thickness
    t_r: float  # single rubber layer thickness
    n_layers: int  # number of rubber layers
    x_1: float  # x component of local x-axis
    x_2: float  # y component of local x-axis
    x_3: float  # z component of local x-axis
    y_1: float  # x component of local y-axis
    y_2: float  # y component of local y-axis
    y_3: float  # z component of local y-axis
    k_c: float = field(default=10.0)  # cavitation parameter
    phi_m: float = field(default=0.5)  # damage parameter
    a_c: float = field(default=1.0)  # strength reduction parameter
    s_d_ratio: float = field(default=0.5)  # shear distance
    mass: float = field(default=0.00)  # element mass
    c_d: float = field(default=0.00)  # viscous damping coefficient
    t_c: float = field(default=0.00)  # cover thickness
    q_l: float = field(default=11200.00)  # density of lead (units: kg/m3)
    c_l: float = field(default=130.00)  # specific heat of lead (units: N-m/kg oC)
    k_s: float = field(
        default=50.00
    )  # thermal conductivity of steel (units: 50W/m oC)
    a_s: float = field(
        default=1.41e-5
    )  # thermal diffusivity of steel (units: 1.41e-5m2/s)
    tag_1: int = field(default=0)  # Include cavitation and post-cavitation
    tag_2: int = field(default=0)  # Include buckling load variation
    tag_3: int = field(default=0)  # Include horizontal stiffness variation
    tag_4: int = field(default=0)  # Include vertical stiffness variation
    tag_5: int = field(
        default=0
    )  # Include strength degradation in shear due to heating of lead core

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return [
            'LeadRubberX',
            self.uid,
            self.nodes[0].uid,
            self.nodes[1].uid,
            self.f_y,
            self.alpha,
            self.g_r,
            self.k_bulk,
            self.d_1,
            self.d_2,
            self.t_s,
            self.t_r,
            self.n_layers,
            self.x_1,
            self.x_2,
            self.x_3,
            self.y_1,
            self.y_2,
            self.y_3,
            self.k_c,
            self.phi_m,
            self.a_c,
            self.s_d_ratio,
            self.mass,
            self.c_d,
            self.t_c,
            self.q_l,
            self.c_l,
            self.k_s,
            self.a_s,
            self.tag_1,
            self.tag_2,
            self.tag_3,
            self.tag_4,
            self.tag_5,
        ]


@dataclass(repr=False)
class TripleFrictionPendulum(Element):
    """
    OpenSees TripleFrictionPendulum element.

    https://openseespydoc.readthedocs.io/en/latest/src/TripleFrictionPendulum.html
    """

    friction_model_1: FrictionModel
    friction_model_2: FrictionModel
    friction_model_3: FrictionModel
    vertical_material: UniaxialMaterial
    rot_z_material: UniaxialMaterial
    rot_x_material: UniaxialMaterial
    rot_y_material: UniaxialMaterial
    r_1_eff: float
    r_2_eff: float
    r_3_eff: float
    d_1_eff: float
    d_2_eff: float
    d_3_eff: float
    initial_axial_force: float
    u_y: float
    tension_stiffness: float
    minimum_vertical_compression_force: float
    tolerance: float

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return [
            'TripleFrictionPendulum',
            self.uid,
            self.nodes[0].uid,
            self.nodes[1].uid,
            self.friction_model_1.uid,
            self.friction_model_2.uid,
            self.friction_model_3.uid,
            self.vertical_material.uid,
            self.rot_z_material.uid,
            self.rot_x_material.uid,
            self.rot_y_material.uid,
            self.r_1_eff,
            self.r_2_eff,
            self.r_3_eff,
            self.d_1_eff,
            self.d_2_eff,
            self.d_3_eff,
            self.initial_axial_force,
            self.u_y,
            self.tension_stiffness,
            self.minimum_vertical_compression_force,
            self.tolerance,
        ]


@dataclass(repr=False)
class ElastomericX(Element):
    """
    OpenSees LeadRubberX element.

    https://openseespydoc.readthedocs.io/en/latest/src/LeadRubberX.html
    """

    f_y: float  # yield strength
    alpha: float  # post-yield stiffness ratio
    g_r: float  # shear modulus of bearing
    k_bulk: float  # bulk modulus of rubber
    d_1: float  # internal diameter
    d_2: float  # outer diameter
    t_s: float  # single steel shim layer thickness
    t_r: float  # single rubber layer thickness
    n_layers: int  # number of rubber layers
    x_1: float  # x component of local x-axis
    x_2: float  # y component of local x-axis
    x_3: float  # z component of local x-axis
    y_1: float  # x component of local y-axis
    y_2: float  # y component of local y-axis
    y_3: float  # z component of local y-axis
    k_c: float = field(default=10.0)  # cavitation parameter
    phi_m: float = field(default=0.5)  # damage parameter
    a_c: float = field(default=1.0)  # strength reduction parameter
    s_d_ratio: float = field(default=0.5)  # shear distance
    mass: float = field(default=0.00)  # element mass
    c_d: float = field(default=0.00)  # viscous damping coefficient
    t_c: float = field(default=0.00)  # cover thickness
    q_l: float = field(default=11200.00)  # density of lead (units: kg/m3)
    c_l: float = field(default=130.00)  # specific heat of lead (units: N-m/kg oC)
    k_s: float = field(
        default=50.00
    )  # thermal conductivity of steel (units: 50W/m oC)
    a_s: float = field(
        default=1.41e-5
    )  # thermal diffusivity of steel (units: 1.41e-5m2/s)
    tag_1: int = field(default=0)  # Include cavitation and post-cavitation
    tag_2: int = field(default=0)  # Include buckling load variation
    tag_3: int = field(default=0)  # Include horizontal stiffness variation
    tag_4: int = field(default=0)  # Include vertical stiffness variation
    tag_5: int = field(
        default=0
    )  # Include strength degradation in shear due to heating of lead core

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return [
            'LeadRubberX',
            self.uid,
            self.nodes[0].uid,
            self.nodes[1].uid,
            self.f_y,
            self.alpha,
            self.g_r,
            self.k_bulk,
            self.d_1,
            self.d_2,
            self.t_s,
            self.t_r,
            self.n_layers,
            self.x_1,
            self.x_2,
            self.x_3,
            self.y_1,
            self.y_2,
            self.y_3,
            self.k_c,
            self.phi_m,
            self.a_c,
            self.s_d_ratio,
            self.mass,
            self.c_d,
            self.t_c,
            self.q_l,
            self.c_l,
            self.k_s,
            self.a_s,
            self.tag_1,
            self.tag_2,
            self.tag_3,
            self.tag_4,
            self.tag_5,
        ]
