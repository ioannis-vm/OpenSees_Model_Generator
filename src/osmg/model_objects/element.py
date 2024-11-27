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

    mats: list[UniaxialMaterial]
    dirs: list[int]
    vecx: numpy_array
    vecyp: numpy_array

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return [
            'zeroLength',
            self.uid,
            *[n.uid for n in self.nodes],
            '-mat',
            *[m.uid for m in self.mats],
            '-dir',
            *self.dirs,
            '-doRayleigh',
            1,
            '-orient',
            *self.vecx,
            *self.vecyp,
        ]

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
        for mat, direction in zip(self.mats, self.dirs):
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

    mats: list[UniaxialMaterial]
    dirs: list[int]
    vecx: numpy_array
    vecyp: numpy_array

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return [
            'twoNodeLink',
            self.uid,
            *[n.uid for n in self.nodes],
            '-mat',
            *[m.uid for m in self.mats],
            '-dir',
            *self.dirs,
            '-orient',
            *self.vecyp,
        ]

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
        for mat, direction in zip(self.mats, self.dirs):
            res += f'  {direction}: {mat.name}\n'
        res += f'vecx: {self.vecx}\n'
        res += f'vecyp: {self.vecyp}\n'
        return res


@dataclass(repr=False)
class TrussBar(Element):
    """
    Truss and Corotational Truss.

    Implements both of the following:
    OpenSees Truss Element
    https://openseespydoc.readthedocs.io/en/latest/src/trussEle.html
    OpenSees Corotational Truss Element
    https://openseespydoc.readthedocs.io/en/latest/src/corotTruss.html

    """

    transf_type: str
    area: float
    mat: UniaxialMaterial
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
            self.mat.uid,
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
