"""
Defines OpenSees Element interfrace objects.
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

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union
from typing import Optional
from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from .uniaxial_material import UniaxialMaterial
from .node import Node
from .section import ElasticSection
from .section import FiberSection
from ..mesh import Mesh
from ..graphics.visibility import ElementVisibility
if TYPE_CHECKING:
    from ..component_assembly import ComponentAssembly


nparr = npt.NDArray[np.float64]


@dataclass(repr=False)
class Element:
    """
    OpenSees element
    https://openseespydoc.readthedocs.io/en/latest/src/element.html

    Attributes:
        parent_component:
          the parent component assembly that this element belongs to.
        uid: the unique identifier of this element.
        nodes: the list of nodes that this element connects.

    """

    parent_component: ComponentAssembly = field(repr=False)
    uid: int
    nodes: list[Node]
    visibility: ElementVisibility = field(init=False)

    def __post_init__(self):
        self.visibility = ElementVisibility()


@dataclass(repr=False)
class ZeroLength(Element):
    """
    OpenSees ZeroLength element
    https://openseespydoc.readthedocs.io/en/latest/src/ZeroLength.html

    """

    mats: list[UniaxialMaterial]
    dirs: list[int]
    vecx: nparr
    vecyp: nparr

    def ops_args(self):
        """
        Returns the arguments required to define the object in
        OpenSees

        """

        return [
            "zeroLength",
            self.uid,
            *[n.uid for n in self.nodes],
            "-mat",
            *[m.uid for m in self.mats],
            "-dir",
            *self.dirs,
            "-orient",
            *self.vecx,
            *self.vecyp,
        ]

    def __repr__(self):
        res = ""
        res += "ZeroLength element object\n"
        res += f"uid: {self.uid}"
        res += "Materials:"
        for mat, direction in zip(self.mats, self.dirs):
            res += f"  {direction}: {mat.name}\n"
        res += f"vecx: {self.vecx}\n"
        res += f"vecyp: {self.vecyp}\n"
        return res


@dataclass(repr=False)
class TwoNodeLink(Element):
    """
    OpenSees TwoNodeLink element
    https://openseespydoc.readthedocs.io/en/latest/src/twoNodeLink.html

    """

    mats: list[UniaxialMaterial]
    dirs: list[int]
    vecx: nparr
    vecyp: nparr

    def ops_args(self):
        """
        Returns the arguments required to define the object in
        OpenSees

        """

        return [
            "twoNodeLink",
            self.uid,
            *[n.uid for n in self.nodes],
            "-mat",
            *[m.uid for m in self.mats],
            "-dir",
            *self.dirs,
            "-orient",
            *self.vecyp,
        ]

    def __repr__(self):
        res = ""
        res += "TwoNodeLink element object\n"
        res += f"uid: {self.uid}"
        res += "Materials:"
        for mat, direction in zip(self.mats, self.dirs):
            res += f"  {direction}: {mat.name}\n"
        res += f"vecx: {self.vecx}\n"
        res += f"vecyp: {self.vecyp}\n"
        return res


@dataclass(repr=False)
class TrussBar(Element):
    """
    Implements both of the following:
    OpenSees Truss Element
    https://openseespydoc.readthedocs.io/en/latest/src/trussEle.html
    OpenSees Corotational Truss Element
    https://openseespydoc.readthedocs.io/en/latest/src/corotTruss.html

    """

    transf_type: str
    area: float
    mat: UniaxialMaterial
    outside_shape: Optional[Mesh] = field(default=None)
    weight_per_length: float = field(default=0.00)
    rho: float = field(default=0.00)
    cflag: int = field(default=0)
    rflag: int = field(default=0)

    def ops_args(self):
        """
        Returns the arguments required to define the object in
        OpenSees

        """

        elm_name = {
            'Linear': 'Truss',
            'Corotational': 'corotTruss'
        }

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
            self.rflag
        ]

    def clear_length(self):
        """
        Returns the clear length of the element (without the rigid
        offsets)

        """

        p_i = np.array(self.nodes[0].coords)
        p_j = np.array(self.nodes[1].coords)
        return np.linalg.norm(p_i - p_j)

    def __repr__(self):
        elm_name = {
            'Linear': 'Truss',
            'Corotational': 'corotTruss'
        }
        res = ""
        res += f"{elm_name[self.transf_type]} element object\n"
        res += f"uid: {self.uid}\n"
        res += f"node_i.uid: {self.nodes[0].uid}\n"
        res += f"node_j.uid: {self.nodes[1].uid}\n"
        res += f"node_i.coords: {self.nodes[0].coords}\n"
        res += f"node_j.coords: {self.nodes[1].coords}\n"
        return res


@dataclass(repr=False)
class GeomTransf:
    """
    OpenSees geomTransf object
    https://openseespydoc.readthedocs.io/en/latest/src/ZeroLength.html
    https://openseespydoc.readthedocs.io/en/latest/src/geomTransf.html?highlight=geometric%20transformation

    """

    transf_type: str
    uid: int
    offset_i: nparr
    offset_j: nparr
    x_axis: nparr
    y_axis: nparr
    z_axis: nparr

    def ops_args(self):
        """
        Returns the arguments required to define the object in
        OpenSees

        """

        return [
            self.transf_type,
            self.uid,
            *self.z_axis,
            "-jntOffset",
            *self.offset_i,
            *self.offset_j,
        ]


@dataclass(repr=False)
class ElasticBeamColumn(Element):
    """
    OpenSees Elastic Beam Column Element
    https://openseespydoc.readthedocs.io/en/latest/src/elasticBeamColumn.html

    """

    section: ElasticSection
    geomtransf: GeomTransf

    def ops_args(self):
        """
        Returns the arguments required to define the object in
        OpenSees

        """

        return [
            "elasticBeamColumn",
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

    def clear_length(self):
        """
        Returns the clear length of the element (without the rigid
        offsets)

        """

        p_i = np.array(self.nodes[0].coords) + self.geomtransf.offset_i
        p_j = np.array(self.nodes[1].coords) + self.geomtransf.offset_j
        return np.linalg.norm(p_i - p_j)

    def __repr__(self):
        res = ""
        res += "elasticBeamColumn element object\n"
        res += f"uid: {self.uid}\n"
        res += f"node_i.uid: {self.nodes[0].uid}\n"
        res += f"node_j.uid: {self.nodes[1].uid}\n"
        res += f"node_i.coords: {self.nodes[0].coords}\n"
        res += f"node_j.coords: {self.nodes[1].coords}\n"
        res += f"offset_i: {self.geomtransf.offset_i}\n"
        res += f"offset_j: {self.geomtransf.offset_j}\n"
        res += f"x_axis: {self.geomtransf.x_axis}\n"
        res += f"y_axis: {self.geomtransf.y_axis}\n"
        res += f"z_axis: {self.geomtransf.z_axis}\n"
        res += f"section.name: {self.section.name}\n"
        return res


@dataclass
class BeamIntegration:
    """
    OpenSees beamIntegration parent class.
    https://openseespydoc.readthedocs.io/en/latest/src/beamIntegration.html?highlight=beamintegration

    """

    uid: int
    parent_section: Union[ElasticSection, FiberSection] = field(repr=False)


@dataclass
class Lobatto(BeamIntegration):
    """
    OpenSees Lobatto beam integration.
    https://openseespydoc.readthedocs.io/en/latest/src/Lobatto.html

    """

    n_p: int

    def ops_args(self):
        """
        Returns the arguments required to define the object in
        OpenSees

        """

        return ["Lobatto", self.uid, self.parent_section.uid, self.n_p]


@dataclass
class DispBeamColumn(Element):
    """
    OpenSees dispBeamColumn element
    https://openseespydoc.readthedocs.io/en/latest/src/ForceBeamColumn.html

    """

    section: FiberSection
    geomtransf: GeomTransf
    integration: BeamIntegration

    def ops_args(self):
        """
        Returns the arguments required to define the object in
        OpenSees

        """

        return [
            "dispBeamColumn",
            self.uid,
            self.nodes[0].uid,
            self.nodes[1].uid,
            self.geomtransf.uid,
            self.integration.uid,
            # '-iter',  # can change it to forceBeamColumn here for testing
            # 50,
            # 1e-1
        ]

    def clear_length(self):
        """
        Returns the clear length of the element (without the rigid
        offsets)

        """

        p_i = np.array(self.nodes[0].coords) + self.geomtransf.offset_i
        p_j = np.array(self.nodes[1].coords) + self.geomtransf.offset_j
        return np.linalg.norm(p_i - p_j)

    def __repr__(self):
        res = ""
        res += "dispBeamColumn element object\n"
        res += f"uid: {self.uid}\n"
        res += f"node_i.uid: {self.nodes[0].uid}\n"
        res += f"node_j.uid: {self.nodes[1].uid}\n"
        res += f"node_i.coords: {self.nodes[0].coords}\n"
        res += f"node_j.coords: {self.nodes[1].coords}\n"
        res += f"offset_i: {self.geomtransf.offset_i}\n"
        res += f"offset_j: {self.geomtransf.offset_j}\n"
        res += f"x_axis: {self.geomtransf.x_axis}\n"
        res += f"y_axis: {self.geomtransf.y_axis}\n"
        res += f"z_axis: {self.geomtransf.z_axis}\n"
        res += f"section.name: {self.section.name}\n"
        return res
