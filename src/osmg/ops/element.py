"""
Model Generator for OpenSees ~ element
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from ..component_assembly import ComponentAssembly
from .uniaxialMaterial import uniaxialMaterial
from .node import Node
from .section import ElasticSection
from .section import FiberSection
from ..graphics.visibility import ElementVisibility

nparr = npt.NDArray[np.float64]

# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes


@dataclass(repr=False)
class Element:
    """
    OpenSees element
    https://openseespydoc.readthedocs.io/en/latest/src/element.html
    """
    parent_component: ComponentAssembly = field(repr=False)
    uid: int
    eleNodes: list[Node]


@dataclass(repr=False)
class ZeroLength(Element):
    """
    OpenSees ZeroLength element
    https://openseespydoc.readthedocs.io/en/latest/src/ZeroLength.html
    """
    mats: list[uniaxialMaterial]
    dirs: list[int]
    vecx: nparr
    vecyp: nparr

    def ops_args(self):
        return [
            'zeroLength',
            self.uid,
            *[n.uid for n in self.eleNodes],
            '-mat',
            *[m.uid for m in self.mats],
            '-dir',
            *self.dirs,
            '-orient',
            *self.vecx,
            *self.vecyp
        ]

    def __repr__(self):
        res = ''
        res += 'ZeroLength element object\n'
        res += f'uid: {self.uid}'
        res += 'Materials:'
        for m, d in zip(self.mats, self.dirs):
            res += f'  {d}: {m.name}\n'
        res += f'vecx: {self.vecx}\n'
        res += f'vecyp: {self.vecyp}\n'
        return res


@dataclass(repr=False)
class geomTransf:
    transfType: str
    uid: int
    offset_i: nparr
    offset_j: nparr
    x_axis: nparr
    y_axis: nparr
    z_axis: nparr

    def ops_args(self):
        return [
            self.transfType,
            self.uid,
            *self.z_axis,
            '-jntOffset',
            *self.offset_i,
            *self.offset_j
        ]


@dataclass(repr=False)
class elasticBeamColumn(Element):
    """
    OpenSees Elastic Beam Column Element
    https://openseespydoc.readthedocs.io/en/latest/src/elasticBeamColumn.html
    """
    section: ElasticSection
    geomtransf: geomTransf
    visibility: ElementVisibility = field(
        default_factory=ElementVisibility)

    def ops_args(self):
        return [
            'elasticBeamColumn',
            self.uid,
            self.eleNodes[0].uid,
            self.eleNodes[1].uid,
            self.section.A,
            self.section.E,
            self.section.G,
            self.section.J,
            self.section.Iy,
            self.section.Ix,
            self.geomtransf.uid
        ]

    def clear_length(self):
        p_i = np.array(self.eleNodes[0].coords) + self.geomtransf.offset_i
        p_j = np.array(self.eleNodes[1].coords) + self.geomtransf.offset_j
        return np.linalg.norm(p_i - p_j)

    def __repr__(self):
        res = ''
        res += 'elasticBeamColumn element object\n'
        res += f'uid: {self.uid}\n'
        res += f'node_i.uid: {self.eleNodes[0].uid}\n'
        res += f'node_j.uid: {self.eleNodes[1].uid}\n'
        res += f'node_i.coords: {self.eleNodes[0].coords}\n'
        res += f'node_j.coords: {self.eleNodes[1].coords}\n'
        res += f'offset_i: {self.geomtransf.offset_i}\n'
        res += f'offset_j: {self.geomtransf.offset_j}\n'
        res += f'x_axis: {self.geomtransf.x_axis}\n'
        res += f'y_axis: {self.geomtransf.y_axis}\n'
        res += f'z_axis: {self.geomtransf.z_axis}\n'
        res += f'section.name: {self.section.name}\n'
        return res


@dataclass
class beamIntegration:
    uid: int
    parent_section: int = field(repr=False)
    # todo: fix this typing anno


@dataclass
class Lobatto(beamIntegration):
    n_p: int

    def ops_args(self):
        return [
            'Lobatto',
            self.uid,
            self.parent_section.uid,
            self.n_p
        ]


@dataclass
class dispBeamColumn(Element):
    """
    OpenSees dispBeamColumn element
    https://openseespydoc.readthedocs.io/en/latest/src/ForceBeamColumn.html
    """
    section: FiberSection
    geomtransf: geomTransf
    integration: beamIntegration
    visibility: ElementVisibility = field(
        default_factory=ElementVisibility)

    def ops_args(self):
        return [
            'dispBeamColumn',
            self.uid,
            self.eleNodes[0].uid,
            self.eleNodes[1].uid,
            self.geomtransf.uid,
            self.integration.uid,
        ]

    def clear_length(self):
        p_i = np.array(self.eleNodes[0].coords) + self.geomtransf.offset_i
        p_j = np.array(self.eleNodes[1].coords) + self.geomtransf.offset_j
        return np.linalg.norm(p_i - p_j)

    def __repr__(self):
        res = ''
        res += 'dispBeamColumn element object\n'
        res += f'uid: {self.uid}\n'
        res += f'node_i.uid: {self.eleNodes[0].uid}\n'
        res += f'node_j.uid: {self.eleNodes[1].uid}\n'
        res += f'node_i.coords: {self.eleNodes[0].coords}\n'
        res += f'node_j.coords: {self.eleNodes[1].coords}\n'
        res += f'offset_i: {self.geomtransf.offset_i}\n'
        res += f'offset_j: {self.geomtransf.offset_j}\n'
        res += f'x_axis: {self.geomtransf.x_axis}\n'
        res += f'y_axis: {self.geomtransf.y_axis}\n'
        res += f'z_axis: {self.geomtransf.z_axis}\n'
        res += f'section.name: {self.section.name}\n'
        return res
