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
from typing import Union
import numpy as np
import numpy.typing as npt
from .uniaxial_material import UniaxialMaterial
from .node import Node
from .section import ElasticSection
from .section import FiberSection
from ..graphics.visibility import ElementVisibility
from .. import component_assembly


nparr = npt.NDArray[np.float64]


@dataclass(repr=False)
class Element:
    """
    OpenSees element
    https://openseespydoc.readthedocs.io/en/latest/src/element.html
    """
    parent_component: component_assembly.ComponentAssembly = field(repr=False)
    uid: int
    nodes: list[Node]


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
            'zeroLength',
            self.uid,
            *[n.uid for n in self.nodes],
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
        for mat, direction in zip(self.mats, self.dirs):
            res += f'  {direction}: {mat.name}\n'
        res += f'vecx: {self.vecx}\n'
        res += f'vecyp: {self.vecyp}\n'
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
            '-jntOffset',
            *self.offset_i,
            *self.offset_j
        ]


@dataclass(repr=False)
class ElasticBeamColumn(Element):
    """
    OpenSees Elastic Beam Column Element
    https://openseespydoc.readthedocs.io/en/latest/src/elasticBeamColumn.html
    """
    section: ElasticSection
    geomtransf: GeomTransf
    visibility: ElementVisibility = field(
        default_factory=ElementVisibility)

    def ops_args(self):
        """
        Returns the arguments required to define the object in
        OpenSees
        """
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
            self.geomtransf.uid
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
        res = ''
        res += 'elasticBeamColumn element object\n'
        res += f'uid: {self.uid}\n'
        res += f'node_i.uid: {self.nodes[0].uid}\n'
        res += f'node_j.uid: {self.nodes[1].uid}\n'
        res += f'node_i.coords: {self.nodes[0].coords}\n'
        res += f'node_j.coords: {self.nodes[1].coords}\n'
        res += f'offset_i: {self.geomtransf.offset_i}\n'
        res += f'offset_j: {self.geomtransf.offset_j}\n'
        res += f'x_axis: {self.geomtransf.x_axis}\n'
        res += f'y_axis: {self.geomtransf.y_axis}\n'
        res += f'z_axis: {self.geomtransf.z_axis}\n'
        res += f'section.name: {self.section.name}\n'
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
        return [
            'Lobatto',
            self.uid,
            self.parent_section.uid,
            self.n_p
        ]


@dataclass
class DispBeamColumn(Element):
    """
    OpenSees dispBeamColumn element
    https://openseespydoc.readthedocs.io/en/latest/src/ForceBeamColumn.html
    """
    section: FiberSection
    geomtransf: GeomTransf
    integration: BeamIntegration
    visibility: ElementVisibility = field(
        default_factory=ElementVisibility)

    def ops_args(self):
        """
        Returns the arguments required to define the object in
        OpenSees
        """
        return [
            'dispBeamColumn',
            self.uid,
            self.nodes[0].uid,
            self.nodes[1].uid,
            self.geomtransf.uid,
            self.integration.uid,
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
        res = ''
        res += 'dispBeamColumn element object\n'
        res += f'uid: {self.uid}\n'
        res += f'node_i.uid: {self.nodes[0].uid}\n'
        res += f'node_j.uid: {self.nodes[1].uid}\n'
        res += f'node_i.coords: {self.nodes[0].coords}\n'
        res += f'node_j.coords: {self.nodes[1].coords}\n'
        res += f'offset_i: {self.geomtransf.offset_i}\n'
        res += f'offset_j: {self.geomtransf.offset_j}\n'
        res += f'x_axis: {self.geomtransf.x_axis}\n'
        res += f'y_axis: {self.geomtransf.y_axis}\n'
        res += f'z_axis: {self.geomtransf.z_axis}\n'
        res += f'section.name: {self.section.name}\n'
        return res
