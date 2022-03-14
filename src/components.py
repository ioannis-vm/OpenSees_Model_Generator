"""
Model builder for OpenSees ~ Components
"""

#   __                 UC Berkeley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   April 2021
#
# https://github.com/ioannis-vm/OpenSees_Model_Builder

from dataclasses import dataclass, field
from itertools import count
from collections import OrderedDict
import numpy as np
from grids import GridLine
from node import Node
from material import Material, Materials
from section import Section, Sections
from utility import common
from utility import transformations


elem_ids = count(0)
line_elem_seq_ids = count(0)


def first(ordct: OrderedDict):
    """
    Returns the first value in an OrderedDict
    """
    return list(ordct.values())[0]

def last(ordct: OrderedDict):
    """
    Returns the first value in an OrderedDict
    """
    return list(ordct.values())[-1]

def nth_item(ordct: OrderedDict, idx):
    """
    Returns the idx-th value in an OrderedDict
    """
    return list(ordct.values())[idx]


def generate_pz_mat(section: Section,
                    pz_length: float,
                    pz_doubler_plate_thickness: float) -> Material:
    fy = section.material.parameters['Fy']
    hardening = section.material.parameters['b_PZ']
    dc = section.properties['d']
    bfc = section.properties['bf']
    tp = section.properties['tw'] + pz_doubler_plate_thickness
    tf = section.properties['tf']
    vy = 0.55 * fy * dc * tp
    g_mod = section.material.parameters['G']
    ke = 0.95 * g_mod * tp * dc
    kp = 0.95 * g_mod * bfc * tf**2 / pz_length
    gamma_1 = vy / ke
    gamma_2 = 4.0 * gamma_1
    gamma_3 = 100. * gamma_1
    m1y = gamma_1 * ke * pz_length
    m2y = m1y + kp * pz_length * (gamma_2 - gamma_1)
    m3y = m2y + (hardening * ke
                 * pz_length) * (gamma_3 - gamma_2)
    mat_pz = Material(
        'auto__panel_zone_spring',
        'Hysteretic',
        0.00,
        {
            'M1y': m1y,
            'gamma1_y': gamma_1,
            'M2y': m2y,
            'gamma2_y': gamma_2,
            'M3y': m3y,
            'gamma3_y': gamma_3,
            'pinchX': 1.00,
            'pinchY': 1.00,
            'damage1': 0.00,
            'damage2': 0.00,
            'beta': 0.00
        })
    return mat_pz


def generate_IMK_mat(section: Section, ends: dict, elm_length) -> Material:
    # define IMK material based on element properties
    assert section.sec_type == 'W', \
        "Error: Only W sections can be used"
    # Young's modulus
    mat_e = section.material.parameters['E0'] / 1.e3
    # Yield stress
    mat_fy = section.material.parameters['Fy'] / 1.e3
    # Moment of inertia - strong axis - original section
    sec_ix = section.properties['Ix']
    # Section depth
    sec_d = section.properties['d']
    # Flange width
    sec_bf = section.properties['bf']
    # Flange and web thicknesses
    sec_tf = section.properties['tf']
    sec_tw = section.properties['tw']
    # Plastic modulus (unreduced)
    sec_zx = section.properties['Zx']
    lbry = ends['Lb/ry']
    # Clear length
    elm_H = elm_length
    # Shear span - 0.5 * elm_H typically.
    elm_L = ends['L/H'] * elm_H
    # RBS reduction factor
    rbs_factor = ends.get('RBS_factor')
    # Floor composite action consideration
    consider_composite = ends.get('composite action')
    if rbs_factor:
        # RBS case
        assert rbs_factor <= 1.00, 'rbs_factor must be <= 1.00'
        # checks ~ acceptable range
        if not (20.00 < sec_d/sec_tw < 55.00):
            print('Warning: sec_d/sec_tw outside regression range')
            print(section, '\n')
        if not (20.00 < lbry < 80.00):
            print('Warning: Lb/ry outside regression range')
            print(section, '\n')
        if not (4.00 < (sec_bf/(2.*sec_tf)) < 8.00):
            print('Warning: bf/(2 tf) outside regression range')
            print(section, '\n')
        if not (2.5 < elm_L/sec_d < 7.0):
            print('Warning: L/d  outside regression range')
            print(section, '\n')
        if not (4.00 < sec_d < 36.00):
            print('Warning: Section d outside regression range')
            print(section, '\n')
        if not (35.00 < mat_fy < 65.00):
            print('Warning: Fy outside regression range')
            print(section, '\n')
        # calculate parameters
        theta_p = 0.19 * (sec_d/sec_tw)**(-0.314) * \
            (sec_bf/(2.*sec_tf))**(-0.10) * \
            lbry**(-0.185) * \
            (elm_L/sec_d)**0.113 * \
            (25.4 * sec_d / 533.)**(-0.76) * \
            (6.895 * mat_fy / 355.)**(-0.07)
        theta_pc = 9.52 * (sec_d/sec_tw)**(-0.513) * \
            (sec_bf/(2.*sec_tf))**(-0.863) * \
            lbry**(-0.108) * \
            (6.895 * mat_fy / 355.)**(-0.36)
        lamda = 585. * (sec_d/sec_tw)**(-1.14) * \
            (sec_bf/(2.*sec_tf))**(-0.632) * \
            lbry**(-0.205) * \
            (6.895 * mat_fy / 355.)**(-0.391)
        rbs_c = sec_bf * (1. - rbs_factor) / 2.
        z_rbs = sec_zx - 2. * rbs_c * sec_tf * (sec_d - sec_tf)
        sec_my = 1.06 * z_rbs * mat_fy * 1.e3
    else:
        # Other-than-RBS case
        theta_p = 0.0865 * (sec_d/sec_tw)**(-0.365) * \
            (sec_bf/(2.*sec_tf))**(-0.14) * \
            (elm_L/sec_d)**0.34 * \
            (25.4 * sec_d / 533.)**(-0.721) * \
            (6.895 * mat_fy / 355.)**(-0.23)
        theta_pc = 5.63 * (sec_d/sec_tw)**(-0.565) * \
            (sec_bf/(2.*sec_tf))**(-0.800) * \
            (25.4 * sec_d / 533.)**(-0.28) *  \
            (6.895 * mat_fy / 355.)**(-0.43)
        lamda = 495. * (sec_d/sec_tw)**(-1.34) * \
            (sec_bf/(2.*sec_tf))**(-0.595) * \
            (6.895 * mat_fy / 355.)**(-0.36)
        sec_my = 1.17 * sec_zx * mat_fy * 1.e3
    theta_u = 0.20
    residual_plus = 0.40
    residual_minus = 0.40
    theta_p_plus = theta_p
    theta_p_minus = theta_p
    theta_pc_plus = theta_pc
    theta_pc_minus = theta_pc
    d_plus = 1.00
    d_minus = 1.00
    mcmy_plus = 1.0001
    mcmy_minus = 1.0001
    my_plus = sec_my
    my_minus = -sec_my
    if consider_composite:
        # Elkady, A., & Lignos, D. G. (2014). Modeling of the
        # composite action in fully restrained beam‐to‐column
        # connections: implications in the seismic design and
        # collapse capacity of steel special moment
        # frames. Earthquake Engineering & Structural Dynamics,
        # 43(13), 1935-1954.  Table II
        theta_p_plus *= 1.80
        theta_p_minus *= 0.95
        theta_pc_plus *= 1.35
        theta_pc_minus *= 0.95
        d_plus *= 1.15
        d_minus *= 1.00
        mcmy_plus *= 1.30
        mcmy_minus *= 1.05
        my_plus *= 1.35
        my_minus *= 1.25
        residual_plus = 0.30
        residual_minus = 0.20
    stiffness = 6.00 * mat_e * sec_ix / elm_H * 1e4
    beta_plus = (mcmy_plus - 1.) * my_plus / (theta_p_plus) / stiffness
    beta_minus = - (mcmy_minus - 1.) * my_minus \
        / (theta_p_minus) / stiffness
    mat_imk = Material(
        'auto_IMK',
        'Bilin',
        0.00,
        {
            'initial_stiffness': stiffness,
            'b+': beta_plus,
            'b-': beta_minus,
            'my+': my_plus,
            'my-': my_minus,
            'lamda': lamda,
            'theta_p+': theta_p_plus,
            'theta_p-': theta_p_minus,
            'theta_pc+': theta_pc_plus,
            'theta_pc-': theta_pc_minus,
            'residual_plus': residual_plus,
            'residual_minus': residual_minus,
            'theta_u': theta_u,
            'd+': d_plus,
            'd-': d_minus,
        })

    return mat_imk


@dataclass
class EndRelease:
    """
    This class is used to simulate end-releases.
    Currently used for
        - pinned connections (see EndSegment_Pinnned)
        - W sec panel zones (see EndSegment_Steel_W_PanelZone)
        - W sec modified IMK deterioration modeling
        - W sec gravity shear tab connections
    It works by connecting two nodes to each other using
    a zeroLength element and passing a uniaxialMaterial for
    each of the DOFs that resist differential movement.
    DOFs that are left without a uniaxialMaterial turn into
    releases. These two nodes must be defined at the same location,
    but they can't be the same node. We work in 3D, so we also
    need two vectors for spatial orientation.
    """

    node_i: Node = field(repr=False)
    node_j: Node = field(repr=False)
    materials: dict[int, Material] = field(repr=False)
    x_vec: np.ndarray = field(repr=False)
    y_vec: np.ndarray = field(repr=False)

    def __post_init__(self):
        self.uid = next(elem_ids)


@dataclass
class LineElement:
    """
    Linear finite element class.
    This class represents the most primitive linear element,
    on which more complex classes build upon.
    Attributes:
        uid (int): unique identifier
        node_i (Node): Node if end i
        node_j (Node): Node of end j
        ang: Parameter that controls the rotation of the
             section around the x-axis
        offset_i (np.ndarray): Components of the vector that starts
                               from the primary node i and goes to
                               the first internal node at the end i.
                               Expressed in the global coordinate system.
        offset_j (np.ndarray): Similarly for node j
        section (Section): Section of the element interior
        len_parent (float): Lenth of the parent element (LineElementSequence)
        len_proportion (float): Proportion of the line element's length
                        to the length of its parent line element sequence.
        model_as (dict): Either
                       {'type': 'elastic'}
                       or
                       {'type': 'fiber', 'n_x': n_x, 'n_y': n_y, 'n_p': n_p}
                       n_x, n_y: # of segments in Y and Z axis for sec
                       subdivision
                       n_p: # of integr points along element length
        geom_transf: {Linear, PDelta, Corotational}
        internal_pt_i (np.ndarray): Coordinates of the internal point i
        internal_pt_j (np.ndarray): Similarly for node j
                                    Internal points are the ones opposite to
                                    the primary nodes, when we subtract the
                                    rigid offsets.
        udl_self (np.ndarray): Array of size 3 containing components of the
                          uniformly distributed load that is applied
                          to the clear length of the element, acting
                          on the local x, y, and z directions, in the
                          direction of the axes (see Section).
                          Values are in units of force per unit length.
        udl_fl, udl_other (np.ndarray): Similar to udl, coming from the floors
                          and anything else
        x_axis: Array of size 3 representing the local x axis vector
                expressed in the global coordinate system.
        y_axis: (similar)
        z_axis: (similar).
                The axes are defined in the same way as they are
                defined in OpenSees.

                        y(green)
                        ^         x(red)
                        :       .
                        :     .
                        :   .
                       ===
                        | -------> z (blue)
                       ===
        tributary_area (float): Area of floor that is supported on the element.
        hidden_when_extruded (bool): controls plotting behavior
        hinned_as_line (bool): --//--
    """

    node_i: Node = field(repr=False)
    node_j: Node = field(repr=False)
    ang: float = field(repr=False)
    offset_i: np.ndarray = field(repr=False)
    offset_j: np.ndarray = field(repr=False)
    section: Section = field(repr=False)
    parent: 'LineElementSequence' = field(repr=False)
    len_parent: float = field(repr=False)
    model_as: dict = field(repr=False)
    geom_transf: str = field(repr=False)
    udl_self: np.ndarray = field(
        default_factory=lambda: np.zeros(shape=3), repr=False)
    udl_fl: np.ndarray = field(
        default_factory=lambda: np.zeros(shape=3), repr=False)
    udl_other: np.ndarray = field(
        default_factory=lambda: np.zeros(shape=3), repr=False)
    hidden_when_extruded: bool = field(default=False)
    hidden_as_line: bool = field(default=False)

    def __post_init__(self):

        assert(isinstance(self.node_i, Node))
        assert(isinstance(self.node_j, Node))

        # ---  this is needed for tributary area analysis  ---
        # ( ... since adding support for steel W panel zones              )
        # ( the convention that every closed shape in plan view could     )
        # ( be extracted by following the connectivity of line elements   )
        # ( was broken, because line elements connected to panel zones    )
        # ( create a gap between the front and the back nodes of the      )
        # ( panel zone. To overcome this, we retain their prior           )
        # ( connevtivity information (before preprocessing) so that       )
        # ( we can still use it to do the tributary area analysis without )
        # ( having to fundamentaly change that part of the code.          )
        self.node_i_trib = self.node_i
        self.node_j_trib = self.node_j
        # ~~~~

        self.uid = next(elem_ids)
        self.tributary_area = 0.00
        # local axes with respect to the global coord system

        self.internal_pt_i = self.node_i.coords + self.offset_i
        self.internal_pt_j = self.node_j.coords + self.offset_j

        self.length_clear = np.linalg.norm(
            self.internal_pt_i - self.internal_pt_j)
        self.x_axis, self.y_axis, self.z_axis = \
            transformations.local_axes_from_points_and_angle(
                self.internal_pt_i, self.internal_pt_j, self.ang)
        self.len_proportion = self.length_clear / self.len_parent
        # note: for dispBeamColumn, 2 integration points
        # suffice. When using forceBeamColumn, they should be increased.
        if self.len_proportion > 0.75:
            n_p = 2
        elif self.len_proportion > 0.50:
            n_p = 2
        else:
            n_p = 2
        self.n_p = n_p

    # def update_axes(self):
    #     """
    #     Recalculate the local axes of the elements, in case
    #     the nodes were changed after the element's definition.
    #     """
    #     self.internal_pt_i = self.node_i.coords + self.offset_i
    #     self.internal_pt_j = self.node_j.coords + self.offset_j
    #     self.x_axis, self.y_axis, self.z_axis = \
    #         transformations.local_axes_from_points_and_angle(
    #             self.internal_pt_i, self.internal_pt_j, self.ang)

    def add_udl_glob(self, udl: np.ndarray, ltype='other'):
        """
        Adds a uniformly distributed load
        to the existing udl of the element.
        The load is defined
        with respect to the global coordinate system
        of the building, and it is converted to the
        local coordinate system prior to adding it.
        Args:
            udl (np.ndarray): Array of size 3 containing components of the
                              uniformly distributed load that is applied
                              to the clear length of the element, acting
                              on the global x, y, and z directions, in the
                              direction of the global axes.
        """
        T_mat = transformations.transformation_matrix(
            self.x_axis, self.y_axis, self.z_axis)
        udl_local = T_mat @ udl
        if ltype == 'self':
            self.udl_self += udl_local
        elif ltype == 'floor':
            self.udl_fl += udl_local
        elif ltype == 'other':
            self.udl_other += udl_local
        else:
            raise ValueError("Unsupported load type")

    def udl_total(self):
        """
        Returns the total udl applied to the element,
        by summing up the floor's contribution to the
        generic component.
        """
        return self.udl_self + self.udl_fl + self.udl_other

    def get_udl_no_floor_glob(self):
        """
        Returns the current value of the total UDL to global
        coordinates
        """
        udl = self.udl_self + self.udl_other
        T_mat = transformations.transformation_matrix(
            self.x_axis, self.y_axis, self.z_axis)
        return T_mat.T @ udl

    def split(self, proportion: float) \
            -> tuple['LineElement', 'LineElement', Node]:
        """
        Splits the LineElement into two LineElements
        and returns the resulting two LineElements
        and a node that connects them.
        Warning! This operation should not be performed after
        processing the building. It does not account for
        loads, masses etc.
        Args:
            proportion (float): Proportion of the distance from
                the clear end i (without the offset) to the
                LineElement's clear length (without the offsets)
        """
        split_location = self.node_i.coords + \
            self.offset_i + self.x_axis * \
            (self.length_clear * proportion)
        split_node = Node(split_location)
        piece_i = LineElement(
            self.node_i, split_node,
            self.ang, self.offset_i, np.zeros(3).copy(),
            self.section, self.parent, self.len_parent, self.model_as,
            self.geom_transf, self.udl_self.copy(), self.udl_fl.copy(),
            self.udl_other.copy())
        piece_j = LineElement(
            split_node, self.node_j,
            self.ang, np.zeros(3).copy(), self.offset_j,
            self.section, self.parent, self.len_parent, self.model_as,
            self.geom_transf, self.udl_self.copy(), self.udl_fl.copy(),
            self.udl_other.copy())
        return piece_i, piece_j, split_node


@dataclass
class EndSegment:
    """
    This class represents an end segment of a LineElementSequence.
    Please read the docstring of that class.
    See figure:
        https://notability.com/n/0Nvu2Gqlt3gfwEcQE2~RKf
    """
    parent: 'LineElementSequence'  # parent line element sequence
    n_external: Node
    n_internal: Node
    end: str

    def __post_init__(self):

        self.internal_nodes = OrderedDict()
        self.internal_line_elems = OrderedDict()
        self.internal_end_releases = OrderedDict()

        self.add(self.n_internal)
        
        if self.end == 'i':
            offset = self.parent.offset_i
        else:
            offset = self.parent.offset_j
        self.internal_pt = self.n_external.coords + offset

    def offset(self):
        """
        Retrieves the offset from the parent LineElementSequence
        depending on whether the endsegment is at the i or the j
        end.
        """
        if self.end == 'i':
            offset = self.parent.offset_i
        else:
            offset = self.parent.offset_j
        return offset

    def add(self, thing):
        """
        Adds internal elements to the EndSegment.
        """
        if isinstance(thing, Node):
            self.internal_nodes[thing.uid] = thing
        elif isinstance(thing, LineElement):
            self.internal_line_elems[thing.uid] = thing
        elif isinstance(thing, EndRelease):
            self.internal_end_releases[thing.uid] = thing
        else:
            raise ValueError(f'Unknown type: {type(thing)}')

    def first_line_elem(self):
        return list(self.internal_line_elems.values())[0]

    def last_line_elem(self):
        return list(self.internal_line_elems.values())[-1]


@dataclass
class EndSegment_Fixed(EndSegment):
    """
    This class represents a pinned end segment of a LineElementSequence.
    Please read the docstring of that class.
    See figure:
        https://notability.com/n/0Nvu2Gqlt3gfwEcQE2~RKf
    Attributes:
        See the attributes of EndSegment
    """
    def __post_init__(self):
        super().__post_init__()
        if self.n_external != self.n_internal:
            if self.end == 'i':
                self.add(LineElement(
                    self.n_external, self.n_internal, self.parent.ang,
                    self.parent.offset_i, np.zeros(3).copy(),
                    self.parent.section,
                    self.parent, self.parent.length_clear,
                    self.parent.model_as, self.parent.geom_transf))
            elif self.end == 'j':
                self.add(LineElement(
                    self.n_internal, self.n_external, self.parent.ang,
                    np.zeros(3).copy(), self.parent.offset_j,
                    self.parent.section,
                    self.parent, self.parent.length_clear,
                    self.parent.model_as, self.parent.geom_transf))


@dataclass
class EndSegment_Pinned(EndSegment):
    """
    This class represents a pinned end segment of a LineElementSequence.
    Please read the docstring of that class.
    See figure:
        https://notability.com/n/0Nvu2Gqlt3gfwEcQE2~RKf
    Attributes:
        See the attributes of EndSegment
    Additional attributes:
        mat_fix (Material): Linear elastic material with a very high stiffness
                            See the Materials class.
    """

    mat_fix: Material = field(repr=False)

    def __post_init__(self):
        super().__post_init__()
        n_release = Node(self.n_internal.coords)
        self.internal_nodes[n_release.uid] = n_release
        if self.end == 'i':
            self.add(LineElement(
                self.n_external, n_release, self.parent.ang,
                self.offset(), np.zeros(3).copy(),
                self.parent.section,
                self.parent, self.parent.length_clear,
                self.parent.model_as, self.parent.geom_transf))
            self.add(EndRelease(
                    n_release,
                    self.n_internal,
                    {1: self.mat_fix,
                     2: self.mat_fix,
                     3: self.mat_fix,
                     4: self.mat_fix},
                    self.parent.x_axis, self.parent.y_axis))

        elif self.end == 'j':
            self.add(EndRelease(
                    self.n_internal,
                    n_release,
                    {1: self.mat_fix,
                     2: self.mat_fix,
                     3: self.mat_fix,
                     4: self.mat_fix},
                    self.parent.x_axis, self.parent.y_axis))
            self.add(LineElement(
                n_release, self.n_external, self.parent.ang,
                np.zeros(3).copy(), self.offset(),
                self.parent.section,
                self.parent, self.parent.length_clear,
                self.parent.model_as, self.parent.geom_transf))


@dataclass
class EndSegment_RBS(EndSegment):
    """
    This class represents an RBS end segment of a LineElementSequence.
    Please read the docstring of that class.
    See figure:
        https://notability.com/n/0Nvu2Gqlt3gfwEcQE2~RKf
    Attributes:
        See the attributes of EndSegment
    Additional attributes:
        end: Whether the EndSegment corresponds to the
             start ("i") or the end ("j") of the LineElementSequence
        len_parent: Clear length of the parent LineElementSequence
        ang, section, model_as, geom_transf:
            Arguments used for element creation. See LineElement.
        x_axis (np.ndarray): X axis vector of the parent LineElementSequence
                             (expressed in global coordinates)
        rbs_length (float): Length of the reduced beam segment
                            (expressed in length units, not proportional)
        rbs_reduction (float): Proportion of the reduced beam section's width
                               relative to the initial section.
        n_sub (int): Number of LineElement objects representing the RBS segment
    """

    def __post_init__(self):
        super().__post_init__()

        assert self.parent.ends['rbs_length'] > 0.00, "RBS length must be > 0"

        # Generate the reduced section
        reduced_sections = []
        b_f = self.parent.section.properties['bf']
        rbs_c = (1.00 - self.parent.ends['rbs_reduction']) * b_f / 2.
        rbs_length = self.parent.ends['rbs_length']

        # i versus j matters because of the orientation of things

        if self.end == 'i':
            for x in np.linspace(
                    0.00, 1.00,
                    num=self.parent.ends['rbs_n_sub'] + 2)[1::][::-1]:
                # define and add internal nodes
                self.add(
                    Node(self.n_internal.coords -
                         self.parent.x_axis *
                         self.parent.ends['rbs_length'] * x)
                )
                # define reduced sections
                x_sec = x - 1/(self.parent.ends['rbs_n_sub']+1)
                rbs_c_cur = 1./8.*(8.*rbs_c - np.sqrt((
                    rbs_length**2+4.*rbs_c**2)**2/(rbs_c)**2) +
                    np.sqrt(rbs_length**4/rbs_c**2 +
                            16.*rbs_c**2-8.*rbs_length**2 *
                            (1.-8.*x_sec+8.*x_sec**2)))
                reduction_cur = (b_f - 2. * rbs_c_cur) / b_f
                reduced_sections.append(
                    self.parent.section.rbs(reduction_cur))
            self.add(self.n_internal)
            self.add(
                LineElement(
                    self.n_external, first(self.internal_nodes),
                    self.parent.ang,
                    self.offset, np.zeros(3).copy(),
                    self.parent.section,
                    self.parent, self.parent.length_clear,
                    self.parent.model_as, self.parent.geom_transf))
            for i in range(self.parent.ends['rbs_n_sub']+1):
                self.add(
                    LineElement(
                        nth_item(self.internal_nodes, i),
                        nth_item(self.internal_nodes, i+1),
                        self.parent.ang,
                        np.zeros(3).copy(), np.zeros(3).copy(),
                        reduced_sections[i],
                        self.parent, self.parent.length_clear,
                        self.parent.model_as, self.parent.geom_transf))

        elif self.end == 'j':
            self.add(self.n_internal)
            for x in np.linspace(
                    0.00, 1.00, num=self.parent.ends['rbs_n_sub'] + 2)[1::]:
                # define and add internal nodes
                self.add(
                    Node(self.n_internal.coords +
                         self.parent.x_axis *
                         self.parent.ends['rbs_length'] * x)
                )
                # define reduced sections
                x_sec = x - 1/(self.parent.ends['rbs_n_sub']+1)
                rbs_c_cur = 1./8.*(8.*rbs_c - np.sqrt((
                    rbs_length**2+4.*rbs_c**2)**2/(rbs_c)**2) +
                    np.sqrt(rbs_length**4/rbs_c**2 +
                            16.*rbs_c**2-8.*rbs_length**2 *
                            (1.-8.*x_sec+8.*x_sec**2)))
                reduction_cur = (b_f - 2. * rbs_c_cur) / b_f
                reduced_sections.append(
                    self.parent.section.rbs(reduction_cur))
            for i in range(self.parent.ends['rbs_n_sub']+1):
                self.add(
                    LineElement(
                        nth_item(self.internal_nodes, i),
                        nth_item(self.internal_nodes, i+1),
                        self.parent.ang,
                        np.zeros(3).copy(), np.zeros(3).copy(),
                        reduced_sections[i],
                        self.parent, self.parent.length_clear,
                        self.parent.model_as, self.parent.geom_transf))
            self.add(
                LineElement(
                    last(self.internal_nodes), self.n_external,
                    self.parent.ang,
                    np.zeros(3).copy(), self.offset,
                    self.parent.section,
                    self.parent, self.parent.length_clear,
                    self.parent.model_as, self.parent.geom_transf))


@dataclass
class EndSegment_IMK(EndSegment):
    """
    TODO ~ update docstring to provide context.
    IMK ~ Ibarra Medina Krawinkler concentrated plasticity deterioration model
    This class represents an IMK end segment of a LineElementSequence.
    Please read the docstring of that class.
    See figure:
        https://notability.com/n/0Nvu2Gqlt3gfwEcQE2~RKf
    Attributes:
        See the attributes of EndSegment
    Additional attributes:
        end: Whether the EndSegment corresponds to the
             start ("i") or the end ("j") of the LineElementSequence
        len_parent: Clear length of the parent LineElementSequence
        ang, section, model_as, geom_transf:
            Arguments used for element creation. See LineElement.
        x_axis (np.ndarray): X axis vector of the parent LineElementSequence
                             (expressed in global coordinates)
        rbs_length (float): Length of the reduced beam segment
                            (expressed in length units, not proportional)
        rbs_reduction (float): Proportion of the reduced beam section's width
                               relative to the initial section.
        n_sub (int): Number of LineElement objects representing the RBS segment
    """
    mat_fix: Material = field(repr=False)

    def __post_init__(self):

        super().__post_init__()

        self.mat_imk = generate_IMK_mat(
            self.parent.section, self.parent.ends, self.parent.length_clear)

        n_release = Node(self.n_internal.coords)
        self.add(n_release)
        if self.end == 'i':
            self.add(
                LineElement(
                    self.n_external, n_release,
                    self.parent.ang,
                    self.offset(), np.zeros(3).copy(),
                    self.parent.section,
                    self.parent, self.parent.length_clear,
                    self.parent.model_as, self.parent.geom_transf))
            self.add(
                EndRelease(
                    n_release,
                    self.n_internal,
                    {1: self.mat_fix,
                     2: self.mat_fix,
                     3: self.mat_fix,
                     4: self.mat_fix,
                     6: self.mat_imk},
                    self.parent.x_axis, self.parent.y_axis))
        elif self.end == 'j':
            self.add(
                EndRelease(
                    self.n_internal,
                    n_release,
                    {1: self.mat_fix,
                     2: self.mat_fix,
                     3: self.mat_fix,
                     4: self.mat_fix,
                     6: self.mat_imk},
                    self.parent.x_axis, self.parent.y_axis))
            self.add(
                LineElement(
                    n_release, self.n_external, self.parent.ang,
                    np.zeros(3).copy(), self.offset(),
                    self.parent.section,
                    self.parent, self.parent.length_clear,
                    self.parent.model_as, self.parent.geom_transf))


@dataclass
class EndSegment_Steel_W_PanelZone(EndSegment):
    """
    This class represents a panel zone end segment of a LineElementSequence.
    Please read the docstring of that class.
    See figure:
        TODO
    Notes:
        This EndSegment is not used during element creation. It is used
        when calling `preprocess` on the building and setting
        steel_panel_zones=True. W columns where panel zones can be defined,
        based on their connectivity with W beams, will be replaced
        such that they include this EndSegment.
        Doing it in that step enables more automation, as the properties
        of the connection (such as column and beam depths etc.) can be derived
        from the defined elements instead of being required at the column
        definition phase.
    """
    rigid_util_section: Section = field(repr=False)
    mat_fix: Material = field(repr=False)

    def __post_init__(self):

        super().__post_init__()

        self.mat_pz = generate_pz_mat(
            self.parent.section,
            self.parent.ends['end_dist'],
            self.parent.ends['doubler plate thickness'])

        column_depth = self.parent.section.properties['d']

        # define nodes
        n_top = self.n_external
        n_bottom = self.n_internal
        n_bottom.coords = n_top.coords + self.parent.x_axis \
            * self.parent.ends['end_dist']

        # these three attributes are used for plotting
        self.internal_pt_i = n_top.coords
        self.internal_pt_j = n_bottom.coords
        self.length_clear = np.linalg.norm(
            self.n_external.coords - self.n_internal.coords)

        n_1 = Node(n_top.coords + self.parent.y_axis * column_depth / 2.)
        n_2 = Node(n_top.coords + self.parent.y_axis * column_depth / 2.)
        self.n_front = Node(
            n_top.coords + self.parent.y_axis * column_depth / 2.
            + self.parent.x_axis * self.parent.ends['end_dist'] / 2.)
        n_3 = Node(n_top.coords + self.parent.y_axis * column_depth / 2.
                   + self.parent.x_axis * self.parent.ends['end_dist'])
        n_4 = Node(n_top.coords + self.parent.y_axis * column_depth / 2.
                   + self.parent.x_axis * self.parent.ends['end_dist'])
        n_5 = Node(n_top.coords - self.parent.y_axis * column_depth / 2.
                   + self.parent.x_axis * self.parent.ends['end_dist'])
        n_6 = Node(n_top.coords - self.parent.y_axis * column_depth / 2.
                   + self.parent.x_axis * self.parent.ends['end_dist'])
        self.n_back = Node(
            n_top.coords - self.parent.y_axis * column_depth / 2.
            + self.parent.x_axis * self.parent.ends['end_dist'] / 2.)
        n_7 = Node(n_top.coords - self.parent.y_axis * column_depth / 2.)
        n_8 = Node(n_top.coords - self.parent.y_axis * column_depth / 2.)

        # define rigid line elements connecting the nodes
        ang = self.parent.ang
        elm_a = LineElement(n_top, n_1, 0.00,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            self.parent,
                            1.00,
                            model_as={'type': 'elastic'},
                            geom_transf='Linear',
                            hidden_when_extruded=True)
        elm_b = LineElement(n_2, self.n_front, ang,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            self.parent,
                            1.00,
                            model_as={'type': 'elastic'},
                            geom_transf='Linear',
                            hidden_when_extruded=True)
        elm_c = LineElement(self.n_front, n_3, ang,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            self.parent,
                            1.00,
                            model_as={'type': 'elastic'},
                            geom_transf='Linear',
                            hidden_when_extruded=True)
        elm_d = LineElement(n_bottom, n_4, 0.00,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            self.parent,
                            1.00,
                            model_as={'type': 'elastic'},
                            geom_transf='Linear',
                            hidden_when_extruded=True)
        elm_e = LineElement(n_5, n_bottom, 0.00,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            self.parent,
                            1.00,
                            model_as={'type': 'elastic'},
                            geom_transf='Linear',
                            hidden_when_extruded=True)
        elm_f = LineElement(self.n_back, n_6, ang,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            self.parent,
                            1.00,
                            model_as={'type': 'elastic'},
                            geom_transf='Linear',
                            hidden_when_extruded=True)
        elm_g = LineElement(n_7, self.n_back, ang,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            self.parent,
                            1.00,
                            model_as={'type': 'elastic'},
                            geom_transf='Linear',
                            hidden_when_extruded=True)
        elm_h = LineElement(n_8, n_top, 0.00,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            self.parent,
                            1.00,
                            model_as={'type': 'elastic'},
                            geom_transf='Linear',
                            hidden_when_extruded=True)

        # define releases
        rel_spring = EndRelease(
            n_1, n_2,
            {1: self.mat_fix,
             2: self.mat_fix,
             3: self.mat_fix,
             4: self.mat_fix,
             5: self.mat_fix,
             6: self.mat_pz},
            self.parent.x_axis, self.parent.y_axis)
        rel_bottom_front = EndRelease(
            n_3, n_4,
            {1: self.mat_fix,
             2: self.mat_fix,
             3: self.mat_fix,
             4: self.mat_fix,
             5: self.mat_fix},
            self.parent.x_axis, self.parent.y_axis)
        rel_bottom_back = EndRelease(
            n_5, n_6,
            {1: self.mat_fix,
             2: self.mat_fix,
             3: self.mat_fix,
             4: self.mat_fix,
             5: self.mat_fix},
            self.parent.x_axis, self.parent.y_axis)
        rel_top_back = EndRelease(
            n_7, n_8,
            {1: self.mat_fix,
             2: self.mat_fix,
             3: self.mat_fix,
             4: self.mat_fix,
             5: self.mat_fix},
            self.parent.x_axis, self.parent.y_axis)

        # store internal objects in corresponding lists
        self.internal_nodes = {
            'bottom': n_bottom,
            '1': n_1,
            '2': n_2,
            'front': self.n_front,
            '3': n_3,
            '4': n_4,
            '5': n_5,
            '6': n_6,
            'back': self.n_back,
            '7': n_7,
            '8': n_8
        }
        self.internal_line_elems = {
            'A': elm_a,
            'B': elm_b,
            'C': elm_c,
            'D': elm_d,
            'E': elm_e,
            'F': elm_f,
            'G': elm_g,
            'H': elm_h
        }
        self.internal_end_releases = {
            'PZ': rel_spring,
            'release_bottom_front': rel_bottom_front,
            'release_bottom_back': rel_bottom_back,
            'release_top_back': rel_top_back
        }


@dataclass
class EndSegment_Steel_W_PanelZone_IMK(EndSegment_Steel_W_PanelZone):
    """
    This class represents a panel zone end segment of a LineElementSequence.
    Please read the docstring of that class.
    See figure:
        TODO
    Notes:
        This EndSegment is not used during element creation. It is used
        when calling `preprocess` on the building and setting
        steel_panel_zones=True. W columns where panel zones can be defined,
        based on their connectivity with W beams, will be replaced
        such that they include this EndSegment.
        Doing it in that step enables more automation, as the properties
        of the connection (such as column and beam depths etc.) can be derived
        from the defined elements instead of being required at the column
        definition phase.
    """

    def __post_init__(self):

        super().__post_init__()

        n_bottom = self.internal_nodes['bottom']
        n_4 = self.internal_nodes['4']
        n_5 = self.internal_nodes['5']
        n_bottom_spring = Node(n_bottom.coords)

        self.internal_line_elems['D'] = LineElement(
            n_bottom_spring, n_4, 0.00,
            np.zeros(3).copy(),
            np.zeros(3).copy(),
            self.rigid_util_section,
            self.parent,
            1.00,
            model_as={'type': 'elastic'},
            geom_transf='Linear',
            hidden_when_extruded=True)

        self.internal_line_elems['E'] = LineElement(
            n_5, n_bottom_spring, 0.00,
            np.zeros(3).copy(),
            np.zeros(3).copy(),
            self.rigid_util_section,
            self.parent,
            1.00,
            model_as={'type': 'elastic'},
            geom_transf='Linear',
            hidden_when_extruded=True)

        self.mat_imk = generate_IMK_mat(
            self.parent.section, self.parent.ends, self.parent.length_clear)

        rel_bottom_imk_spring = EndRelease(
            n_bottom_spring, n_bottom,
            {1: self.mat_fix,
             2: self.mat_fix,
             3: self.mat_fix,
             4: self.mat_fix,
             5: self.mat_fix,
             6: self.mat_imk},
            self.parent.x_axis, self.parent.y_axis)

        # store internal objects in corresponding lists
        self.internal_nodes['bottom_spring'] = n_bottom_spring
        self.internal_end_releases['release_imk'] = rel_bottom_imk_spring


@dataclass
class EndSegment_W_grav_shear_tab(EndSegment):
    """
    TODO ~ docstring
    TODO ~ all these pin-like endsegments can be combined to avoid all this
           code repetition
        (IMK, W_grav_shear_tab, pinned)
    """
    mat_fix: Material = field(repr=False)

    def __post_init__(self):

        super().__post_init__()

        # define IMK material based on element properties
        assert self.parent.section.sec_type == 'W', \
            "Error: Only W sections can be used"
        mat_fy = self.parent.section.material.parameters['Fy'] / 1.e3
        sec_zx = self.parent.section.properties['Zx']
        # gap = self.ends['gap']
        consider_composite = self.parent.ends['composite action']
        # Plastic moment of the section
        sec_mp = sec_zx * mat_fy * 1.e3
        if not consider_composite:
            m_max_pos = 0.121 * sec_mp
            m_max_neg = 0.121 * sec_mp
            m1_p = +0.521 * m_max_pos
            m1_n = -0.521 * m_max_neg
            m2_p = +0.967 * m_max_pos
            m2_n = -0.967 * m_max_neg
            m3_p = +1.000 * m_max_pos
            m3_n = -1.000 * m_max_pos
            m4_p = +0.901 * m_max_pos
            m4_n = -0.901 * m_max_neg
            th_1_p = 0.0045
            th_1_n = -0.0045
            th_2_p = 0.0465
            th_2_n = -0.0465
            th_3_p = 0.0750
            th_3_n = -0.0750
            th_4_p = 0.1000
            th_4_n = -0.1000
            rdispp = 0.57
            rdispn = 0.57
            rforcep = 0.40
            rforcen = 0.40
            uforcep = 0.05
            uforcen = 0.05
            gklim = 0.2
            gdlim = 0.1
            gflim = 0.0
            ge = 10
            dmgtype = 'energy'
            # th_u_p = + gap
            # th_u_n = - gap
        else:
            # m_max_pos = 0.35 * sec_mp
            # m_max_neg = 0.64*0.35 * sec_mp
            # m1_p = +0.250 * m_max_pos
            # m1_n = -0.250 * m_max_neg
            # m2_p = +0.90 * m_max_pos
            # m2_n = -1.00 * m_max_neg
            # m3_p = +1.00 * m_max_pos
            # m3_n = -1.01 * m_max_pos
            # m4_p = +0.530 * m_max_pos
            # m4_n = -0.540 * m_max_neg
            # th_1_p = 0.0042
            # th_1_n = -0.0042
            # th_2_p = 0.02
            # th_2_n = -0.011
            # th_3_p = 0.036
            # th_3_n = -0.03
            # th_4_p = 0.045
            # th_4_n = -0.055
            # rdispp = 0.40
            # rdispn = 0.50
            # rforcep = 0.13
            # rforcen = 0.53
            # uforcep = 0.01
            # uforcen = 0.05
            # gklim = 0.30
            # gdlim = 0.05
            # gflim = 0.05
            # ge = 10
            # dmgtype = 'energy'
            m_max_pos = 0.35 * sec_mp
            m_max_neg = 0.64*0.35 * sec_mp
            m1_p = +0.250 * m_max_pos
            m1_n = -0.250 * m_max_neg
            m2_p = +1.00 * m_max_pos
            m2_n = -1.00 * m_max_neg
            m3_p = +1.01 * m_max_pos
            m3_n = -1.01 * m_max_pos
            m4_p = +0.540 * m_max_pos
            m4_n = -0.540 * m_max_neg
            th_1_p = 0.0042
            th_1_n = -0.0042
            th_2_p = 0.011
            th_2_n = -0.011
            th_3_p = 0.03
            th_3_n = -0.03
            th_4_p = 0.055
            th_4_n = -0.055
            rdispp = 0.50
            rdispn = 0.50
            rforcep = 0.53
            rforcen = 0.53
            uforcep = 0.05
            uforcen = 0.05
            gklim = 0.30
            gdlim = 0.05
            gflim = 0.05
            ge = 10
            dmgtype = 'energy'
            # th_u_p = + gap
            # th_u_n = - gap
        params = {'m1_p': m1_p, 'th_1_p': th_1_p,
                  'm2_p': m2_p, 'th_2_p': th_2_p,
                  'm3_p': m3_p, 'th_3_p': th_3_p,
                  'm4_p': m4_p, 'th_4_p': th_4_p,
                  'm1_n': m1_n, 'th_1_n': th_1_n,
                  'm2_n': m2_n, 'th_2_n': th_2_n,
                  'm3_n': m3_n, 'th_3_n': th_3_n,
                  'm4_n': m4_n, 'th_4_n': th_4_n,
                  'rdispp': rdispp, 'rforcep': rforcep, 'uforcep': uforcep,
                  'rdispn': rdispn, 'rforcen': rforcen, 'uforcen': uforcen,
                  'gk1': 0., 'gk2': 0., 'gk3': 0.,
                  'gk4': 0., 'gklim': gklim,
                  'gd1': 0., 'gd2': 0., 'gd3': 0., 'gd4': 0,
                  'gdlim': gdlim, 'gF1': 0., 'gF2': 0.,
                  'gF3': 0., 'gF4': 0., 'gflim': gflim,
                  'ge': ge, 'dmgtype': dmgtype}
        self.mat_pinching = Material(
            'auto_pinching',
            'Pinching4',
            0.00,
            params)

        n_release = Node(self.n_internal.coords)

        self.add(n_release)

        if self.end == 'i':
            self.add(
                LineElement(
                    self.n_external, n_release, self.parent.ang,
                    self.offset(), np.zeros(3).copy(),
                    self.parent.section,
                    self.parent, self.parent.length_clear,
                    self.parent.model_as, self.parent.geom_transf))
            self.add(
                EndRelease(
                    n_release,
                    self.n_internal,
                    {1: self.mat_fix,
                     2: self.mat_fix,
                     3: self.mat_fix,
                     4: self.mat_fix,
                     5: self.mat_fix,
                     6: self.mat_pinching},
                    self.parent.x_axis, self.parent.y_axis))
        elif self.end == 'j':
            self.add(
                EndRelease(
                    self.n_internal,
                    n_release,
                    {1: self.mat_fix,
                     2: self.mat_fix,
                     3: self.mat_fix,
                     4: self.mat_fix,
                     5: self.mat_fix,
                     6: self.mat_pinching},
                    self.parent.x_axis, self.parent.y_axis))
            self.add(
                LineElement(
                    n_release, self.n_external, self.parent.ang,
                    np.zeros(3).copy(), self.offset(),
                    self.parent.section,
                    self.parent, self.parent.length_clear,
                    self.parent.model_as, self.parent.geom_transf))


@dataclass
class MiddleSegment:
    """
    This class represents components of a LineElementSequence.
    Please read the docstring of that class.
    See figure:
        https://notability.com/n/0Nvu2Gqlt3gfwEcQE2~RKf
    Attributes:
        n_i (Node): Transition internal node between
                    the EndSegment at end i and the MiddleSegment
        n_j (Node): Transition internal node between
                    the EndSegment at end j and the MiddleSegment
        ang, section, model_as, geom_transf:
            Arguments used for element creation. See LineElement.
        len_parent: Clear length of the parent LineElementSequence
        x_axis_parent (np.ndarray): X axis vector of the parent
                                    LineElementSequence
                                    (expressed in global coordinates)
        y_axis_parent, z_axis_parent (np.ndarray): Similar to X axis
        n_sub: Number of internal LineElements
        camber (float): Initial imperfection modeled as
                        parabolic camber, expressed
                        as a proportion of the element's
                        length.
    """

    parent: 'LineElementSequence'  # parent line element sequence
    offset_i: np.ndarray = field(repr=False)
    offset_j: np.ndarray = field(repr=False)
    camber: float = field(repr=False)

    def __post_init__(self):

        self.internal_nodes = OrderedDict()
        self.internal_line_elems = OrderedDict()

        p_i = self.parent.n_i.coords + self.offset_i
        p_j = self.parent.n_j.coords + self.offset_j

        internal_pt_coords = np.linspace(
            tuple(p_i),
            tuple(p_j),
            num=self.parent.n_sub+1)

        # apply camber
        if self.camber != 0.00:
            t_param = np.linspace(0, 1, self.parent.n_sub+1)
            x = self.camber * self.parent.length_clear
            y_deform = 4.0 * x * (t_param**2 - t_param)
            deformation_local = np.column_stack(
                (np.zeros(self.parent.n_sub+1), y_deform,
                 np.zeros(self.parent.n_sub+1)))
            deformation_global = (transformations.transformation_matrix(
                self.parent.x_axis_parent,
                self.parent.y_axis_parent,
                self.parent.z_axis_parent).T
                @ deformation_local.T).T
            internal_pt_coords += deformation_global

        # internal nodes (if required)
        if self.parent.n_sub > 1:
            for i in range(1, len(internal_pt_coords)-1):
                self.add(Node(internal_pt_coords[i]))
        # internal elements
        for i in range(self.parent.n_sub):
            if i == 0:
                node_i = self.parent.n_i
                o_i = self.offset_i
            else:
                node_i = nth_item(self.internal_nodes, i-1)
                o_i = np.zeros(3).copy()
            if i == self.parent.n_sub-1:
                node_j = self.parent.n_j
                o_j = self.offset_j
            else:
                node_j = nth_item(self.internal_nodes, i)
                o_j = np.zeros(3).copy()
            self.add(
                LineElement(
                    node_i, node_j, self.parent.ang,
                    o_i, o_j,
                    self.parent.section,
                    self.parent, self.parent.length_clear,
                    self.parent.model_as, self.parent.geom_transf))

    def crosses_point(self, pt: np.ndarray) -> bool:
        line = GridLine('', self.parent.n_i.coords[0:2],
                        self.parent.n_j.coords[0:2])
        return line.intersects_pt(pt)

    def connect(self, pt: np.ndarray, elev: float) \
            -> tuple[Node, np.ndarray]:
        """
        Perform a split or move internal nodes to accommodate
        for an internal node that can be used as a connection
        point for a beam-to-beam connection at a given point.
        """
        def do_split(
                segment: MiddleSegment,
                ielm: LineElement,
                proportion: float) -> Node:
            """
            Perform the splitting operation.
            """
            piece_i, piece_j, split_node = \
                ielm.split(proportion)
            internal_elems = list(segment.internal_line_elems.values())
            internal_nodes = list(segment.internal_nodes.values())
            # get index of the internal element in list
            idx = internal_elems.index(ielm)
            # remove the initial internal element
            del internal_elems[idx]
            # add the two pieces
            internal_elems.insert(idx, piece_j)
            internal_elems.insert(idx, piece_i)
            # add the internal node
            internal_nodes.insert(idx, split_node)
            ielm_ids = [elm.uid for elm in internal_elems]
            inod_ids = [ind.uid for ind in internal_nodes]
            segment.internal_line_elems = OrderedDict(zip(ielm_ids, internal_elems))
            segment.internal_nodes = OrderedDict(zip(inod_ids, internal_nodes))
            return split_node

        # check that the beam is horizontal
        # (otherwise this method doesn't work)
        assert np.abs(self.parent.x_axis[2]) < common.EPSILON, \
            'Error: Only horizontal supporting beams can be modeled.'
        # obtain offset
        delta_z = elev - self.parent.n_i.coords[2]
        offset = np.array((0., 0., delta_z))
        # now work on providing a connection internal
        # node at the given point.
        #
        # Implementation idea:
        # Split an existing internal element at the right place
        # and return the newly defined internal node.
        # But if that split results in a very short
        # and a very long internal element, don't split
        # and just move an existing internal node instead
        # (and update the internal elements connected to it)
        # Always split if just 1 internal element.

        # find the internal element that crosses
        # the point
        ielm = None
        for elm in self.internal_line_elems.values():
            if GridLine(
                'temp',
                elm.node_i.coords[0:2],
                    elm.node_j.coords[0:2]).intersects_pt(pt):
                ielm = elm
                break
        if not ielm:
            # This should never happen
            raise ValueError("Problem with beam-on-beam connection. " +
                             "No internal elements found on middle segment.")
        # length of the crossing internal element
        ielm_len = ielm.length_clear
        # length of the pieces if we were to split it
        piece_i = np.linalg.norm(ielm.node_i.coords[0:2]-pt)
        piece_j = np.linalg.norm(ielm.node_j.coords[0:2]-pt)
        # proportions
        proportion_i = piece_i / ielm_len
        proportion_j = piece_j / ielm_len
        min_proportion = np.minimum(proportion_i, proportion_j)
        # split or shift existing node?
        if len(self.internal_line_elems) == 1:
            # split regardless
            node = do_split(
                self,
                ielm,
                proportion_i)
        elif min_proportion < 0.25:
            # move
            internal_line_elems = list(self.internal_line_elems.values())
            # get index of the internal element in list
            idx = internal_line_elems.index(ielm)
            if proportion_i < proportion_j:
                # move node at end i
                if idx == 0:
                    # Can't move that node! Split instead.
                    node = do_split(
                        self,
                        ielm,
                        proportion_i)
                else:
                    # Can move node.
                    # find the other LineElement connected to the node
                    oelm = internal_line_elems[idx-1]
                    # node to be moved
                    node = ielm.node_i
                    # update coordinates
                    node.coords[0:2] = pt
                    # update internal_pts
                    ielm.internal_pt_i = ielm.node_i.coords + ielm.offset_i
                    oelm.internal_pt_j = oelm.node_j.coords + oelm.offset_j
            else:
                # move node at end j
                if idx+1 == len(internal_line_elems):
                    # Can't move that node! Split instead.
                    node = do_split(
                        self,
                        ielm,
                        proportion_i)
                else:
                    # Can move node.
                    # find the other LineElement connected to the node
                    oelm = internal_line_elems[idx+1]
                    # node to be moved
                    node = ielm.node_j
                    # update coordinates
                    node.coords[0:2] = pt
                    # update internal_pts
                    ielm.internal_pt_j = ielm.node_j.coords + ielm.offset_j
                    oelm.internal_pt_i = oelm.node_i.coords + oelm.offset_i
        else:
            # split
            node = do_split(
                self,
                ielm,
                proportion_i)

        return node, offset

    def add(self, thing):
        """
        Adds internal elements to the MiddleSegment.
        """
        if isinstance(thing, Node):
            self.internal_nodes[thing.uid] = thing
        elif isinstance(thing, LineElement):
            self.internal_line_elems[thing.uid] = thing
        else:
            raise ValueError(f'Unknown type: {type(thing)}')

    def first_line_elem(self):
        return list(self.internal_line_elems.values())[0]

    def last_line_elem(self):
        return list(self.internal_line_elems.values())[-1]

@dataclass
class LineElementSequence:
    """
    A LineElementSequence represents a collection
    of line elements connected in series.
    See figure:
    https://notability.com/n/0Nvu2Gqlt3gfwEcQE2~RKf
    """

    node_i: Node = field(repr=False)
    node_j: Node = field(repr=False)
    ang: float = field(repr=False)
    offset_i: np.ndarray = field(repr=False)
    offset_j: np.ndarray = field(repr=False)
    n_sub: int = field(repr=False)
    model_as: dict = field(repr=False)
    geom_transf: str = field(repr=False)
    placement: str = field(repr=False)
    ends: dict = field(repr=False)
    materials: Materials = field(repr=False)
    sections: Sections = field(repr=False)

    def __post_init__(self):

        self.uid = next(line_elem_seq_ids)

        p_i = self.node_i.coords + self.offset_i
        p_j = self.node_j.coords + self.offset_j

        self.length_clear = np.linalg.norm(p_j - p_i)

        self.section = self.sections.active

        # obtain offset from section (local system)
        dz, dy = self.sections.active.snap_points[self.placement]
        sec_offset_local = np.array([0.00, dy, dz])
        # retrieve local coordinate system
        self.x_axis, self.y_axis, self.z_axis = \
            transformations.local_axes_from_points_and_angle(
                p_i, p_j, self.ang)
        t_glob_to_loc = transformations.transformation_matrix(
            self.x_axis, self.y_axis, self.z_axis)
        t_loc_to_glob = t_glob_to_loc.T
        sec_offset_global = t_loc_to_glob @ sec_offset_local

        p_i += sec_offset_global
        p_j += sec_offset_global
        self.offset_i = self.offset_i.copy() + sec_offset_global
        self.offset_j = self.offset_j.copy() + sec_offset_global

        if self.ends['type'] == 'fixed':
            self.n_i = self.node_i
            self.n_j = self.node_j
            o_i = self.offset_i
            o_j = self.offset_j
        else:
            # location of transition from start segment
            # to middle segment
            start_loc = p_i\
                + self.x_axis * self.ends['end_dist'] * self.length_clear
            self.n_i = Node(start_loc)
            # location of transition from middle segment
            # to end segment
            end_loc = p_i\
                + self.x_axis * (1.0 - self.ends['end_dist']) \
                * self.length_clear
            self.n_j = Node(end_loc)
            o_i = np.zeros(3).copy()
            o_j = np.zeros(3).copy()

        self.middle_segment = MiddleSegment(
            self,
            offset_i=o_i,
            offset_j=o_j,
            camber=0.00)

        if self.ends['type'] in ['fixed', 'steel_W_PZ']:
            # note: panel zones are never defined at element definition.
            # they are automatically assigned during model postprocessing
            self.end_segment_i = EndSegment_Fixed(
                self, self.node_i, self.n_i, "i")
            self.end_segment_j = EndSegment_Fixed(
                self, self.node_j, self.n_j, "j")

        elif self.ends['type'] == 'pinned':
            self.end_segment_i = EndSegment_Pinned(
                self, self.node_i, self.n_i, "i",
                self.materials.registry['fix'])
            self.end_segment_j = EndSegment_Pinned(
                self, self.node_j, self.n_j, "j",
                self.materials.registry['fix'])

        elif self.ends['type'] == 'fixed-pinned':
            self.end_segment_i = EndSegment_Fixed(
                self, self.node_i, self.n_i, "i")
            self.end_segment_j = EndSegment_Pinned(
                self, self.node_j, self.n_j, "j",
                self.materials.registry['fix'])

        elif self.ends['type'] == 'RBS':
            self.end_segment_i = EndSegment_RBS(
                self, self.node_i, self.n_i, "i")
            self.end_segment_j = EndSegment_RBS(
                self, self.node_i, self.n_j, "j")

        elif self.ends['type'] == 'RBS_j':
            self.end_segment_i = EndSegment_Fixed(
                self, self.node_i, self.n_i, "i")
            self.end_segment_j = EndSegment_RBS(
                self, self.node_i, self.n_j, "j")

        elif self.ends['type'] in ['steel_W_IMK', 'steel_W_PZ_IMK']:
            self.end_segment_i = EndSegment_IMK(
                self, self.node_i, self.n_i,
                "i", self.materials.registry['fix'])
            self.end_segment_j = EndSegment_IMK(
                self, self.node_j, self.n_j,
                "j", self.materials.registry['fix'])

        elif self.ends['type'] == 'steel W shear tab':
            self.end_segment_i = EndSegment_W_grav_shear_tab(
                self, self.node_i, self.n_i,
                "i", self.materials.registry['fix'])
            self.end_segment_j = EndSegment_W_grav_shear_tab(
                self, self.node_j, self.n_j,
                "j", self.materials.registry['fix'])
        else:
            raise ValueError(f"Invalid end type: {self.ends['type']}")

    def snap_offset(self, tag: str):
        """
        Used to easily retrieve the required
        offset to connect a beam's end to the
        top node of a column.
        The method is called on the column object.
        Args:
            tag (str): Placement tag (see Section)
        Returns:
            The offset vector starting from the primary
            node and pointing to the connection point
            on the column, expressed in the global
            coordinate system.
        """
        # obtain offset from section (local system)
        # given the snap point tag
        dz, dy = self.section.snap_points[tag]
        snap_offset = np.array([0.00, dy, dz])
        t_glob_to_loc = transformations.transformation_matrix(
            self.x_axis, self.y_axis, self.z_axis)
        t_loc_to_glob = t_glob_to_loc.T
        snap_offset_global = t_loc_to_glob @ snap_offset

        return snap_offset_global

    def add_udl_glob(self, udl: np.ndarray, ltype='other'):
        """
        Adds a uniformly distributed load
        to the existing udl of the element.
        The load is defined
        with respect to the global coordinate system
        of the building, and it is converted to the
        local coordinate system prior to adding it.
        Args:
            udl (np.ndarray): Array of size 3 containing components of the
                              uniformly distributed load that is applied
                              to the clear length of the element, acting
                              on the global x, y, and z directions, in the
                              direction of the global axes.
        """
        for elm in self.internal_line_elems():
            elm.add_udl_glob(udl, ltype=ltype)

    def apply_self_weight_and_mass(self):
        """
        Applies self-weight as a and distributes mass
        by lumping it at the nodes where the ends of the
        internal elements are connected.
        Args:
            multiplier: A parameter that is multiplied to the
                        automatically obtained self-weight and self-mass.
        """

        for sub_elm in self.internal_line_elems():
            if sub_elm.section.sec_type == 'utility':
                # line elements that function as rigid links
                # and other such elements that do not contribute
                # to self-weight/mass etc. --> skip
                continue
            cross_section_area = sub_elm.section.properties["A"]
            # lb-s**2/in**2
            mass_per_length = cross_section_area * \
                sub_elm.section.material.density
            # lb/in
            weight_per_length = mass_per_length * common.G_CONST
            sub_elm.add_udl_glob(
                np.array([0., 0., -weight_per_length]), ltype='self')
            total_mass_per_length = - \
                sub_elm.get_udl_no_floor_glob()[2] / common.G_CONST
            mass = total_mass_per_length * \
                sub_elm.length_clear / 2.00  # lb-s**2/in
            sub_elm.node_i.mass += np.array([mass, mass, mass])
            sub_elm.node_j.mass += np.array([mass, mass, mass])

    def primary_nodes(self):
        return [self.node_i, self.node_j]

    def internal_nodes(self):
        result = []
        result.extend(self.end_segment_i.internal_nodes.values())
        result.extend(self.middle_segment.internal_nodes.values())
        result.extend(self.end_segment_j.internal_nodes.values())
        return result

    def internal_end_releases(self):
        result = []
        result.extend(self.end_segment_i.internal_end_releases.values())
        result.extend(self.end_segment_j.internal_end_releases.values())
        return result

    def internal_line_elems(self):
        result = []
        result.extend(self.end_segment_i.internal_line_elems.values())
        result.extend(self.middle_segment.internal_line_elems.values())
        result.extend(self.end_segment_j.internal_line_elems.values())
        return result


@dataclass
class LineElementSequences:
    """
    This class is a collector for line element sequences.
    """

    registry: OrderedDict[int, LineElementSequence] = field(
        default_factory=OrderedDict, repr=False)

    def add(self, elm: LineElementSequence):
        """
        Add an element in the registry
        """

        if elm.uid not in self.registry:
            self.registry[elm.uid] = elm
        else:
            raise ValueError('LineElementSequence already exists')

    def remove(self, key: int):
        """
        Remove an element from the element collection,
        if it was there, using it's unique ID
        """
        if key in self.registry:
            self.registry.pop(key)

    def internal_elems(self):
        result = []
        for element in self.registry.values():
            result.extend(element.internal_elems)
        return result
