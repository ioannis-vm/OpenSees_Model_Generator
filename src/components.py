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
from functools import total_ordering
from itertools import count
import numpy as np
from grids import GridLine
from node import Node
from material import Material
from section import Section
from utility import common
from utility import transformations


line_elem_ids = count(0)
end_release_ids = count(0)


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
    Args:
        dofs (list[int]): List containing the DOFs that will be
        matched to a uniaxialMaterial.
        materials (list[Material]): List containing the corresponding
        material objects. dofs and materials are matched one to one.
    """

    node_i: Node
    node_j: Node
    dofs: list[int]
    materials: list[Material]
    x_vec: np.ndarray
    y_vec: np.ndarray

    def __post_init__(self):
        self.uniq_id = next(end_release_ids)
        assert len(self.dofs) == len(self.materials), \
            "Dimensions don't match."


@dataclass
class LineElement:
    """
    Linear finite element class.
    This class represents the most primitive linear element,
    on which more complex classes build upon.
    Attributes:
        uniq_id (int): unique identifier
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
        geomTransf: {Linear, PDelta, Corotational}
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

    node_i: Node
    node_j: Node
    ang: float
    offset_i: np.ndarray
    offset_j: np.ndarray
    section: Section
    len_parent: float
    model_as: dict
    geomTransf: str
    udl_self: np.ndarray = field(default_factory=lambda: np.zeros(shape=3))
    udl_fl: np.ndarray = field(default_factory=lambda: np.zeros(shape=3))
    udl_other: np.ndarray = field(default_factory=lambda: np.zeros(shape=3))
    hidden_when_extruded: bool = field(default=False)
    hidden_as_line: bool = field(default=False)

    def __post_init__(self):

        # ---  this is needed for tributary area analysis  ---
        # ( ... since adding support for steel W panel zones              )
        # ( the convention that every closed shape in plan view could     )
        # ( be extracted by following the connevtivity of line elements   )
        # ( was broken, because line elements connected to panel zones    )
        # ( create a gap between the front and the back nodes of the      )
        # ( panel zone. To overcome this, we retain their prior           )
        # ( connevtivity information (before preprocessing) so that       )
        # ( we can still use it to do the tributary area analysis without )
        # ( having to fundamentaly change that part of the code.          )
        self.node_i_trib = self.node_i
        self.node_j_trib = self.node_j
        # ~~~~

        self.uniq_id = next(line_elem_ids)
        self.tributary_area = 0.00
        # local axes with respect to the global coord system
        self.internal_pt_i = self.node_i.coords + self.offset_i
        self.internal_pt_j = self.node_j.coords + self.offset_j
        self.x_axis, self.y_axis, self.z_axis = \
            transformations.local_axes_from_points_and_angle(
                self.internal_pt_i, self.internal_pt_j, self.ang)
        self.len_proportion = self.length_clear() / self.len_parent
        # note: we are using dispBeamColumn, so 2 integration points
        # suffice. When using forceBeamColumn, they should be increased.
        if self.len_proportion > 0.75:
            n_p = 2
        elif self.len_proportion > 0.50:
            n_p = 2
        else:
            n_p = 2
        self.n_p = n_p

    def length_clear(self):
        """
        Computes the clear length of the element, excluding the offsets.
        Returns:
            float: distance
        """
        p_i = self.node_i.coords + self.offset_i
        p_j = self.node_j.coords + self.offset_j
        return np.linalg.norm(p_j - p_i)

    def update_axes(self):
        """
        Recalculate the local axes of the elements, in case
        the nodes were changed after the element's definition.
        """
        self.internal_pt_i = self.node_i.coords + self.offset_i
        self.internal_pt_j = self.node_j.coords + self.offset_j
        self.x_axis, self.y_axis, self.z_axis = \
            transformations.local_axes_from_points_and_angle(
                self.internal_pt_i, self.internal_pt_j, self.ang)

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
            (self.length_clear() * proportion)
        split_node = Node(split_location)
        piece_i = LineElement(
            self.node_i, split_node,
            self.ang, self.offset_i, np.zeros(3).copy(),
            self.section, self.len_parent, self.model_as,
            self.geomTransf, self.udl_self.copy(), self.udl_fl.copy(),
            self.udl_other.copy())
        piece_j = LineElement(
            split_node, self.node_j,
            self.ang, np.zeros(3).copy(), self.offset_j,
            self.section, self.len_parent, self.model_as,
            self.geomTransf, self.udl_self.copy(), self.udl_fl.copy(),
            self.udl_other.copy())
        return piece_i, piece_j, split_node


@dataclass
class EndSegment:
    """
    This class represents an end segment of a LineElementSequence.
    Please read the docstring of that class.
    See figure:
        https://notability.com/n/0Nvu2Gqlt3gfwEcQE2~RKf
    Attributes:
        n_external (Node): Primary node of the structure
        n_internal (Node): Transition internal node between
                           the EndSegment and the MiddleSegment
        offset (np.ndarray): Offset vector, pointing from the
                             primary node to the internal point
        internal_pt (np.ndarray): The internal point (we don't
                                  define a node there, but we
                                  need the coordinates.)
        internal_nodes (list[Node]): List of internal nodes of
                                     the EndSegment
        internal_elems (list): List of internal elements
    """
    n_external: Node
    n_internal: Node
    offset: np.ndarray

    def __post_init__(self):
        self.internal_pt = self.n_external.coords + self.offset
        self.internal_nodes = []
        self.internal_elems = []


@dataclass
class EndSegment_Fixed(EndSegment):
    """
    This class represents a fixed end segment of a LineElementSequence.
    Please read the docstring of that class.
    See figure:
        https://notability.com/n/0Nvu2Gqlt3gfwEcQE2~RKf
    Attributes:
        See the attributes of EndSegment
    Additional attributes:
        end: Whether the EndSegment corresponds to the
             start ("i") or the end ("j") of the LineElementSequence
        len_parent: Clear length of the parent LineElementSequence
        ang, section, model_as, geomTransf:
            Arguments used for element creation. See LineElement.
    """
    end: str
    len_parent: float
    ang: float
    section: Section
    model_as: dict
    geomTransf: str

    def __post_init__(self):
        super().__post_init__()
        self.internal_nodes.append(
            self.n_internal)
        if self.end == 'i':
            self.internal_elems.append(
                LineElement(
                    self.n_external, self.n_internal, self.ang,
                    self.offset, np.zeros(3).copy(),
                    self.section, self.len_parent,
                    self.model_as, self.geomTransf))
        elif self.end == 'j':
            self.internal_elems.append(
                LineElement(
                    self.n_internal, self.n_external, self.ang,
                    np.zeros(3).copy(), self.offset,
                    self.section, self.len_parent,
                    self.model_as, self.geomTransf))


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
        end: Whether the EndSegment corresponds to the
             start ("i") or the end ("j") of the LineElementSequence
        len_parent: Clear length of the parent LineElementSequence
        ang, section, model_as, geomTransf:
            Arguments used for element creation. See LineElement.
        x_axis (np.ndarray): X axis vector of the parent LineElementSequence
                             (expressed in global coordinates)
        y_axis (np.ndarray): Similar to X axis
        mat_fix (Material): Linear elastic material with a very high stiffness
                            See the Materials class.
    """

    end: str
    len_parent: float
    ang: float
    section: Section
    model_as: dict
    geomTransf: str
    x_axis: np.ndarray
    y_axis: np.ndarray
    mat_fix: Material

    def __post_init__(self):
        super().__post_init__()
        self.internal_pt = self.n_external.coords + self.offset
        self.internal_nodes.append(
            self.n_internal)
        n_release = Node(self.n_internal.coords)
        self.internal_nodes.append(n_release)
        if self.end == 'i':
            self.internal_elems.append(
                LineElement(
                    self.n_external, n_release, self.ang,
                    self.offset, np.zeros(3).copy(),
                    self.section, self.len_parent,
                    self.model_as, self.geomTransf))
            self.internal_elems.append(
                EndRelease(
                    n_release,
                    self.n_internal,
                    [1, 2, 3, 4],
                    [self.mat_fix]*4,
                    self.x_axis, self.y_axis))
        elif self.end == 'j':
            self.internal_elems.append(
                EndRelease(
                    self.n_internal,
                    n_release,
                    [1, 2, 3, 4],
                    [self.mat_fix]*4,
                    self.x_axis, self.y_axis))
            self.internal_elems.append(
                LineElement(
                    n_release, self.n_external, self.ang,
                    np.zeros(3).copy(), self.offset,
                    self.section, self.len_parent,
                    self.model_as, self.geomTransf))


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
        ang, section, model_as, geomTransf:
            Arguments used for element creation. See LineElement.
        x_axis (np.ndarray): X axis vector of the parent LineElementSequence
                             (expressed in global coordinates)
        rbs_length (float): Length of the reduced beam segment
                            (expressed in length units, not proportional)
        rbs_reduction (float): Proportion of the reduced beam section's width
                               relative to the initial section.
        n_sub (int): Number of LineElement objects representing the RBS segment
    """
    end: str
    len_parent: float
    ang: float
    section: Section
    model_as: dict
    geomTransf: str
    x_axis: np.ndarray
    rbs_length: float
    rbs_reduction: float
    rbs_n_sub: int

    def __post_init__(self):
        super().__post_init__()
        self.internal_pt = self.n_external.coords + self.offset

        # Generate the reduced section
        reduced_sections = []
        bf = self.section.properties['bf']
        c = (1.00 - self.rbs_reduction) * bf / 2.
        rbs_length = self.rbs_length

        # i versus j matters because of the orientation of things

        if self.end == 'i':
            for x in np.linspace(
                    0.00, 1.00, num=self.rbs_n_sub + 2)[1::][::-1]:
                # define and add internal nodes
                self.internal_nodes.append(
                    Node(self.n_internal.coords -
                         self.x_axis * self.rbs_length * x)
                )
                # define reduced sections
                x_sec = x - 1/(self.rbs_n_sub+1)
                c_cur = 1./8.*(8.*c - np.sqrt((
                    rbs_length**2+4.*c**2)**2/(c)**2) +
                    np.sqrt(rbs_length**4/c**2+16.*c**2-8.*rbs_length**2 *
                            (1.-8.*x_sec+8.*x_sec**2)))
                reduction_cur = (bf - 2. * c_cur) / bf
                reduced_sections.append(self.section.rbs(reduction_cur))
            self.internal_nodes.append(self.n_internal)
            self.internal_elems.append(
                LineElement(
                    self.n_external, self.internal_nodes[0],
                    self.ang,
                    self.offset, np.zeros(3).copy(),
                    self.section, self.len_parent,
                    self.model_as, self.geomTransf))
            for i in range(self.rbs_n_sub+1):
                self.internal_elems.append(
                    LineElement(
                        self.internal_nodes[i],
                        self.internal_nodes[i+1],
                        self.ang,
                        np.zeros(3).copy(), np.zeros(3).copy(),
                        reduced_sections[i], self.len_parent,
                        self.model_as, self.geomTransf))

        elif self.end == 'j':
            self.internal_nodes.append(self.n_internal)
            for x in np.linspace(
                    0.00, 1.00, num=self.rbs_n_sub + 2)[1::]:
                # define and add internal nodes
                self.internal_nodes.append(
                    Node(self.n_internal.coords +
                         self.x_axis * self.rbs_length * x)
                )
                # define reduced sections
                x_sec = x - 1/(self.rbs_n_sub+1)
                c_cur = 1./8.*(8.*c - np.sqrt((
                    rbs_length**2+4.*c**2)**2/(c)**2) +
                    np.sqrt(rbs_length**4/c**2+16.*c**2-8.*rbs_length**2 *
                            (1.-8.*x_sec+8.*x_sec**2)))
                reduction_cur = (bf - 2. * c_cur) / bf
                reduced_sections.append(self.section.rbs(reduction_cur))
            for i in range(self.rbs_n_sub+1):
                self.internal_elems.append(
                    LineElement(
                        self.internal_nodes[i],
                        self.internal_nodes[i+1],
                        self.ang,
                        np.zeros(3).copy(), np.zeros(3).copy(),
                        reduced_sections[i], self.len_parent,
                        self.model_as, self.geomTransf))
            self.internal_elems.append(
                LineElement(
                    self.internal_nodes[-1], self.n_external, self.ang,
                    np.zeros(3).copy(), self.offset,
                    self.section, self.len_parent,
                    self.model_as, self.geomTransf))


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
        ang, section, model_as, geomTransf:
            Arguments used for element creation. See LineElement.
        x_axis (np.ndarray): X axis vector of the parent LineElementSequence
                             (expressed in global coordinates)
        rbs_length (float): Length of the reduced beam segment
                            (expressed in length units, not proportional)
        rbs_reduction (float): Proportion of the reduced beam section's width
                               relative to the initial section.
        n_sub (int): Number of LineElement objects representing the RBS segment
    """
    end: str
    len_parent: float
    ang: float
    section: Section
    model_as: dict
    geomTransf: str
    x_axis: np.ndarray
    y_axis: np.ndarray
    mat_fix: Material
    mat_IMK: Material

    def __post_init__(self):
        super().__post_init__()
        self.internal_pt = self.n_external.coords + self.offset
        self.internal_nodes.append(
            self.n_internal)
        n_release = Node(self.n_internal.coords)
        self.internal_nodes.append(n_release)
        if self.end == 'i':
            self.internal_elems.append(
                LineElement(
                    self.n_external, n_release, self.ang,
                    self.offset, np.zeros(3).copy(),
                    self.section, self.len_parent,
                    self.model_as, self.geomTransf))
            self.internal_elems.append(
                EndRelease(
                    n_release,
                    self.n_internal,
                    [1, 2, 3, 4, 6],
                    [self.mat_fix]*4 + [self.mat_IMK],
                    self.x_axis, self.y_axis))
        elif self.end == 'j':
            self.internal_elems.append(
                EndRelease(
                    self.n_internal,
                    n_release,
                    [1, 2, 3, 4, 6],
                    [self.mat_fix]*4 + [self.mat_IMK],
                    self.x_axis, self.y_axis))
            self.internal_elems.append(
                LineElement(
                    n_release, self.n_external, self.ang,
                    np.zeros(3).copy(), self.offset,
                    self.section, self.len_parent,
                    self.model_as, self.geomTransf))


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
    Attributes:
        See attributes of EndSegment
    Additional Attributes:
        TODO
    """
    col_section: Section
    col_ang: float
    rigid_util_section: Section
    x_axis: np.ndarray
    y_axis: np.ndarray
    mat_fix: Material
    mat_pz: Material
    beam_depth: float
    # internal_nodes: list[Node]
    # internal_elems: list

    def __post_init__(self):
        super().__post_init__()
        column_depth = self.col_section.properties['d']

        # define nodes
        n_top = self.n_external
        n_bottom = self.n_internal
        n_bottom.coords = n_top.coords + self.x_axis * self.beam_depth

        # these two attributes are used for plotting
        self.internal_pt_i = n_top.coords
        self.internal_pt_j = n_bottom.coords

        n_1 = Node(n_top.coords + self.y_axis * column_depth / 2.)
        n_2 = Node(n_top.coords + self.y_axis * column_depth / 2.)
        self.n_front = Node(n_top.coords + self.y_axis * column_depth / 2.
                            + self.x_axis * self.beam_depth / 2.)
        n_3 = Node(n_top.coords + self.y_axis * column_depth / 2.
                   + self.x_axis * self.beam_depth)
        n_4 = Node(n_top.coords + self.y_axis * column_depth / 2.
                   + self.x_axis * self.beam_depth)
        n_5 = Node(n_top.coords - self.y_axis * column_depth / 2.
                   + self.x_axis * self.beam_depth)
        n_6 = Node(n_top.coords - self.y_axis * column_depth / 2.
                   + self.x_axis * self.beam_depth)
        self.n_back = Node(n_top.coords - self.y_axis * column_depth / 2.
                           + self.x_axis * self.beam_depth / 2.)
        n_7 = Node(n_top.coords - self.y_axis * column_depth / 2.)
        n_8 = Node(n_top.coords - self.y_axis * column_depth / 2.)

        # define rigid line elements connecting the nodes
        col_ang = self.col_ang
        elm_a = LineElement(n_top, n_1, 0.00,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)
        elm_b = LineElement(n_2, self.n_front, col_ang,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)
        elm_c = LineElement(self.n_front, n_3, col_ang,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)
        elm_d = LineElement(n_bottom, n_4, 0.00,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)
        elm_e = LineElement(n_5, n_bottom, 0.00,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)
        elm_f = LineElement(self.n_back, n_6, col_ang,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)
        elm_g = LineElement(n_7, self.n_back, col_ang,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)
        elm_h = LineElement(n_8, n_top, 0.00,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)

        # define releases
        rel_spring = EndRelease(
            n_1, n_2, [1, 2, 3, 4, 5, 6],
            [self.mat_fix, self.mat_fix, self.mat_fix,
             self.mat_fix, self.mat_fix, self.mat_pz],
            self.x_axis, self.y_axis)
        rel_bottom_front = EndRelease(
            n_3, n_4, [1, 2, 3, 4, 5],
            [self.mat_fix, self.mat_fix, self.mat_fix,
             self.mat_fix, self.mat_fix],
            self.x_axis, self.y_axis)
        rel_bottom_back = EndRelease(
            n_5, n_6, [1, 2, 3, 4, 5],
            [self.mat_fix, self.mat_fix, self.mat_fix,
             self.mat_fix, self.mat_fix],
            self.x_axis, self.y_axis)
        rel_top_back = EndRelease(
            n_7, n_8, [1, 2, 3, 4, 5],
            [self.mat_fix, self.mat_fix, self.mat_fix,
             self.mat_fix, self.mat_fix],
            self.x_axis, self.y_axis)

        # store internal objects in corresponding lists
        self.internal_nodes = [
            n_bottom, n_1, n_2, self.n_front, n_3, n_4,
            n_5, n_6, self.n_back, n_7, n_8]
        self.internal_elems = [
            elm_a, elm_b, elm_c, elm_d, elm_e,
            elm_f, elm_g, elm_h,
            rel_spring, rel_bottom_front,
            rel_bottom_back, rel_top_back
        ]

    def length_clear(self):
        """
        (used in postprocessing_3D extruded view plots).
        Clear length of the panel zone (in the longitudinal direction)
        """
        return np.linalg.norm(self.n_external.coords - self.n_internal.coords)


@dataclass
class EndSegment_Steel_W_PanelZone_IMK(EndSegment):
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
    Attributes:
        See attributes of EndSegment
    Additional Attributes:
        TODO
    """
    col_section: Section
    col_ang: float
    rigid_util_section: Section
    x_axis: np.ndarray
    y_axis: np.ndarray
    mat_fix: Material
    mat_pz: Material
    mat_IMK: Material
    beam_depth: float

    def __post_init__(self):
        super().__post_init__()
        column_depth = self.col_section.properties['d']

        # define nodes
        n_top = self.n_external
        n_bottom = self.n_internal
        n_bottom.coords = n_top.coords + self.x_axis * self.beam_depth
        n_bottom_spring = Node(n_bottom.coords)

        # these two attributes are used for plotting
        self.internal_pt_i = n_top.coords
        self.internal_pt_j = n_bottom.coords

        n_1 = Node(n_top.coords + self.y_axis * column_depth / 2.)
        n_2 = Node(n_top.coords + self.y_axis * column_depth / 2.)
        self.n_front = Node(n_top.coords + self.y_axis * column_depth / 2.
                            + self.x_axis * self.beam_depth / 2.)
        n_3 = Node(n_top.coords + self.y_axis * column_depth / 2.
                   + self.x_axis * self.beam_depth)
        n_4 = Node(n_top.coords + self.y_axis * column_depth / 2.
                   + self.x_axis * self.beam_depth)
        n_5 = Node(n_top.coords - self.y_axis * column_depth / 2.
                   + self.x_axis * self.beam_depth)
        n_6 = Node(n_top.coords - self.y_axis * column_depth / 2.
                   + self.x_axis * self.beam_depth)
        self.n_back = Node(n_top.coords - self.y_axis * column_depth / 2.
                           + self.x_axis * self.beam_depth / 2.)
        n_7 = Node(n_top.coords - self.y_axis * column_depth / 2.)
        n_8 = Node(n_top.coords - self.y_axis * column_depth / 2.)

        # define rigid line elements connecting the nodes
        col_ang = self.col_ang
        elm_a = LineElement(n_top, n_1, 0.00,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)
        elm_b = LineElement(n_2, self.n_front, col_ang,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)
        elm_c = LineElement(self.n_front, n_3, col_ang,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)
        elm_d = LineElement(n_bottom_spring, n_4, 0.00,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)
        elm_e = LineElement(n_5, n_bottom_spring, 0.00,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)
        elm_f = LineElement(self.n_back, n_6, col_ang,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)
        elm_g = LineElement(n_7, self.n_back, col_ang,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)
        elm_h = LineElement(n_8, n_top, 0.00,
                            np.zeros(3).copy(),
                            np.zeros(3).copy(),
                            self.rigid_util_section,
                            1.00,
                            model_as={'type': 'elastic'},
                            geomTransf='Linear',
                            hidden_when_extruded=True)

        # define releases
        rel_spring = EndRelease(
            n_1, n_2, [1, 2, 3, 4, 5, 6],
            [self.mat_fix, self.mat_fix, self.mat_fix,
             self.mat_fix, self.mat_fix, self.mat_pz],
            self.x_axis, self.y_axis)
        rel_bottom_front = EndRelease(
            n_3, n_4, [1, 2, 3, 4, 5],
            [self.mat_fix, self.mat_fix, self.mat_fix,
             self.mat_fix, self.mat_fix],
            self.x_axis, self.y_axis)
        rel_bottom_back = EndRelease(
            n_5, n_6, [1, 2, 3, 4, 5],
            [self.mat_fix, self.mat_fix, self.mat_fix,
             self.mat_fix, self.mat_fix],
            self.x_axis, self.y_axis)
        rel_top_back = EndRelease(
            n_7, n_8, [1, 2, 3, 4, 5],
            [self.mat_fix, self.mat_fix, self.mat_fix,
             self.mat_fix, self.mat_fix],
            self.x_axis, self.y_axis)
        rel_bottom_IMK_spring = EndRelease(
            n_bottom_spring, n_bottom, [1, 2, 3, 4, 5, 6],
            [self.mat_fix, self.mat_fix, self.mat_fix,
             self.mat_fix, self.mat_fix, self.mat_IMK],
            self.x_axis, self.y_axis)

        # store internal objects in corresponding lists
        self.internal_nodes = [
            n_bottom, n_bottom_spring, n_1, n_2, self.n_front, n_3, n_4,
            n_5, n_6, self.n_back, n_7, n_8]
        self.internal_elems = [
            elm_a, elm_b, elm_c, elm_d, elm_e,
            elm_f, elm_g, elm_h,
            rel_spring, rel_bottom_front,
            rel_bottom_back, rel_top_back,
            rel_bottom_IMK_spring
        ]

    def length_clear(self):
        """
        (used in postprocessing_3D extruded view plots).
        Clear length of the panel zone (in the longitudinal direction)
        """
        return np.linalg.norm(self.n_external.coords - self.n_internal.coords)


@dataclass
class EndSegment_W_grav_shear_tab(EndSegment):
    """
    TODO ~ docstring
    TODO ~ all these pin-like endsegments can be combined to avoid all this
           code repetition
        (IMK, W_grav_shear_tab, pinned)
    """
    end: str
    len_parent: float
    ang: float
    section: Section
    model_as: dict
    geomTransf: str
    x_axis: np.ndarray
    y_axis: np.ndarray
    mat_fix: Material
    mat_pinching: Material

    def __post_init__(self):
        super().__post_init__()
        self.internal_pt = self.n_external.coords + self.offset
        self.internal_nodes.append(
            self.n_internal)
        n_release = Node(self.n_internal.coords)
        self.internal_nodes.append(n_release)
        if self.end == 'i':
            self.internal_elems.append(
                LineElement(
                    self.n_external, n_release, self.ang,
                    self.offset, np.zeros(3).copy(),
                    self.section, self.len_parent,
                    self.model_as, self.geomTransf))
            self.internal_elems.append(
                EndRelease(
                    n_release,
                    self.n_internal,
                    [1, 2, 3, 4, 5, 6],
                    [self.mat_fix]*5 + [self.mat_pinching],
                    self.x_axis, self.y_axis))
        elif self.end == 'j':
            self.internal_elems.append(
                EndRelease(
                    self.n_internal,
                    n_release,
                    [1, 2, 3, 4, 5, 6],
                    [self.mat_fix]*5 + [self.mat_pinching],
                    self.x_axis, self.y_axis))
            self.internal_elems.append(
                LineElement(
                    n_release, self.n_external, self.ang,
                    np.zeros(3).copy(), self.offset,
                    self.section, self.len_parent,
                    self.model_as, self.geomTransf))


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
        ang, section, model_as, geomTransf:
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

    n_i: Node
    n_j: Node
    ang: float
    section: Section
    model_as: dict
    geomTransf: str
    len_parent: float
    x_axis_parent: np.ndarray
    y_axis_parent: np.ndarray
    z_axis_parent: np.ndarray
    n_sub: int
    camber: float

    def __post_init__(self):
        self.internal_nodes = []
        self.internal_elems = []

        p_i = self.n_i.coords
        p_j = self.n_j.coords

        internal_pt_coords = np.linspace(
            tuple(p_i),
            tuple(p_j),
            num=self.n_sub+1)

        # apply camber
        if self.camber != 0.00:
            t_param = np.linspace(0, 1, self.n_sub+1)
            x = self.camber * self.len_parent
            y_deform = 4.0 * x * (t_param**2 - t_param)
            deformation_local = np.column_stack(
                (np.zeros(self.n_sub+1), y_deform, np.zeros(self.n_sub+1)))
            deformation_global = (transformations.transformation_matrix(
                self.x_axis_parent, self.y_axis_parent, self.z_axis_parent).T
                @ deformation_local.T).T
            internal_pt_coords += deformation_global

        # internal nodes (if required)
        if self.n_sub > 1:
            for i in range(1, len(internal_pt_coords)-1):
                self.internal_nodes.append(Node(internal_pt_coords[i]))
        # internal elements
        for i in range(self.n_sub):
            if i == 0:
                node_i = self.n_i
            else:
                node_i = self.internal_nodes[i-1]
            if i == self.n_sub-1:
                node_j = self.n_j
            else:
                node_j = self.internal_nodes[i]
            self.internal_elems.append(
                LineElement(
                    node_i, node_j, self.ang,
                    np.zeros(3).copy(), np.zeros(3).copy(),
                    self.section, self.len_parent,
                    self.model_as, self.geomTransf))

    def crosses_point(self, pt: np.ndarray) -> bool:
        line = GridLine('', self.n_i.coords[0:2], self.n_j.coords[0:2])
        return line.intersects_pt(pt)

    def connect(self, pt: np.ndarray, elev: float) \
            -> tuple[Node, np.ndarray]:
        """
        Perform a split or move internal nodes to accommodate
        for an internal node that can be used as a connection
        point for a beam-to-beam connection at a given point.
        """
        def do_split(ielm: LineElement,
                     proportion: float,
                     internal_elems: list[LineElement],
                     internal_nodes: list[Node]) -> Node:
            """
            Perform the splitting operation.
            """
            piece_i, piece_j, split_node = \
                ielm.split(proportion)
            # get index of the internal element in list
            idx = internal_elems.index(ielm)
            # remove the initial internal element
            del internal_elems[idx]
            # add the two pieces
            internal_elems.insert(idx, piece_j)
            internal_elems.insert(idx, piece_i)
            # add the internal node
            internal_nodes.insert(idx, split_node)
            return split_node

        # check that the beam is horizontal
        # (otherwise this method doesn't work)
        assert np.abs(self.x_axis_parent[2]) < common.EPSILON, \
            'Error: Only horizontal supporting beams can be modeled.'
        # obtain offset
        delta_z = elev - self.n_i.coords[2]
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
        for elm in self.internal_elems:
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
        ielm_len = ielm.length_clear()
        # length of the pieces if we were to split it
        piece_i = np.linalg.norm(ielm.node_i.coords[0:2]-pt)
        piece_j = np.linalg.norm(ielm.node_j.coords[0:2]-pt)
        # proportions
        proportion_i = piece_i / ielm_len
        proportion_j = piece_j / ielm_len
        min_proportion = np.minimum(proportion_i, proportion_j)
        # split or shift existing node?
        if len(self.internal_elems) == 1:
            # split regardless
            node = do_split(
                ielm,
                proportion_i,
                self.internal_elems,
                self.internal_nodes)
        elif min_proportion < 0.25:
            # move
            # get index of the internal element in list
            idx = self.internal_elems.index(ielm)
            if proportion_i < proportion_j:
                # move node at end i
                if idx == 0:
                    # Can't move that node! Split instead.
                    node = do_split(
                        ielm,
                        proportion_i,
                        self.internal_elems,
                        self.internal_nodes)
                else:
                    # Can move node.
                    # find the other LineElement connected to the node
                    oelm = self.internal_elems[idx-1]
                    # node to be moved
                    node = ielm.node_i
                    # update coordinates
                    node.coords[0:2] = pt
                    # update internal_pts
                    ielm.internal_pt_i = ielm.node_i.coords + ielm.offset_i
                    oelm.internal_pt_j = oelm.node_j.coords + oelm.offset_j
            else:
                # move node at end j
                if idx+1 == len(self.internal_elems):
                    # Can't move that node! Split instead.
                    node = do_split(
                        ielm,
                        proportion_i,
                        self.internal_elems,
                        self.internal_nodes)
                else:
                    # Can move node.
                    # find the other LineElement connected to the node
                    oelm = self.internal_elems[idx+1]
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
                ielm,
                proportion_i,
                self.internal_elems,
                self.internal_nodes)

        return node, offset


@dataclass
@total_ordering
class LineElementSequence:
    """
    A LineElementSequence represents a collection
    of line elements connected in series.
    See figure:
    https://notability.com/n/0Nvu2Gqlt3gfwEcQE2~RKf

    After instantiating the object, a `generate` method needs to
    be called, to populate the internal components of the object.
    Attributes:
        node_i (int): primary node for end i
        node_j (int): primary node for end j
        ang: Parameter that controls the rotation of the
             section around the x-axis
        offset_i (list[float]): Components of the vector that starts
                                from the primary node i and goes to
                                the first internal node at the end i.
                                Expressed in the global coordinate system.
        offset_j (list[float]): Similarly for node j.
        section (Section): Section of the element.
        n_sub (int): Number of line elements between
                     the primary nodes node_i and node_j.
        model_as (dict): Either
                       {'type': 'elastic'}
                       or
                       {'type': 'fiber', 'n_x': n_x, 'n_y': n_y}
        geomTransf: {Linear, PDelta}
        placement (str): String flag that controls the
                         placement point of the element relative
                         to its section.
        end_dist (float): Distance between the start/end points and the
                          point along the clear length (without offsets) of
                          the LineElementSequence where it transitions from
                          the end segments to the middle segment, expressed
                          as a proportion of the sequence's clear length.
    """

    node_i: Node
    node_j: Node
    ang: float
    offset_i: np.ndarray
    offset_j: np.ndarray
    section: Section
    n_sub: int
    model_as: dict
    geomTransf: str
    placement: str
    end_dist: float
    metadata: dict

    def __post_init__(self):

        assert self.end_dist > 0.0, "end_dist must be > 0"

        p_i = self.node_i.coords + self.offset_i
        p_j = self.node_j.coords + self.offset_j

        self.length_clear = np.linalg.norm(p_j - p_i)
        # obtain offset from section (local system)
        if self.section.sec_type != 'utility':
            dz, dy = self.section.snap_points[self.placement]
            sec_offset_local = np.array([0.00, dy, dz])
        else:
            sec_offset_local = np.array([0., 0., 0.])
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

        # location of transition from start segment
        # to middle segment
        start_loc = p_i\
            + self.x_axis * self.end_dist * self.length_clear
        self.n_i = Node(start_loc)
        # location of transition from middle segment
        # to end segment
        end_loc = p_i\
            + self.x_axis * (1.0 - self.end_dist) * self.length_clear
        self.n_j = Node(end_loc)

        self.end_segment_i = None
        self.end_segment_j = None
        self.middle_segment = None

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

    def length_clear(self):
        """
        Returns the clear length of the sequence,
        (not counting the end offsets, but including
        the internal elements of the two end segments)
        """
        return self.length_clear

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
        for elm in self.internal_elems():
            if isinstance(elm, LineElement):
                elm.add_udl_glob(udl, ltype=ltype)

    def apply_self_weight_and_mass(self, multiplier: float):
        """
        Applies self-weight as a and distributes mass
        by lumping it at the nodes where the ends of the
        internal elements are connected.
        Args:
            multiplier: A parameter that is multiplied to the
                        automatically obtained self-weight and self-mass.
        """

        if self.section.sec_type == 'utility':
            return

        if multiplier == 0.:
            return

        cross_section_area = self.section.properties["A"]
        mass_per_length = cross_section_area * \
            self.section.material.density              # lb-s**2/in**2
        weight_per_length = mass_per_length * common.G_CONST  # lb/in
        mass_per_length *= multiplier
        weight_per_length *= multiplier
        self.add_udl_glob(
            np.array([0., 0., -weight_per_length]), ltype='self')
        total_mass_per_length = - \
            self.internal_line_elems()[0].get_udl_no_floor_glob()[
                2] / common.G_CONST
        for sub_elm in self.internal_elems():
            if isinstance(sub_elm, LineElement):
                mass = total_mass_per_length * \
                    sub_elm.length_clear() / 2.00  # lb-s**2/in
                sub_elm.node_i.mass += np.array([mass, mass, mass])
                sub_elm.node_j.mass += np.array([mass, mass, mass])

    def primary_nodes(self):
        return [self.node_i, self.node_j]

    def internal_nodes(self):
        result = []
        result.extend(self.end_segment_i.internal_nodes)
        result.extend(self.middle_segment.internal_nodes)
        result.extend(self.end_segment_j.internal_nodes)
        return result

    def internal_elems(self):
        result = []
        result.extend(self.end_segment_i.internal_elems)
        result.extend(self.middle_segment.internal_elems)
        result.extend(self.end_segment_j.internal_elems)
        return result

    def internal_line_elems(self):
        result = []
        for elm in self.internal_elems():
            if isinstance(elm, LineElement):
                result.append(elm)
        return result

    def __eq__(self, other):
        return (self.node_i == other.node_i and
                self.node_j == other.node_j)

    def __le__(self, other):
        return self.node_i <= other.node_i


@dataclass
@total_ordering
class LineElementSequence_Fixed(LineElementSequence):
    """
    A fixed LineElementSequence consists of a middle
    segment and two end segments, all having the same
    section, connected rigidly at the primary end nodes,
    with the specified rigid offsets.
    """

    def __post_init__(self):
        super().__post_init__()
        # middle segment
        camber = 0.00
        self.middle_segment = MiddleSegment(
            self.n_i,
            self.n_j,
            self.ang,
            self.section,
            self.model_as,
            self.geomTransf,
            self.length_clear,
            self.x_axis,
            self.y_axis,
            self.z_axis,
            self.n_sub,
            camber)

        # end segments
        self.end_segment_i = EndSegment_Fixed(
            self.node_i, self.n_i,
            self.offset_i,
            "i",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf)

        self.end_segment_j = EndSegment_Fixed(
            self.node_j, self.n_j,
            self.offset_j,
            "j",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf)


@dataclass
@total_ordering
class LineElementSequence_Pinned(LineElementSequence):
    """
    A pinned LineElementSequence consists of a middle
    segment that can have an initial imperfection (camber),
    and two end segments that have the same section with
    a moment release between them and the middle segment.
    The release only affects the two bending moments.
    """
    mat_fix: Material
    camber: float

    def __post_init__(self):
        super().__post_init__()

        # generate middle segment
        self.middle_segment = MiddleSegment(
            self.n_i,
            self.n_j,
            self.ang,
            self.section,
            self.model_as,
            self.geomTransf,
            self.length_clear,
            self.x_axis,
            self.y_axis,
            self.z_axis,
            self.n_sub,
            self.camber)

        # generate end segments
        self.end_segment_i = EndSegment_Pinned(
            self.node_i, self.n_i,
            self.offset_i,
            "i",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf,
            self.x_axis, self.y_axis,
            self.mat_fix)

        self.end_segment_j = EndSegment_Pinned(
            self.node_j, self.n_j,
            self.offset_j,
            "j",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf,
            self.x_axis, self.y_axis,
            self.mat_fix)


@dataclass
@total_ordering
class LineElementSequence_FixedPinned(LineElementSequence):
    """
    A FixedPinned LineElementSequence consists of a middle
    segment that can have an initial imperfection (camber),
    and two end segments of which one is pinned.
    The release only affects the two bending moments.
    """
    mat_fix: Material
    mat_release: Material
    camber: float

    def __post_init__(self):
        super().__post_init__()

        # generate middle segment
        self.middle_segment = MiddleSegment(
            self.n_i,
            self.n_j,
            self.ang,
            self.section,
            self.model_as,
            self.geomTransf,
            self.length_clear,
            self.x_axis,
            self.y_axis,
            self.z_axis,
            self.n_sub,
            self.camber)

        # generate end segment i (fixed)
        self.end_segment_i = EndSegment_Fixed(
            self.node_i, self.n_i,
            self.offset_i,
            "i",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf)

        # generate end segment j (pinned)
        self.end_segment_j = EndSegment_Pinned(
            self.node_j, self.n_j,
            self.offset_j,
            "j",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf,
            self.x_axis, self.y_axis,
            self.mat_fix)


@dataclass
@total_ordering
class LineElementSequence_RBS(LineElementSequence):
    """
    An RBS LineElementSequence consists of a middle segment
    and two RBS end segments, each connected rigidly to
    the primary nodes (with the specified rigid offset) and
    the middle segment. Each RBS end segment consists of two
    internal LineElements, one towards the end that has the initial
    section, and another towards the middle segment having the
    reduced section.
    """
    rbs_length: float
    rbs_reduction: float
    rbs_n_sub: int

    def __post_init__(self):
        """
        Generate a sequence representing a RBS beam.
        Args:
            rbs_distance (float): Where the reduction starts
                         relative to the sequence's clear length
                         (without the offsets)
            rbs_length (float): The length of the part where the
                       section is reduced.
            rbs_reduction (float): Reduction factor for the rbs part,
                          expressed as a proportion of the section's width.
        """
        super().__post_init__()
        assert self.rbs_length > 0.00, "RBS length must be > 0"

        # generate middle segment
        camber = 0.00
        self.middle_segment = MiddleSegment(
            self.n_i,
            self.n_j,
            self.ang,
            self.section,
            self.model_as,
            self.geomTransf,
            self.length_clear,
            self.x_axis,
            self.y_axis,
            self.z_axis,
            self.n_sub,
            camber)

        # generate end segments
        self.end_segment_i = EndSegment_RBS(
            self.node_i, self.n_i,
            self.offset_i,
            "i",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf,
            self.x_axis, self.rbs_length, self.rbs_reduction,
            self.rbs_n_sub)

        self.end_segment_j = EndSegment_RBS(
            self.node_j, self.n_j,
            self.offset_j,
            "j",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf,
            self.x_axis, self.rbs_length, self.rbs_reduction,
            self.rbs_n_sub)


@dataclass
@total_ordering
class LineElementSequence_RBS_j(LineElementSequence):
    """
    """
    rbs_length: float
    rbs_reduction: float
    rbs_n_sub: int

    def __post_init__(self):
        """
        Generate a sequence representing a RBS beam.
        Args:
            rbs_distance (float): Where the reduction starts
                         relative to the sequence's clear length
                         (without the offsets)
            rbs_length (float): The length of the part where the
                       section is reduced.
            rbs_reduction (float): Reduction factor for the rbs part,
                          expressed as a proportion of the section's width.
        """
        super().__post_init__()
        assert self.rbs_length > 0.00, "RBS length must be > 0"

        # generate middle segment
        camber = 0.00
        self.middle_segment = MiddleSegment(
            self.n_i,
            self.n_j,
            self.ang,
            self.section,
            self.model_as,
            self.geomTransf,
            self.length_clear,
            self.x_axis,
            self.y_axis,
            self.z_axis,
            self.n_sub,
            camber)

        # generate end segments
        self.end_segment_i = EndSegment_Fixed(
            self.node_i, self.n_i,
            self.offset_i,
            "i",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf)

        self.end_segment_j = EndSegment_RBS(
            self.node_j, self.n_j,
            self.offset_j,
            "j",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf,
            self.x_axis, self.rbs_length, self.rbs_reduction,
            self.rbs_n_sub)


@dataclass
@total_ordering
class LineElementSequence_IMK(LineElementSequence):
    """
    TODO ~ add docstring
    """
    mat_fix: Material

    def __post_init__(self):
        super().__post_init__()

        # generate middle segment
        self.middle_segment = MiddleSegment(
            self.n_i,
            self.n_j,
            self.ang,
            self.section,
            self.model_as,
            self.geomTransf,
            self.length_clear,
            self.x_axis,
            self.y_axis,
            self.z_axis,
            self.n_sub,
            0.00)

        # define IMK material based on element properties
        assert self.section.sec_type == 'W', \
            "Error: Only W sections can be used"

        # Young's modulus
        mat_e = self.section.material.parameters['E0'] / 1.e3
        # Yield stress
        mat_fy = self.section.material.parameters['Fy'] / 1.e3
        # Moment of inertia - strong axis - original section
        sec_ix = self.section.properties['Ix']
        # Section depth
        sec_d = self.section.properties['d']
        # Flange width
        sec_bf = self.section.properties['bf']
        # Flange and web thicknesses
        sec_tf = self.section.properties['tf']
        sec_tw = self.section.properties['tw']
        # Plastic modulus (unreduced)
        sec_zx = self.section.properties['Zx']
        lbry = self.metadata['Lb/ry']
        # Clear length
        elm_H = self.length_clear
        # Shear span - 0.5 * elm_H typically.
        elm_L = self.metadata['L/H'] * elm_H
        # RBS reduction factor
        rbs_factor = self.metadata['RBS_factor']
        assert rbs_factor <= 1.00, 'rbs_factor must be <= 1.00'
        # Floor composite action consideration
        consider_composite = self.metadata['composite action']

        if rbs_factor < 1.00:
            # RBS case

            # checks ~ acceptable range
            if not (20.00 < sec_d/sec_tw < 55.00):
                print('Warning: sec_d/sec_tw outside regression range')
                print(self.section, '\n')
            if not (20.00 < lbry < 80.00):
                print('Warning: Lb/ry outside regression range')
                print(self.section, '\n')
            if not (4.00 < (sec_bf/(2.*sec_tf)) < 8.00):
                print('Warning: bf/(2 tf) outside regression range')
                print(self.section, '\n')
            if not (2.5 < elm_L/sec_d < 7.0):
                print('Warning: L/d  outside regression range')
                print(self.section, '\n')
            if not (4.00 < sec_d < 36.00):
                print('Warning: Section d outside regression range')
                print(self.section, '\n')
            if not (35.00 < mat_fy < 65.00):
                print('Warning: Fy outside regression range')
                print(self.section, '\n')

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
            raise ValueError("Oops! Not implemented yet!")

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

        mat_IMK = Material(
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

        # generate end segments
        self.end_segment_i = EndSegment_IMK(
            self.node_i, self.n_i,
            self.offset_i,
            "i",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf,
            self.x_axis, self.y_axis,
            self.mat_fix, mat_IMK)

        self.end_segment_j = EndSegment_IMK(
            self.node_j, self.n_j,
            self.offset_j,
            "j",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf,
            -self.x_axis, -self.y_axis,
            self.mat_fix, mat_IMK)


@dataclass
@total_ordering
class LineElementSequence_Steel_W_PanelZone(LineElementSequence):
    """
    TODO - add docstring
    """

    rigid_util_section: Section
    mat_fix: Material
    beam_depth: float
    strain_hardening_ratio: float

    def __post_init__(self):
        super().__post_init__()

        # define nonlinear spring uniaxialMaterial
        if 'doubler plate thickness' in self.metadata['ends'].keys():
            doubler_thickness = \
                self.metadata['ends']['doubler plate thickness']
        else:
            raise ValueError('No doubler plate thickness specified')
        fy = self.section.material.parameters['Fy']
        dc = self.section.properties['d']
        bfc = self.section.properties['bf']
        tp = self.section.properties['tw'] + doubler_thickness
        tf = self.section.properties['tf']
        vy = 0.55 * fy * dc * tp
        g_mod = self.section.material.parameters['G']
        ke = 0.95 * g_mod * tp * dc
        kp = 0.95 * g_mod * bfc * tf**2 / self.beam_depth
        gamma_1 = vy / ke
        gamma_2 = 4.0 * gamma_1
        gamma_3 = 100. * gamma_1
        m1y = gamma_1 * ke * self.beam_depth
        m2y = m1y + kp * self.beam_depth * (gamma_2 - gamma_1)
        m3y = m2y + (self.strain_hardening_ratio * ke
                     * self.beam_depth) * (gamma_3 - gamma_2)
        spring_mat = Material(
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

        # generate end segment i (panel zone)
        self.end_segment_i = EndSegment_Steel_W_PanelZone(
            self.node_i, self.n_i,
            np.zeros(3).copy(), self.section, self.ang,
            self.rigid_util_section, self.x_axis, self.y_axis,
            self.mat_fix, spring_mat, self.beam_depth)

        # generate end segment j (fixed)
        self.end_segment_j = EndSegment_Fixed(
            self.node_j, self.n_j,
            self.offset_j,
            "j",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf)

        # generate middle segment
        self.middle_segment = MiddleSegment(
            self.n_i,
            self.n_j,
            self.ang,
            self.section,
            self.model_as,
            self.geomTransf,
            self.length_clear,
            self.x_axis,
            self.y_axis,
            self.z_axis,
            self.n_sub,
            camber=0.00)


@dataclass
@total_ordering
class LineElementSequence_Steel_W_PanelZone_IMK(LineElementSequence):
    """
    TODO - add docstring
    """

    rigid_util_section: Section
    mat_fix: Material
    beam_depth: float
    strain_hardening_ratio: float

    def __post_init__(self):
        super().__post_init__()

        # define nonlinear spring uniaxialMaterial
        if 'doubler plate thickness' in self.metadata['ends'].keys():
            doubler_thickness = \
                self.metadata['ends']['doubler plate thickness']
        else:
            raise ValueError('No doubler plate thickness specified')
        fy = self.section.material.parameters['Fy']
        dc = self.section.properties['d']
        bfc = self.section.properties['bf']
        tp = self.section.properties['tw'] + doubler_thickness
        tf = self.section.properties['tf']
        vy = 0.55 * fy * dc * tp
        g_mod = self.section.material.parameters['G']
        ke = 0.95 * g_mod * tp * dc
        kp = 0.95 * g_mod * bfc * tf**2 / self.beam_depth
        gamma_1 = vy / ke
        gamma_2 = 4.0 * gamma_1
        gamma_3 = 100. * gamma_1
        m1y = gamma_1 * ke * self.beam_depth
        m2y = m1y + kp * self.beam_depth * (gamma_2 - gamma_1)
        m3y = m2y + (self.strain_hardening_ratio * ke
                     * self.beam_depth) * (gamma_3 - gamma_2)
        spring_mat = Material(
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

        # define IMK material based on element properties
        assert self.section.sec_type == 'W', \
            "Error: Only W sections can be used"

        # Young's modulus
        mat_e = self.section.material.parameters['E0'] / 1.e3
        # Yield stress
        mat_fy = self.section.material.parameters['Fy'] / 1.e3
        # Moment of inertia - strong axis - original section
        sec_ix = self.section.properties['Ix']
        # Section depth
        sec_d = self.section.properties['d']
        # Flange width
        sec_bf = self.section.properties['bf']
        # Flange and web thicknesses
        sec_tf = self.section.properties['tf']
        sec_tw = self.section.properties['tw']
        # Plastic modulus (unreduced)
        sec_zx = self.section.properties['Zx']
        lbry = self.metadata['ends']['Lb/ry']
        pgpye = self.metadata['ends']['pgpye']
        # Clear length
        elm_H = self.length_clear

        # checks ~ acceptable range
        # TODO
        # if not (20.00 < sec_d/sec_tw < 55.00):
        #     print('Warning: sec_d/sec_tw outside regression range')
        #     print(self.section, '\n')
        # if not (20.00 < lbry < 80.00):
        #     print('Warning: Lb/ry outside regression range')
        #     print(self.section, '\n')
        # if not (4.00 < (sec_bf/(2.*sec_tf)) < 8.00):
        #     print('Warning: bf/(2 tf) outside regression range')
        #     print(self.section, '\n')
        # if not (2.5 < elm_L/sec_d < 7.0):
        #     print('Warning: L/d  outside regression range')
        #     print(self.section, '\n')
        # if not (4.00 < sec_d < 36.00):
        #     print('Warning: Section d outside regression range')
        #     print(self.section, '\n')
        # if not (35.00 < mat_fy < 65.00):
        #     print('Warning: Fy outside regression range')
        #     print(self.section, '\n')

        # calculate parameters
        theta_p = min(
            294. * (sec_d/sec_tw)**(-1.7) *
            lbry**(-0.7) *
            (1. - pgpye)**1.60,
            0.20
        )

        theta_pc = min(
            90. * (sec_d/sec_tw)**(-0.8) *
            lbry**(-0.80) *
            (1. - pgpye)**2.50,
            0.30)

        if pgpye <= 0.35:
            lamda = 25500. * (sec_d/sec_tw)**(-2.140) * \
                lbry**(-0.530) * \
                (1. - pgpye)**4.92
        else:
            lamda = 268000. * (sec_d/sec_tw)**(-2.30) * \
                lbry**(-1.30) * \
                (1. - pgpye)**1.19

        if pgpye <= 0.2:
            sec_my = 1.15/1.1 * sec_zx * mat_fy * \
                1.e3 * (1. - pgpye/2.00)
        else:
            sec_my = 1.15/1.1 * sec_zx * mat_fy * \
                1.e3 * (9./8.) * (1. - pgpye)

        mcmy = 12.5 * (sec_d/sec_tw)**(-0.20) * \
            lbry**(-0.40) * (1. - pgpye)**0.40
        if mcmy > 1.30:
            mcmy = 1.30
        elif mcmy < 1.00:
            mcmy = 1.00

        theta_u = 0.15
        residual_plus = 0.50 - 0.4 * pgpye
        residual_minus = 0.50 - 0.4 * pgpye
        theta_p_plus = theta_p
        theta_p_minus = theta_p
        theta_pc_plus = theta_pc
        theta_pc_minus = theta_pc
        d_plus = 1.00
        d_minus = 1.00
        mcmy_plus = mcmy
        mcmy_minus = mcmy
        my_plus = sec_my
        my_minus = -sec_my

        stiffness = 6.00 * mat_e * sec_ix / elm_H * 1e4
        beta_plus = (mcmy_plus - 1.) * my_plus / (theta_p_plus) / stiffness
        beta_minus = - (mcmy_minus - 1.) * my_minus \
            / (theta_p_minus) / stiffness

        mat_IMK = Material(
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

        self.end_segment_i = EndSegment_Steel_W_PanelZone_IMK(
            self.node_i, self.n_i,
            np.zeros(3).copy(), self.section, self.ang,
            self.rigid_util_section, self.x_axis, self.y_axis,
            self.mat_fix, spring_mat, mat_IMK, self.beam_depth)

        self.end_segment_j = EndSegment_IMK(
            self.node_j, self.n_j,
            self.offset_j,
            "j",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf,
            self.x_axis, self.y_axis,
            self.mat_fix, mat_IMK)

        # generate middle segment
        self.middle_segment = MiddleSegment(
            self.n_i,
            self.n_j,
            self.ang,
            self.section,
            self.model_as,
            self.geomTransf,
            self.length_clear,
            self.x_axis,
            self.y_axis,
            self.z_axis,
            self.n_sub,
            camber=0.00)


@dataclass
@total_ordering
class LineElementSequence_W_grav_sear_tab(LineElementSequence):
    """
    TODO ~ add docstring
    """
    mat_fix: Material

    def __post_init__(self):
        super().__post_init__()

        # generate middle segment
        self.middle_segment = MiddleSegment(
            self.n_i,
            self.n_j,
            self.ang,
            self.section,
            self.model_as,
            self.geomTransf,
            self.length_clear,
            self.x_axis,
            self.y_axis,
            self.z_axis,
            self.n_sub,
            0.00)

        # define IMK material based on element properties
        assert self.section.sec_type == 'W', \
            "Error: Only W sections can be used"

        mat_fy = self.section.material.parameters['Fy'] / 1.e3
        sec_zx = self.section.properties['Zx']
        # gap = self.metadata['gap']
        consider_composite = self.metadata['composite action']

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

        spring_mat = Material(
            'auto_pinching',
            'Pinching4',
            0.00,
            params)

        # generate end segments
        self.end_segment_i = EndSegment_W_grav_shear_tab(
            self.node_i, self.n_i,
            self.offset_i,
            "i",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf,
            self.x_axis, self.y_axis,
            self.mat_fix, spring_mat)

        self.end_segment_j = EndSegment_W_grav_shear_tab(
            self.node_j, self.n_j,
            self.offset_j,
            "j",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf,
            self.x_axis, self.y_axis,
            self.mat_fix, spring_mat)


@dataclass
class LineElementSequences:
    """
    This class is a collector for columns, and provides
    methods that perform operations using columns.
    """

    element_list: list[LineElementSequence] = field(default_factory=list)

    def add(self, elm: LineElementSequence) -> bool:
        """
        Add an element in the element collection,
        if it does not already exist
        """

        if elm not in self.element_list:
            self.element_list.append(elm)
            self.element_list.sort()
            return True
        else:
            return False

    def remove(self, elm: LineElementSequence):
        """
        Remove an element from the element collection,
        if it was there.
        """
        if elm in self.element_list:
            self.element_list.remove(elm)
            self.element_list.sort()

    def clear(self):
        """
        Removes all elements
        """
        self.element_list = []

    def __repr__(self):
        out = str(len(self.element_list)) + " elements\n"
        for elm in self.element_list:
            out += repr(elm) + "\n"
        return out

    def internal_elems(self):
        result = []
        for element in self.element_list:
            result.extend(element.internal_elems)
        return result
