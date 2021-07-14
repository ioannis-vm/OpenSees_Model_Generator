"""
Building Modeler for OpenSeesPy ~ Components
"""

#   __                 UC Berkeley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   April 2021
#
# https://github.com/ioannis-vm/OpenSeesPy_Building_Modeler

from typing import Optional
from dataclasses import dataclass, field
from functools import total_ordering
from itertools import count
import numpy as np
from grids import GridLine
from utility import common
from utility import transformations
from utility import mesher
from utility import mesher_section_gen

_ids = count(0)


@dataclass
@total_ordering
class Node:
    """
    Node object.
    Attributes:
        uniq_id (int): unique identifier
        coords (np.ndarray): Coordinates of the location of the node
        restraint_type (str): Can be either "free", "pinned", or "fixed".
                       It can also be "parent" or "internal", but
                       this is only specified for nodes that are made
                       automatically.
        mass (np.ndarray): Mass with respect to the global coordinate system
                           (shape = 3 for all nodes except parent nodes, where
                            inertia terms are also present, and shape = 6).
        load (np.ndarray): Load with respect to the global coordinate system.
        load_fl (np.ndarray): similar to load, coming from the floors.
        tributary_area: This attribute holds the results of tributary area
                        analysis done inside the `preprocess` method of the
                        `building` objects, and is used to store the floor
                        area that corresponds to that node (if beams  with
                        offsets are connected to it)
        column_above, column_below, beams, braces_above, braces_below, ...:
            Pointers to connected elements (only for primary nodes)
    """

    coords: np.ndarray
    restraint_type: str = field(default="free")
    mass: np.ndarray = field(
        default_factory=lambda: np.zeros(shape=3), repr=False)
    load: np.ndarray = field(
        default_factory=lambda: np.zeros(shape=6), repr=False)
    load_fl: np.ndarray = field(
        default_factory=lambda: np.zeros(shape=6), repr=False)
    tributary_area: float = field(default=0.00, repr=False)
    column_above: Optional['LineElementSequence'] = field(
        default=None, repr=False)
    column_below: Optional['LineElementSequence'] = field(
        default=None, repr=False)
    beams: list['LineElementSequence'] = field(
        default_factory=list, repr=False)

    def __post_init__(self):
        self.uniq_id = next(_ids)

    def __eq__(self, other):
        """
        For nodes, a fudge factor is used to
        assess equality.
        """
        p0 = np.array(self.coords)
        p1 = np.array(other.coords)
        return np.linalg.norm(p0 - p1) < common.EPSILON

    def __le__(self, other):
        d_self = self.coords[1] * common.ALPHA + self.coords[0]
        d_other = other.coords[1] * common.ALPHA + other.coords[0]
        return d_self <= d_other

    def load_total(self):
        """
        Returns the total load applied on the node,
        by summing up the floor's contribution to the
        generic component.
        """
        return self.load + self.load_fl


@dataclass
class Nodes:
    """
    This class is a collector for the nodes, and provides
    methods that perform operations using nodes.
    """

    node_list: list[Node] = field(default_factory=list)

    def add(self, node: Node):
        """
        Add a node in the nodes collection,
        if it does not already exist
        """
        if node not in self.node_list:
            self.node_list.append(node)
        else:
            raise ValueError('Node already exists: '
                             + repr(node))
        self.node_list.sort()

    def remove(self, node: Node):
        """
        Remove a node from the nodes collection,
        if it was there.
        """
        if node in self.node_list:
            self.node_list.remove(node)
        self.node_list.sort()

    def clear(self):
        """
        Removes all nodes
        """
        self.node_list = []

    def __repr__(self):
        out = str(len(self.node_list)) + " nodes\n"
        for node in self.node_list:
            out += repr(node) + "\n"
        return out


@dataclass
class Material:
    """
    Material object.
    Attributes:
        uniq_id (int): unique identifier
        name (str): Name of the material
        ops_material (str): Name of the material model to use in OpenSees
        density (float): Mass per unit volume of the material
        parameters (dict): Parameters needed to define the material in OpenSees
                           These depend on the meterial model specified.
    """
    name: str
    ops_material: str
    density: float  # mass per unit volume, specified in lb-s**2/in**4
    parameters: dict

    def __post_init__(self):
        self.uniq_id = next(_ids)


@dataclass
class Materials:
    """
    This class is a collector for materials.
    """

    material_list: list[Material] = field(default_factory=list)
    active: Optional[Material] = field(default=None)

    def __post_init__(self):
        """
        Add some default materials used
        to model the connectivity of elements.
        """
        self.material_list.append(Material(
            'fix',
            'Elastic',
            0.00,
            {
                'E': 1.0e12
            })
        )
        self.material_list.append(Material(
            'release',
            'Elastic',
            0.00,
            {
                'E': 1.0e-12
            })
        )

    def add(self, material: Material):
        """
        Add a material in the materials collection,
        if it does not already exist
        """
        if material not in self.material_list:
            self.material_list.append(material)
        else:
            raise ValueError('Material already exists: '
                             + repr(material))

    def set_active(self, name: str):
        """
        Assigns the active material.
        """
        self.active = None
        found = False
        for material in self.material_list:
            if material.name == name:
                self.active = material
                found = True
                break
        if found is False:
            raise ValueError("Material " + name + " does not exist")

    def retrieve(self, name: str):
        """
        Returns the specified material.
        """
        result = None
        for material in self.material_list:
            if material.name == name:
                result = material
                break
        return result

    def enable_Steel02(self):
        """
        Adds a predefined A992Fy50 steel material modeled
        using Steel02.
        """
        # units: lb, in
        self.add(Material('steel',
                          'Steel02',
                          0.0007342054137099255,
                          {
                              'Fy': 50000,
                              'E0': 29000000,
                              'G':   11153846.15,
                              'b': 0.01,
                              'params': [18.0, 0.925, 0.15],
                              'a1': 0.00,
                              'a2': 1.00,
                              'a3': 0.00,
                              'a4': 1.00,
                              'sigInit': 0.00
                          })
                 )

    def __repr__(self):
        out = "Defined sections: " + str(len(self.material_list)) + "\n"
        for material in self.material_list:
            out += repr(material) + "\n"
        return out


@dataclass
class Section:
    """
    Section object.
    The axes are defined in the same way as they are
    defined in OpenSees. The colors assigned to
    the axes for plotting follow the
    AutoCAD convention.

            y(green)
            ^         x(red)
            :       .
            :     .
            :   .
           ===
            | -------> z (blue)
           ===
    Attributes:
        uniq_id (int): unique identifier
        sec_type (str): Flag representing the type of section
                  (e.g. W -> steel W section)
        name (str): Unique name for the section
        material (Material): Material of the section
        snap_points (dict): Dictionary containing the local
                    coordinates of a set of characetristic points.
                    These points are:
                    'centroid', 'top_center', 'top_left', 'top_right',
                    'center_left', 'center_right', 'bottom_center',
                    'bottom_left', bottom_right'
        mesh (mesher.Mesh): Mesh object defining the geometry
                            of the section
        properties (dict): Dictionary with geometric properties
                           needed for structural analysis.
                           These are:
                           A, Ix, Iy, J
    """
    sec_type: str
    name: str
    material: Material = field(repr=False)
    snap_points: Optional[dict] = field(default=None, repr=False)
    mesh: Optional[mesher.Mesh] = field(default=None, repr=False)
    properties: Optional[dict] = field(default=None, repr=False)

    def __post_init__(self):
        self.uniq_id = next(_ids)

    def __eq__(self, other):
        return (self.name == other.name)

    def subdivide_section(self, n_x=10, n_y=25, plot=False):
        """
        Used to define the fibers of fiber sections.
        Args:
            n_x (int): Number of spatial partitions in the x direction
            n_y (int): Number of spatial partitions in the y direction
            plot (bool): Plots the resulting polygons for debugging
        Returns:
            pieces (list[shapely_Polygon]): shapely_Polygon
                   objects that represent single fibers.
        """
        return mesher.subdivide_polygon(
            self.mesh.halfedges, n_x=n_x, n_y=n_y, plot=plot)

    # def retrieve_offset(self, placement: str):
    #     """
    #     Obtain the necessary offset in the y-z plane
    #     (local system)
    #     such that the element of that section has
    #     the specified placement point.
    #     The offset is expressed as the vector that moves
    #     from the placement point to the centroid.
    #     Args:
    #         placement (str): Can be one of:
    #             'centroid', 'top_center', 'top_left', 'top_right',
    #             'center_left', 'center_right', 'bottom_center',
    #             'bottom_left', 'bottom_right'
    #     """
    #     bbox = self.mesh.bounding_box()
    #     z_min, y_min, z_max, y_max = bbox.flatten()
    #     assert placement in ['centroid',
    #                          'top_center',
    #                          'top_left',
    #                          'top_right',
    #                          'center_left',
    #                          'center_right',
    #                          'bottom_center',
    #                          'bottom_left',
    #                          'bottom_right'], \
    #         "Invalid placement"
    #     if placement == 'centroid':
    #         return - np.array([0., 0.])
    #     elif placement == 'top_center':
    #         return - np.array([0., y_max])
    #     elif placement == 'top_left':
    #         return - np.array([z_min, y_max])
    #     elif placement == 'top_right':
    #         return - np.array([z_max, y_max])
    #     elif placement == 'center_left':
    #         return - np.array([z_min, 0.])
    #     elif placement == 'center_right':
    #         return - np.array([z_max, 0.])
    #     elif placement == 'bottom_center':
    #         return - np.array([0., y_min])
    #     elif placement == 'bottom_left':
    #         return - np.array([z_min, y_min])
    #     elif placement == 'bottom_right':
    #         return - np.array([z_max, y_min])

    def rbs(self, reduction_factor) -> 'Section':
        """
        Given a reduction factor
        expressed as a proportion of the section's width,
        the method returns a reduced section.
        Only works for W sections.
        """
        if self.sec_type != 'W':
            raise ValueError("Only W sections are supported for RBS")
        name = self.name + '_reduced'
        b = self.properties['bf']
        h = self.properties['d']
        tw = self.properties['tw']
        tf = self.properties['tf']
        t = self.properties['T']
        b_red = b * reduction_factor
        properties = dict(self.properties)
        properties['bf'] = b_red
        mesh = mesher_section_gen.w_mesh(b_red, h, t, tw, tf)
        section = Section(
            'W', name,
            self.material,
            mesh, properties)
        return section


@dataclass
class Sections:
    """
    This class is a collector for sections.
    """

    section_list: list[Section] = field(default_factory=list)
    active: Optional[Section] = field(default=None, repr=False)

    def add(self, section: Section):
        """
        Add a section in the section collection,
        if it does not already exist
        """
        if section not in self.section_list:
            self.section_list.append(section)
        else:
            raise ValueError('Section already exists: '
                             + repr(section))

    def set_active(self, name: str):
        """
        Sets the active section.
        Any elements defined while this section is active
        will have that section.
        Args:
            name (str): Name of the previously defined
                 section to set as active.
        """
        self.active = None
        found = False
        for section in self.section_list:
            if section.name == name:
                self.active = section
                found = True
        if found is False:
            raise ValueError("Section " + name + " does not exist")

    def __repr__(self):
        out = "Defined sections: " + str(len(self.section_list)) + "\n"
        for section in self.section_list:
            out += repr(section) + "\n"
        return out

    ####################
    # Shape generators #
    ####################

    def generate_W(self,
                   name: str,
                   material: Material,
                   properties: dict):
        """
        Generate a W section with specified parameters
        and add it to the sections list.
        """
        b = properties['bf']
        h = properties['d']
        tw = properties['tw']
        tf = properties['tf']
        t = properties['T']
        mesh = mesher_section_gen.w_mesh(b, h, t, tw, tf)
        bbox = mesh.bounding_box()
        z_min, y_min, z_max, y_max = bbox.flatten()
        snap_points = {
            'centroid': np.array([0., 0.]),
            'top_center': np.array([0., -y_max]),
            'top_left': np.array([-z_min, -y_max]),
            'top_right': np.array([-z_max, -y_max]),
            'center_left': np.array([-z_min, 0.]),
            'center_right': np.array([-z_max, 0.]),
            'bottom_center': np.array([0., -y_min]),
            'bottom_left': np.array([-z_min, -y_min]),
            'bottom_right': np.array([-z_max, -y_min])
        }
        section = Section('W', name, material,
                          snap_points,  mesh, properties)
        self.add(section)

    def generate_HSS(self,
                     name: str,
                     material: Material,
                     properties: dict):
        """
        Generate a HSS with specified parameters
        and add it to the sections list.
        """
        # use the name to assess whether it's a rectangular
        # or circular section
        xs = name.count('X')
        if xs == 2:
            # it's a rectangular section
            ht = properties['Ht']
            b = properties['B']
            t = properties['tdes']
            mesh = mesher_section_gen.HSS_rect_mesh(ht, b, t)
            bbox = mesh.bounding_box()
            z_min, y_min, z_max, y_max = bbox.flatten()
            snap_points = {
                'centroid': np.array([0., 0.]),
                'top_center': np.array([0., -y_max]),
                'top_left': np.array([-z_min, -y_max]),
                'top_right': np.array([-z_max, -y_max]),
                'center_left': np.array([-z_min, 0.]),
                'center_right': np.array([-z_max, 0.]),
                'bottom_center': np.array([0., -y_min]),
                'bottom_left': np.array([-z_min, -y_min]),
                'bottom_right': np.array([-z_max, -y_min])
            }
            section = Section('HSS', name, material,
                              snap_points, mesh, properties)
            self.add(section)
        elif xs == 1:
            # it's a circular section
            od = properties['OD']
            tdes = properties['tdes']
            n_pts = 25
            mesh = mesher_section_gen.HSS_circ_mesh(od, tdes, n_pts)
            bbox = mesh.bounding_box()
            z_min, y_min, z_max, y_max = bbox.flatten()
            snap_points = {
                'centroid': np.array([0., 0.]),
                'top_center': np.array([0., -y_max]),
                'top_left': np.array([-z_min, -y_max]),
                'top_right': np.array([-z_max, -y_max]),
                'center_left': np.array([-z_min, 0.]),
                'center_right': np.array([-z_max, 0.]),
                'bottom_center': np.array([0., -y_min]),
                'bottom_left': np.array([-z_min, -y_min]),
                'bottom_right': np.array([-z_max, -y_min])
            }
            section = Section('HSS', name, material,
                              snap_points, mesh, properties)
            self.add(section)
        else:
            raise ValueError("This should never happen...")

    def generate_rect(self,
                      name: str,
                      material: Material,
                      properties: dict):
        """
        Generate a rectangular section with specified
        parameters and add it to the sections list.
        """
        b = properties['b']
        h = properties['h']
        mesh = mesher_section_gen.rect_mesh(b, h)
        bbox = mesh.bounding_box()
        z_min, y_min, z_max, y_max = bbox.flatten()
        snap_points = {
            'centroid': np.array([0., 0.]),
            'top_center': np.array([0., -y_max]),
            'top_left': np.array([-z_min, -y_max]),
            'top_right': np.array([-z_max, -y_max]),
            'center_left': np.array([-z_min, 0.]),
            'center_right': np.array([-z_max, 0.]),
            'bottom_center': np.array([0., -y_min]),
            'bottom_left': np.array([-z_min, -y_min]),
            'bottom_right': np.array([-z_max, -y_min])
        }
        section = Section('rect', name, material,
                          snap_points, mesh, properties)
        self.add(section)
        temp = mesh.geometric_properties()
        properties['A'] = temp['area']
        properties['Ix'] = temp['inertia']['ixx']
        properties['Iy'] = temp['inertia']['iyy']
        properties['J'] = h * b**3 *\
            (16./3. - 3.36 * b/h * (1 - b**4/(12.*h**4)))


@dataclass
class EndRelease:
    """
    This class is used to simulate end-releases.
    """

    node_i: Node
    node_j: Node
    free_dofs: list[int]
    x_vec: np.ndarray
    y_vec: np.ndarray
    mat_fix: Material
    mat_release: Material

    def __post_init__(self):
        self.uniq_id = next(_ids)


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
        n_p (int): Number of integration points
        model_as (dict): Either
                       {'type': 'elastic'}
                       or
                       {'type': 'fiber', 'n_x': n_x, 'n_y': n_y}
        geomTransf: {Linear, PDelta}
        internal_pt_i (np.ndarray): Coordinates of the internal point i
        internal_pt_j (np.ndarray): Similarly for node j
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

    def __post_init__(self):
        self.uniq_id = next(_ids)
        self.tributary_area = 0.00
        # local axes with respect to the global coord system
        self.internal_pt_i = self.node_i.coords + self.offset_i
        self.internal_pt_j = self.node_j.coords + self.offset_j
        self.x_axis, self.y_axis, self.z_axis = \
            transformations.local_axes_from_points_and_angle(
                self.internal_pt_i, self.internal_pt_j, self.ang)
        self.len_proportion = self.length_clear() / self.len_parent
        if self.len_proportion > 0.75:
            n_p = 4
        elif self.len_proportion > 0.50:
            n_p = 3
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
            self.ang, self.offset_i, np.zeros(3),
            self.section, self.len_parent, self.model_as,
            self.geomTransf, self.udl_self, self.udl_fl,
            self.udl_other)
        piece_j = LineElement(
            split_node, self.node_j,
            self.ang, np.zeros(3), self.offset_j,
            self.section, self.len_parent, self.model_as,
            self.geomTransf, self.udl_self, self.udl_fl,
            self.udl_other)
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
                    self.offset, np.zeros(3),
                    self.section, self.len_parent,
                    self.model_as, self.geomTransf))
        elif self.end == 'j':
            self.internal_elems.append(
                LineElement(
                    self.n_internal, self.n_external, self.ang,
                    np.zeros(3), self.offset,
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
        mat_release (Material): Linear elastic material with a
                                very low stiffness. See the Material class.

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
    mat_release: Material

    def __post_init__(self):
        super().__post_init__()
        self.internal_pt = self.n_external.coords + self.offset
        self.internal_nodes.append(
            self.n_internal)
        n_release = Node(self.n_internal.coords)
        self.internal_nodes.append(n_release)
        self.internal_elems.append(
            EndRelease(self.n_internal,
                       n_release,
                       [5, 6],
                       self.x_axis,
                       self.y_axis,
                       self.mat_fix,
                       self.mat_release))
        if self.end == 'i':
            self.internal_elems.append(
                LineElement(
                    self.n_external, n_release, self.ang,
                    self.offset, np.zeros(3),
                    self.section, self.len_parent,
                    self.model_as, self.geomTransf))
        elif self.end == 'j':
            self.internal_elems.append(
                LineElement(
                    n_release, self.n_external, self.ang,
                    np.zeros(3), self.offset,
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

    def __post_init__(self):
        super().__post_init__()
        self.internal_pt = self.n_external.coords + self.offset

        rbs_sec = self.section.rbs(self.rbs_reduction)

        if self.end == 'i':
            n_RBS_len = Node(self.n_internal.coords -
                             self.x_axis * self.rbs_length)
            self.internal_nodes.append(n_RBS_len)
            self.internal_nodes.append(self.n_internal)
            self.internal_elems.append(
                LineElement(
                    self.n_external, n_RBS_len, self.ang,
                    self.offset, np.zeros(3),
                    self.section, self.len_parent,
                    self.model_as, self.geomTransf))
            self.internal_elems.append(
                LineElement(
                    n_RBS_len, self.n_internal, self.ang,
                    np.zeros(3), np.zeros(3),
                    rbs_sec, self.len_parent,
                    self.model_as, self.geomTransf))

        elif self.end == 'j':
            n_RBS_len = Node(self.n_internal.coords +
                             self.x_axis * self.rbs_length)
            self.internal_nodes.append(self.n_internal)
            self.internal_nodes.append(n_RBS_len)
            self.internal_elems.append(
                LineElement(
                    self.n_internal, n_RBS_len, self.ang,
                    np.zeros(3), np.zeros(3),
                    rbs_sec, self.len_parent,
                    self.model_as, self.geomTransf))
            self.internal_elems.append(
                LineElement(
                    n_RBS_len, self.n_external, self.ang,
                    np.zeros(3), self.offset,
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
                    np.zeros(3), np.zeros(3),
                    self.section, self.len_parent,
                    self.model_as, self.geomTransf))

    def crosses_point(self, pt: np.ndarray) -> bool:
        line = GridLine('', self.n_i.coords[0:2], self.n_j.coords[0:2])
        return line.intersects_pt(pt)

    def connect(self, pt: np.ndarray, elev: float) \
            -> tuple[Node, np.ndarray]:
        """
        Perform a split or move internal nodes to accomodate
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
    be called, to populte the internal components of the object.
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
                          point along the clear length (wihtout offsets) of
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

    def __post_init__(self):

        assert self.end_dist > 0.0, "end_dist must be > 0"

        p_i = self.node_i.coords + self.offset_i
        p_j = self.node_j.coords + self.offset_j

        self.length_clear = np.linalg.norm(p_j - p_i)
        # obtain offset from section (local system)
        dz, dy = self.section.snap_points[self.placement]
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
        Used to easiliy retrieve the required
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
            self.internal_elems()[1].get_udl_no_floor_glob()[
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


@dataclass(eq=False, order=False)
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


@dataclass(eq=False, order=False)
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

        # generate end segments
        self.end_segment_i = EndSegment_Pinned(
            self.node_i, self.n_i,
            self.offset_i,
            "i",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf,
            self.x_axis, self.y_axis,
            self.mat_fix, self.mat_release)

        self.end_segment_j = EndSegment_Pinned(
            self.node_j, self.n_j,
            self.offset_j,
            "j",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf,
            self.x_axis, self.y_axis,
            self.mat_fix, self.mat_release)


@dataclass(eq=False, order=False)
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
            self.x_axis, self.rbs_length, self.rbs_reduction)

        self.end_segment_j = EndSegment_RBS(
            self.node_j, self.n_j,
            self.offset_j,
            "j",
            self.length_clear,
            self.ang, self.section,
            self.model_as, self.geomTransf,
            self.x_axis, self.rbs_length, self.rbs_reduction)


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
