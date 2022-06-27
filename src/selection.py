"""
Model Generator for OpenSees ~ selection
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generato

from __future__ import annotations
from dataclasses import dataclass, field
from components import ComponentAssemblies
from node import Nodes
from components import BeamColumnElement
import numpy as np


@dataclass
class Selection:
    """
    This class enables the ability to select elements
    to modify them.

    """
    nodes: Nodes = field(default_factory=Nodes, repr=False)
    beams: ComponentAssemblies = field(
        default_factory=ComponentAssemblies, repr=False)
    columns: ComponentAssemblies = field(
        default_factory=ComponentAssemblies, repr=False)
    braces: ComponentAssemblies = field(
        default_factory=ComponentAssemblies, repr=False)
    line_elements: list[BeamColumnElement] = field(
        default_factory=list, repr=False)

    def clear(self):
        """
        Clears all selected elements.
        """
        self.nodes = Nodes()
        self.beams = ComponentAssemblies()
        self.columns = ComponentAssemblies()
        self.braces = ComponentAssemblies()
        self.line_elements = []

    #############################################
    # Methods that modify selected elements     #
    #############################################
    def add_UDL(self, udl: np.ndarray):
        """
        Adds the specified UDL to the selected
        line elements.
        """
        for line_element in self.line_elements:
            line_element.udl.add_glob(udl, ltype='other_load')

    #############################################
    # Methods that return objects               #
    #############################################

    def list_of_line_element_sequences(self):
        """
        Returns all selected ComponentAssemblies.
        """
        return list(self.beams.registry.values()) + \
            list(self.columns.registry.values()) + \
            list(self.braces.registry.values())

    def list_of_line_elements(self):
        sequences = self.list_of_line_element_sequences()
        result = []
        for sequence in sequences:
            for elm in sequence.internal_line_elems():
                result.append(elm)
        result.extend(self.line_elements)
        return result

    def list_of_primary_nodes(self):
        """
        Returns a list of unique primary nodes on which all the
        selected elements are connected to.
        """
        gather = []
        for elem in self.list_of_line_element_sequences():
            gather.extend(elem.primary_nodes())
        # remove duplicates
        result = []
        return [result.append(x) for x in gather if x not in gather]

    def list_of_internal_nodes(self):
        """
        Returns a list of all secondary nodes that exist
        in the selected elements.
        """
        result = []
        for elem in self.list_of_line_element_sequences():
            result.extend(elem.internal_nodes())
        return result
