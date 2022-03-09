"""
Model Builder for OpenSeesPy ~ Selection module
"""

#   __                 UC Berkeley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   April 2021
#
# https://github.com/ioannis-vm/OpenSees_Model_Builder

from __future__ import annotations
from dataclasses import dataclass, field
from components import LineElementSequences
from node import Nodes
from components import LineElement
import numpy as np


@dataclass
class Selection:
    """
    This class enables the ability to select elements
    to modify them.

    """
    nodes: Nodes = field(default_factory=Nodes, repr=False)
    beams: LineElementSequences = field(
        default_factory=LineElementSequences, repr=False)
    columns: LineElementSequences = field(
        default_factory=LineElementSequences, repr=False)
    braces: LineElementSequences = field(
        default_factory=LineElementSequences, repr=False)
    line_elements: list[LineElement] = field(
        default_factory=list, repr=False)

    def clear(self):
        """
        Clears all selected elements.
        """
        self.nodes = Nodes()
        self.beams = LineElementSequences()
        self.columns = LineElementSequences()
        self.braces = LineElementSequences()
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
            line_element.add_udl_glob(udl, ltype='other')

    #############################################
    # Methods that return objects               #
    #############################################

    def list_of_line_element_sequences(self):
        """
        Returns all selected LineElementSequences.
        """
        return self.beams.element_list + \
            self.columns.element_list + self.braces.element_list

    def list_of_line_elements(self):
        sequences = self.list_of_line_element_sequences()
        result = []
        for sequence in sequences:
            for elm in sequence:
                if isinstance(elm, LineElement):
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
