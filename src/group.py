"""
Model Builder for OpenSeesPy ~ group module
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
from functools import total_ordering


# pylint: disable=unsubscriptable-object
# pylint: disable=invalid-name


@dataclass
@total_ordering
class Group:
    """
    This class is be used to group together
    elements of any kind.
    """

    name: str
    elements: list = field(init=False, repr=False)

    def __post_init__(self):
        self.elements = []

    def __eq__(self, other):
        return self.name == other.name

    def __le__(self, other):
        return self.name <= other.name

    def add(self, element):
        """
        Add an element in the group,
        if it is not already in
        """
        if element not in self.elements:
            self.elements.append(element)

    def remove(self, element):
        """
        Remove something from the group
        """
        self.elements.remove(element)

    def __repr__(self):
        return(
            "Group(name=" + self.name + "): "
            + str(len(self.elements)) + " elements.")


@dataclass
class Groups:
    """
    Stores the  groups of a building.
    No two groups can have the same name.
    Elements can belong in multiple groups.
    """

    group_list: list[Group] = field(default_factory=list)
    active:     list[Group] = field(default_factory=list)

    def add(self, grp: Group):
        """
        Adds a new element group

        Parameters:
            grp(Group): the element group to add
        """
        # Verify element group name is unique
        if grp in self.group_list:
            raise ValueError('Group name already exists: ' + repr(grp))
        # Append the new element group in the list
        self.group_list.append(grp)
        # Sort the element groups in ascending order (name-wise)
        self.group_list.sort()

    def retrieve_by_name(self, name: str) -> Group:
        """
        Returns a variable pointing to the group that has the
        given name.
        Args:
            name (str): Name of the group to retrieve
        Returns:
            group (Group)
        """
        for grp in self.group_list:
            if grp.name == name:
                return grp
        raise ValueError("Group " + name + " does not exist")

    def set_active(self, names: list[str]):
        """
        Specifies the active groups(one or more).
        Adding any element to the building will also
        add that element to the active groups.
        The active groups can also be set to an empty list.
        In that case, new elements will not be added toa any groups.
        Args:
            names (list[str]): Names of groups to set as active
        """
        self.active = []
        for name in names:
            grp = self.retrieve_by_name(name)
            self.active.append(grp)

    def add_element(self, element):
        """
        Adds an element to all active groups.
        """
        for grp in self.active:
            grp.add(element)

    def __repr__(self):
        out = "The building has " + \
            str(len(self.group_list)) + " groups\n"
        for grp in self.group_list:
            out += repr(grp) + "\n"
        return out
