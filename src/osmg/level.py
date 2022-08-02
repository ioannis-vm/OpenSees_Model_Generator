"""
Model Generator for OpenSees ~ level
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from .collections import Collection
from .collections import NodeCollection
if TYPE_CHECKING:
    from .model import Model
    from .component_assembly import ComponentAssembly


@dataclass
class Level:
    """
    Level Object
    Levels are part of a model and they contain primary nodes and
      component assemblies.
    Attributes:
        parent_model (Model)
        uid (int)
        elevation (float)
        nodes (NodeCollection)
        components (Collection)
    """
    parent_model: Model = field(repr=False)
    uid: int
    elevation: float
    nodes: NodeCollection = field(init=False, repr=False)
    components: Collection[
        int, ComponentAssembly] = field(init=False, repr=False)

    def __post_init__(self):
        self.nodes = NodeCollection(self)
        self.components = Collection(self)

    def __repr__(self):
        res = ''
        res += 'Level object\n'
        res += f'parent_model: {self.parent_model.name}\n'
        res += f'uid: {self.uid}\n'
        res += f'elevation: {self.elevation}\n'
        res += 'Nodes: \n'
        res += self.nodes.__srepr__() + '\n'
        res += 'Components: \n'
        res += self.components.__srepr__() + '\n'
        return res
