"""
Model Generator for OpenSees ~ level
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
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from .obj_collections import Collection
from .obj_collections import NodeCollection

if TYPE_CHECKING:
    from .model import Model
    from .component_assembly import ComponentAssembly


@dataclass
class Level:
    """
    Level Object. Levels are part of a model and they contain primary
      nodes and component assemblies.

    Attributes:
        parent_model (Model)
        uid (int)
        elevation (float)
        nodes (NodeCollection)
        components (Collection)

    Example:
        >>> from osmg.model import Model
        >>> model = Model(name='example_model')
        >>> level = Level(parent_model=model, uid=1, elevation=0.0)
        >>> level.parent_model.name
        'example_model'
        >>> level.uid
        1
        >>> level.elevation
        0.0
        >>> type(level.nodes)
        <class 'osmg.obj_collections.NodeCollection'>
        >>> type(level.components)
        <class 'osmg.obj_collections.Collection'>
    """

    parent_model: Model = field(repr=False)
    uid: int
    elevation: float
    nodes: NodeCollection = field(init=False, repr=False)
    components: Collection[int, ComponentAssembly] = field(
        init=False, repr=False
    )

    def __post_init__(self):
        self.nodes = NodeCollection(self)
        self.components = Collection(self)

    def __repr__(self):
        res = ""
        res += "Level object\n"
        res += f"parent_model: {self.parent_model.name}\n"
        res += f"uid: {self.uid}\n"
        res += f"elevation: {self.elevation}\n"
        res += "Nodes: \n"
        res += self.nodes.__srepr__() + "\n"
        res += "Components: \n"
        res += self.components.__srepr__() + "\n"
        return res
