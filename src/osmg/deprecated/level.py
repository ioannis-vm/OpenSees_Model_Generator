"""Model Generator for OpenSees ~ level."""

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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from osmg.osmg_collections import Collection, NodeCollection

if TYPE_CHECKING:
    from osmg.component_assemblies import ComponentAssembly


@dataclass
class Level:
    """
    Level Object.

    Levels are part of a model and they contain primary nodes and
    component assemblies.

    Attributes:
    ----------
        parent_model (Model)
        uid (int)
        elevation (float)
        nodes (NodeCollection)
        components (Collection)
    """

    uid: int
    elevation: float
    nodes: NodeCollection = field(init=False, repr=False)
    components: Collection[int, ComponentAssembly] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization."""
        self.nodes = NodeCollection(self)
        self.components = Collection(self)

    def __repr__(self) -> str:
        """
        Get string representation.

        Returns:
            str: The string representation of the object.
        """
        node_count = len(self.nodes)  # Assuming NodeCollection implements __len__
        component_count = len(
            self.components
        )  # Assuming Collection implements __len__

        res = (
            f'Level Object\n'
            f'  UID: {self.uid}\n'
            f'  Elevation: {self.elevation} units\n'
            f'  Number of Nodes: {node_count}\n'
            f'  Number of Components: {component_count}\n'
        )

        # Include a preview of nodes and components if necessary
        if node_count > 0:
            res += f"  Nodes: {self.nodes[:5]}{'...' if node_count > 5 else ''}\n"
        if component_count > 0:
            res += f"  Components: {self.components[:5]}{'...' if component_count > 5 else ''}\n"

        return res
