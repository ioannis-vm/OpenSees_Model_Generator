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
from .collections import ComponentCollection
from .collections import NodeCollection
from .component_assembly import ComponentAssembly
if TYPE_CHECKING:
    from .model import Model

# pylint: disable=unsubscriptable-object
# pylint: disable=invalid-name


@dataclass
class Level:
    """
    Level Object
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
    components: ComponentCollection = field(init=False, repr=False)

    def __post_init__(self):
        self.nodes = NodeCollection(self)
        self.components = ComponentCollection(self)

        

    # def look_for_node(self, x_coord: float, y_coord: float):
    #     """
    #     Returns the node that occupies a given point
    #     at the current level, if it exists
    #     """
    #     candidate_pt = np.array([x_coord, y_coord,
    #                              self.elevation])
    #     for other_node in self.nodes_primary.registry.values():
    #         other_pt = other_node.coords
    #         if np.linalg.norm(candidate_pt - other_pt) < common.EPSILON:
    #             return other_node
    #     return None

    # def look_for_beam(self, x_coord: float, y_coord: float):
    #     """
    #     Returns a beam if the path of its middle_segment
    #     crosses the given point.
    #     """
    #     candidate_pt = np.array([x_coord, y_coord])
    #     for beam in self.beams.registry.values():
    #         if beam.middle_segment.crosses_point(candidate_pt):
    #             return beam
    #     return None

    # def assign_surface_load(self,
    #                         load_per_area: float):
    #     self.surface_load = load_per_area

    # def assign_surface_load_massless(self,
    #                                  load_per_area: float):
    #     self.surface_load_massless = load_per_area

    # def list_of_primary_nodes(self):
    #     return self.nodes_primary.registry.values()

    # def list_of_all_nodes(self):
    #     """
    #     Returns a list containing all the nodes
    #     of that level *except* the parent node.
    #     """
    #     primary = list(self.nodes_primary.registry.values())
    #     internal = []
    #     for col in self.columns.registry.values():
    #         internal.extend(col.internal_nodes())
    #     for bm in self.beams.registry.values():
    #         internal.extend(bm.internal_nodes())
    #     result = [i for i in primary + internal if i]
    #     # (to remove Nones if they exist)
    #     return result

    # def list_of_line_elems(self):
    #     result = []
    #     for elm in self.beams.registry.values() + \
    #             self.columns.registry.values() + \
    #             self.braces.registry.values():
    #         if isinstance(elm, elasticBeamColumnElement):
    #             result.append(elm)
    #     return result

    # def list_of_steel_W_panel_zones(self):
    #     cols = self.columns.registry.values()
    #     pzs = []
    #     for col in cols:
    #         if isinstance(col, ComponentAssembly_Steel_W_PanelZone):
    #             pzs.append(col.end_segment_i)
    #     return pzs


