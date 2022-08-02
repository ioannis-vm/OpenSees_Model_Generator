"""
Model Generator for OpenSees ~ visibility
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from dataclasses import dataclass, field


@dataclass
class ElementVisibility:
    """
    Element visibility object.
    Controls whether an element is displayed in the plots
    and whether it is defined in OpenSees or not
    """
    hidden_when_extruded: bool = field(default=False)
    hidden_at_line_plots: bool = field(default=False)
    skip_opensees_definition: bool = field(default=False)


@dataclass
class NodeVisibility:
    """
    Node visibility object.
    Controls whether a node is displayed in the plots
    """
    connected_to_zerolength: bool = field(default=False)
