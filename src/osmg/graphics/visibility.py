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

from typing import TYPE_CHECKING
from dataclasses import dataclass, field


@dataclass
class ElementVisibility:
    """

    """
    hidden_when_extruded: bool = field(default=False)
    hidden_at_line_plots: bool = field(default=False)
    skip_OpenSees_definition: bool = field(default=False)

@dataclass
class NodeVisibility:
    """

    """
    connected_to_zerolength: bool = field(default=False)
