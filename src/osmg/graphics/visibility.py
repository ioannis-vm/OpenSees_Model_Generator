"""Model Generator for OpenSees ~ visibility."""

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
    hidden_basic_forces: bool = field(default=False)
