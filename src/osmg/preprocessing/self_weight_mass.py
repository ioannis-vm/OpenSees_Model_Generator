"""
Self-weight, self-mass.

Defines methods to assign self-weight and self-mass to a loadcase
using a given model.
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

import numpy as np

from osmg.core import common
from osmg.core.load_case import LoadCase
from osmg.core.model import Model
from osmg.elements import element


def self_weight(mdl: Model, lcase: LoadCase, factor: float = 1.00) -> None:
    """Assign the structure's self weight to its members."""
    for elm in mdl.list_of_elements():
        if isinstance(elm, (element.ElasticBeamColumn, element.DispBeamColumn)):
            weight_per_length = elm.section.weight_per_length()
            # apply weight as UDL
            if elm.visibility.skip_opensees_definition:
                # in that case apply its weight to the connecting nodes
                elm_len = elm.clear_length()
                elm_w = weight_per_length * elm_len * factor
                lcase.node_loads[elm.nodes[0].uid].add(
                    [0.00, 0.00, -elm_w / 2.00, 0.00, 0.00, 0.00]
                )
                lcase.node_loads[elm.nodes[1].uid].add(
                    [0.00, 0.00, -elm_w / 2.00, 0.00, 0.00, 0.00]
                )
            else:
                lcase.line_element_udl[elm.uid].add_glob(
                    np.array([0.0, 0.0, -weight_per_length * factor])
                )
        if isinstance(elm, element.TrussBar):
            weight_per_length = elm.weight_per_length
            # apply its weight to the connecting nodes
            elm_len = elm.clear_length()
            elm_w = weight_per_length * elm_len * factor
            lcase.node_loads[elm.nodes[0].uid].add(
                [0.00, 0.00, -elm_w / 2.00, 0.00, 0.00, 0.00]
            )
            lcase.node_loads[elm.nodes[1].uid].add(
                [0.00, 0.00, -elm_w / 2.00, 0.00, 0.00, 0.00]
            )


def self_mass(mdl: Model, lcase: LoadCase) -> None:
    """Assign the structure's self mass to its members."""
    if mdl.settings.imperial_units:
        g_const = common.G_CONST_IMPERIAL
    else:
        g_const = common.G_CONST_SI

    for elm in mdl.list_of_elements():
        if isinstance(elm, (element.ElasticBeamColumn, element.DispBeamColumn)):
            weight_per_length = elm.section.weight_per_length()
        elif isinstance(elm, element.TrussBar):
            weight_per_length = elm.weight_per_length
        else:
            # don't consider other types of elements
            continue

        mass_per_length = weight_per_length / g_const
        # apply lumped mass at the connecting nodes
        half_mass = mass_per_length * elm.clear_length() / 2.00
        lcase.node_mass[elm.nodes[0].uid].add([half_mass] * 3 + [0.00] * 3)
        lcase.node_mass[elm.nodes[1].uid].add([half_mass] * 3 + [0.00] * 3)
