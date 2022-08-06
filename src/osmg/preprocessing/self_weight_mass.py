"""
Model Generator for OpenSees ~ self weight, self mass
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

import numpy as np
from .. import common


def self_weight(mdl, lcase, factor=1.00):
    """
    Assigns the structure's self weight to its members
    """
    for elm in mdl.list_of_beamcolumn_elements():

        # if mdl.settings.imperial_units:
        #     g_const = common.G_CONST_IMPERIAL
        # else:
        #     g_const = common.G_CONST_SI

        weight_per_length = elm.section.weight_per_length()
        # apply weight as UDL
        if elm.visibility.skip_opensees_definition:
            # in that case apply its weight to the connecting nodes
            elm_len = elm.clear_length()
            elm_w = weight_per_length * elm_len * factor
            lcase.node_loads[
                elm.nodes[0].uid].add(
                    [0.00, 0.00, -elm_w/2.00, 0.00, 0.00, 0.00])
            lcase.node_loads[
                elm.nodes[1].uid].add(
                    [0.00, 0.00, -elm_w/2.00, 0.00, 0.00, 0.00])
        else:
            lcase.line_element_udl[
                elm.uid].add_glob(np.array(
                    [0., 0., -weight_per_length*factor]))


def self_mass(mdl, lcase):
    """
    Assigns the structure's self mass to its members
    """
    for elm in mdl.list_of_beamcolumn_elements():

        if mdl.settings.imperial_units:
            g_const = common.G_CONST_IMPERIAL
        else:
            g_const = common.G_CONST_SI

        weight_per_length = elm.section.weight_per_length()
        mass_per_length = weight_per_length / g_const
        # apply lumped mass at the connecting nodes
        half_mass = (mass_per_length *
                     elm.clear_length() / 2.00)
        lcase.node_mass[
            elm.nodes[0].uid].add([half_mass]*3+[0.00]*3)
        lcase.node_mass[
            elm.nodes[1].uid].add([half_mass]*3+[0.00]*3)
