from .. import common
import numpy as np


def self_weight(mdl, lcase, factor=1.00):
    """
    """
    for elm in mdl.list_of_beamcolumn_elements():

        # if mdl.settings.imperial_units:
        #     g_const = common.G_CONST_IMPERIAL
        # else:
        #     g_const = common.G_CONST_SI

        weight_per_length = elm.section.weight_per_length()
        # apply weight as UDL
        if elm.visibility.skip_OpenSees_definition:
            # in that case apply its weight to the connecting nodes
            elm_len = elm.clear_length()
            elm_w = weight_per_length * elm_len * factor
            lcase.node_loads[
                elm.eleNodes[0].uid].add(
                    [0.00, 0.00, -elm_w/2.00, 0.00, 0.00, 0.00])
            lcase.node_loads[
                elm.eleNodes[1].uid].add(
                    [0.00, 0.00, -elm_w/2.00, 0.00, 0.00, 0.00])
        else:
            lcase.line_element_udl[
                elm.uid].add_glob(np.array(
                    [0., 0., -weight_per_length*factor]))


def self_mass(mdl, lcase):
    """
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
            elm.eleNodes[0].uid].add([half_mass]*3+[0.00]*3)
        lcase.node_mass[
            elm.eleNodes[1].uid].add([half_mass]*3+[0.00]*3)
