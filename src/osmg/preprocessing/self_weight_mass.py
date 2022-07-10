from .. import common
import numpy as np

def self_weight_mass(mdl, lcase):
    """
    """
    for elm in mdl.list_of_beamcolumn_elements():

        if mdl.settings.imperial_units:
            g_const = common.G_CONST_IMPERIAL
        else:
            g_const = common.G_CONST_SI

        weight_per_length = elm.section.weight_per_length()
        mass_per_length = weight_per_length / g_const
        # mass applied dead loads must also be accounted for
        added_mass = (- lcase.line_element_udl
                      .registry[elm.uid]
                      .get_udl_self_other_glob()[2] / g_const)

        # apply lumped mass at the connecting nodes
        half_mass = ((mass_per_length + added_mass) *
                     elm.clear_length() / 2.00)
        lcase.node_mass.registry[
            elm.eleNodes[0].uid].add([half_mass]*6, 'other')
        lcase.node_mass.registry[
            elm.eleNodes[1].uid].add([half_mass]*6, 'other')

        # apply weight as UDL
        if elm.visibility.skip_OpenSees_definition:
            # in that case apply its weight to the connecting nodes
            elm_len = elm.clear_length()
            elm_w = weight_per_length * elm_len
            lcase.node_loads.registry[
                elm.eleNodes[0].uid].add(
                    [0.00, 0.00, -elm_w/2.00, 0.00, 0.00, 0.00], 'other')
            lcase.node_loads.registry[
                elm.eleNodes[1].uid].add(
                    [0.00, 0.00, -elm_w/2.00, 0.00, 0.00, 0.00], 'other')
        else:
            lcase.line_element_udl.registry[
                elm.uid].add_glob(np.array(
                    [0., 0., -weight_per_length]), 'self_weight')

    # for lvl in mdl.levels.registry.values():
    #     if not lvl.diaphragm:
    #         continue
    #     if lvl.restraint != "free":
    #         continue
    #     # accumulate all the mass at the parent nodes
    #     properties = mesher.geometric_properties(lvl.floor_coordinates)
    #     floor_mass = -lvl.surface_load * \
    #         properties['area'] / common.G_CONST
    #     assert(floor_mass >= 0.00),\
    #         "Error: floor area properties\n" + \
    #         "Overall floor area should be negative (by convention)."
    #     floor_centroid = properties['centroid']
    #     floor_mass_inertia = properties['inertia']['ir_mass']\
    #         * floor_mass
    #     self_mass_centroid = np.array([0.00, 0.00])  # excluding floor
    #     total_self_mass = 0.00
    #     for nd in lvl.list_of_all_nodes():
    #         self_mass_centroid += nd.coords[0:2] * nd.mass[0]
    #         total_self_mass += nd.mass[0]
    #     self_mass_centroid = self_mass_centroid * \
    #         (1.00/total_self_mass)
    #     total_mass = total_self_mass + floor_mass
    #     # combined
    #     centroid = [
    #         (self_mass_centroid[0] * total_self_mass +
    #          floor_centroid[0] * floor_mass) / total_mass,
    #         (self_mass_centroid[1] * total_self_mass +
    #          floor_centroid[1] * floor_mass) / total_mass
    #     ]
    #     lvl.parent_node = node.Node(
    #         np.array([centroid[0], centroid[1],
    #                   lvl.elevation]), "parent")
    #     lvl.parent_node.mass = np.array([total_mass,
    #                                      total_mass,
    #                                      0.,
    #                                      0., 0., 0.])
    #     lvl.parent_node.mass[5] = floor_mass_inertia
    #     for nd in lvl.list_of_all_nodes():
    #         lvl.parent_node.mass[5] += nd.mass[0] * \
    #             np.linalg.norm(lvl.parent_node.coords - nd.coords)**2
    #         nd.mass[0] = common.EPSILON
    #         nd.mass[1] = common.EPSILON
    #         nd.mass[2] = common.EPSILON
    # mdl.dct_update_required = True
