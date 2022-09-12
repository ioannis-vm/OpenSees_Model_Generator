"""
Model Generator for OpenSees ~ material generator
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator


from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from ..ops.section import FiberSection
from ..physical_material import PhysicalMaterial
from ..model import Model
from ..ops.uniaxial_material import Steel02
from ..ops.uniaxial_material import Fatigue


nparr = npt.NDArray[np.float64]


@dataclass(repr=False)
class MaterialGenerator:
    """
    This object introduces element-specific materials to a model.
    """
    model: Model

    def generate_steel_hss_rect_brace_fatigue_mat(
            self,
            section: FiberSection,
            physical_material: PhysicalMaterial,
            brace_length: float
    ):
        """
        Karamanci, E., & Lignos, D. G. (2014). Computational approach
        for collapse assessment of concentrically braced frames in
        seismic regions. Journal of Structural Engineering, 140(8),
        A4014019.
        Adapted from https://github.com/amaelkady/OpenSEES_Models_CBF
        """
        var_b = 0.001
        var_r0 = 22.00
        var_c_r1 = 0.925
        var_c_r2 = 0.25
        var_a1 = 0.03
        var_a2 = 1.00
        var_a3 = 0.02
        var_a4 = 1.00
        var_m = -0.30
        var_l = brace_length
        assert section.properties is not None
        var_ry = min(section.properties['rx'], section.properties['ry'])
        var_wt = section.properties['h'] / section.properties['tdes']
        var_e = physical_material.e_mod
        var_g = physical_material.g_mod
        var_fy = physical_material.f_y
        var_e_0 = (0.291 * (var_l/var_ry)**(-0.484)
                   * var_wt**(-0.613) * (var_e/var_fy)**0.303) + 1e90

        steel02_mat = Steel02(
            self.model.uid_generator.new('uniaxial material'),
            'auto_steel02_brace_mat',
            var_fy, var_e, var_b, (var_r0, var_c_r1, var_c_r2),
            var_a1, var_a2, var_a3, var_a4, 0.00, var_g)

        fatigue_mat = Fatigue(
            self.model.uid_generator.new('uniaxial material'),
            'auto_fatigue_brace_mat',
            steel02_mat,
            var_e_0, var_m)

        return fatigue_mat
