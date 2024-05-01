"""
BRB element generator.

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

from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from ...gen.mesh_shapes import rect_mesh
from ...model import Model
from ...gen.component_gen import TrussBarGenerator
from ...ops.uniaxial_material import Steel4
from ...ops.uniaxial_material import Fatigue


# pylint: disable=invalid-name

nparr = npt.NDArray[np.float64]


@dataclass(repr=False)
class BRBGenSettings:
    """
    This object holds default values that are used by BRBGenerator
    objects, to better organize those values and improve the usability
    of the BRBGenerator objects by allowing their methods to have
    fewer arguments.

    """

    steel4_b_k: float = field(default=0.003)
    steel4_b_kc: float = field(default=0.023)
    steel4_R_0: float = field(default=25.00)
    steel4_R_0c: float = field(default=25.00)
    steel4_r_1: float = field(default=0.90)
    steel4_r_1c: float = field(default=0.90)
    steel4_r_2: float = field(default=0.15)
    steel4_r_2c: float = field(default=0.15)
    steel4_b_i: float = field(default=0.0025)
    steel4_b_ic: float = field(default=0.0045)
    steel4_rho_i: float = field(default=1.34)
    steel4_rho_ic: float = field(default=0.77)
    steel4_b_l: float = field(default=0.004)
    steel4_b_lc: float = field(default=0.004)
    steel4_R_i: float = field(default=1.0)
    steel4_R_ic: float = field(default=1.0)
    steel4_l_yp: float = field(default=1.0)
    fatigue_e_mod: float = field(default=0.12)
    fatigue_var_m: float = field(default=-0.458)
    transf_type: str = field(default='Corotational')


@dataclass(repr=False)
class BRBGenerator:
    """
    Uses the lower-level `gen` classes to simplify the definition of
    BRB elements simulated using a corotational truss opensees element
    and the Steel4 material wrapped under a Fatigue material, as done
    in [1] and [2].

    1. Zsarnoczay, Adam. "Experimental and numerical investigation of
    buckling restrained braced frames for Eurocode conform design
    procedure development." (2013).

    2. Simpson, Barbara Gwynne. Design development for steel
    strongback braced frames to mitigate concentrations of
    damage. University of California, Berkeley, 2018.

    """

    model: Model
    settings: BRBGenSettings = field(default_factory=BRBGenSettings)

    def add_brb(
        self,
        xi_coord: float,
        yi_coord: float,
        lvl_key_i: int,
        offset_i: nparr,
        snap_i: str,
        xj_coord: float,
        yj_coord: float,
        lvl_key_j: int,
        offset_j: nparr,
        snap_j: str,
        area: float,
        f_y: float,
        e_0: float,
        casing_size: float,
        unit_weight: float,
    ) -> None:
        """
        Adds a BRB element to the model.

        Parameters:
          xi_coord: x coordinate of the i-end
          yi_coord: y coordinate of the i-end
          lvl_key_i: Level ID of the i-end
          offset_i: Rigid offset at the i-end
          snap_i: Snap tag for the i-end connectivity used to
            automatically determine an offset based on existing
            elements.
          xj_coord: x coordinate of the j-end
          yj_coord: y coordinate of the j-end
          lvl_key_j: Level ID of the j-end
          offset_j: Rigid offset at the j-end
          snap_j: Snap tag for the j-end connectivity used to
            automatically determine an offset based on existing
            elements.
          area: Core area of the BRB element.
          f_y: Steel stress.
          e_0: Young's modulus, adjusted to account for
            non-prismatic shape. See [2], Figure 4.15, and Equations
            4.7, 4.8.
          casing_size: Assuming a square cross-section,
            controls the side length. Used for visualization purposes
            and to determine the weight of the BRB element, which is
            lumped to the connecting nodes. This parameter has no
            effect on the force-deformation relationship of the
            element.
          unit_weight: Weight per unit volume, used together
            with the casing_size, assuming a square section and a
            workpoint-to-workpoint length, in order to determine the
            weight of the BRB, which is lumped to the two connecting
            nodes of the Truss element as weight and mass when the
            self-weight and self-mass preprocessing methods are
            called.

        """

        trg = TrussBarGenerator(self.model)

        uid = self.model.uid_generator.new("uniaxial material")
        mat = Steel4(
            uid=uid,
            name=f'auto_BRB_{uid}',
            Fy=f_y,
            E0=e_0,
            b_k=self.settings.steel4_b_k,
            b_kc=self.settings.steel4_b_kc,
            R_0=self.settings.steel4_R_0,
            R_0c=self.settings.steel4_R_0c,
            r_1=self.settings.steel4_r_1,
            r_1c=self.settings.steel4_r_1c,
            r_2=self.settings.steel4_r_2,
            r_2c=self.settings.steel4_r_2c,
            b_i=self.settings.steel4_b_i,
            b_ic=self.settings.steel4_b_ic,
            rho_i=self.settings.steel4_rho_i,
            rho_ic=self.settings.steel4_rho_ic,
            b_l=self.settings.steel4_b_l,
            b_lc=self.settings.steel4_b_lc,
            R_i=self.settings.steel4_R_i,
            R_ic=self.settings.steel4_R_ic,
            l_yp=self.settings.steel4_l_yp,
        )
        uid = self.model.uid_generator.new("uniaxial material")
        mat_fatigue = Fatigue(
            uid=uid,
            name=f'auto_BRB_fatigue_{uid}',
            predecessor=mat,
            e_mod=self.settings.fatigue_e_mod,
            var_m=self.settings.fatigue_var_m,
        )
        trg.add(
            xi_coord,
            yi_coord,
            lvl_key_i,
            offset_i,
            snap_i,
            xj_coord,
            yj_coord,
            lvl_key_j,
            offset_j,
            snap_j,
            transf_type=self.settings.transf_type,
            area=area,
            mat=mat_fatigue,
            outside_shape=rect_mesh(casing_size, casing_size),
            weight_per_length=unit_weight * casing_size**2,
        )
