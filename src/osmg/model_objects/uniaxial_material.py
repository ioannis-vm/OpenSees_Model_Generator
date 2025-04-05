"""Defines OpenSees uniaxialMaterial interfrace objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from osmg.core.uid_object import UIDObject


@dataclass
class UniaxialMaterial(UIDObject):
    """
    OpenSees uniaxialMaterial.

    https://openseespydoc.readthedocs.io/en/latest/src/uniaxialMaterial.html

    """

    name: str

    def ops_args(self) -> list[object]:  # noqa: PLR6301
        """Obtain the OpenSees arguments."""
        msg = 'Subclasses should implement this.'
        raise NotImplementedError(msg)


@dataclass
class Elastic(UniaxialMaterial):
    """
    OpenSees Elastic.

    https://openseespydoc.readthedocs.io/en/latest/src/ElasticUni.html

    """

    e_mod: float

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return ['Elastic', self.uid, self.e_mod]


@dataclass
class ElasticPPGap(UniaxialMaterial):
    """
    OpenSees ElasticPPGap.

    https://opensees.berkeley.edu/wiki/index.php/Elastic-Perfectly_Plastic_Gap_Material

    """

    e_mod: float
    fy: float
    gap: float
    eta: float = field(default=0.0)
    damage: Literal['noDamage', 'damage'] = field(default='noDamage')

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return [
            'ElasticPPGap',
            self.uid,
            self.e_mod,
            self.fy,
            self.gap,
            self.eta,
            self.damage,
        ]


@dataclass
class Steel02(UniaxialMaterial):
    """
    OpenSees Steel02.

    https://openseespydoc.readthedocs.io/en/latest/src/steel02.html

    """

    Fy: float
    E0: float
    G: float
    b: float
    c_r0: float
    c_r1: float
    c_r2: float
    a1: float | None = field(default=None)
    a2: float | None = field(default=None)
    a3: float | None = field(default=None)
    a4: float | None = field(default=None)
    sig_init: float | None = field(default=None)

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        args = [
            'Steel02',
            self.uid,
            self.Fy,
            self.E0,
            self.b,
            self.c_r0,
            self.c_r1,
            self.c_r2,
        ]
        if self.a1:
            args.append(self.a1)
        if self.a2:
            args.append(self.a2)
        if self.a3:
            args.append(self.a3)
        if self.a4:
            args.append(self.a4)
        if self.sig_init:
            args.append(self.sig_init)

        return args


@dataclass
class Steel4(UniaxialMaterial):
    """
    OpenSees Steel4.

    https://openseespydoc.readthedocs.io/en/latest/src/steel4.html

    """

    Fy: float
    E0: float
    b_k: float | None = field(default=None)
    R_0: float = field(default=20.00)
    r_1: float = field(default=0.90)
    r_2: float = field(default=0.15)

    b_kc: float | None = field(default=False)
    R_0c: float = field(default=20.00)
    r_1c: float = field(default=0.90)
    r_2c: float = field(default=0.15)

    b_i: float | None = field(default=None)
    b_l: float | None = field(default=None)
    rho_i: float | None = field(default=None)
    R_i: float | None = field(default=None)
    l_yp: float | None = field(default=None)
    f_u: float | None = field(default=None)
    R_u: float | None = field(default=None)

    f_uc: float | None = field(default=None)
    R_uc: float | None = field(default=None)
    b_ic: float | None = field(default=None)
    b_lc: float | None = field(default=None)
    rho_ic: float | None = field(default=None)
    R_ic: float | None = field(default=None)

    sig_init: float | None = field(default=None)
    cycNum: float | None = field(default=None)  # noqa: N815

    def ops_args(self) -> list[object]:  # noqa: C901
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        # non-symmetric behavior
        if self.b_kc:
            assert self.R_0c is not None
            assert self.r_1c is not None
            assert self.r_2c is not None
            assert self.b_k
            asym = True
        else:
            asym = False
        # ultimate strength limit
        if self.f_u:
            ultimate = True
            if asym:
                assert self.f_uc is not None
                assert self.R_uc is not None
        else:
            ultimate = False
        # isotropic hardening
        if self.b_i:
            iso = True
            assert self.b_l is not None
            assert self.rho_i is not None
            assert self.R_i is not None
            assert self.l_yp is not None
            if asym:
                assert self.b_lc is not None
                assert self.rho_ic is not None
                assert self.R_ic is not None
        else:
            iso = False
        # kinematic hardening
        if self.b_k:
            kinematic = True
            assert self.R_0 is not None
            assert self.r_1 is not None
            assert self.r_2 is not None
            if asym:
                assert self.R_0c is not None
                assert self.r_1c is not None
                assert self.r_2c is not None
        else:
            kinematic = False

        #
        # construct argument list
        #

        # these are required and will always be there
        args = ['Steel4', self.uid, self.Fy, self.E0]

        # optional arguments:
        if asym:
            args.extend(['-asym'])
        if kinematic:
            args.extend(['-kin', self.b_k, self.R_0, self.r_1, self.r_2])
            if asym:
                args.extend([self.b_kc, self.R_0c, self.r_1c, self.r_2c])
        if iso:
            args.extend(
                ['-iso', self.b_i, self.rho_i, self.b_l, self.R_i, self.l_yp]
            )
            if asym:
                args.extend([self.b_ic, self.rho_ic, self.b_lc, self.R_ic])
        if ultimate:
            args.extend(['-ult', self.f_u, self.R_u])
            if asym:
                args.extend([self.f_uc, self.R_uc])
        if self.sig_init:
            args.extend(['-init', self.sig_init])
        if self.cycNum:
            args.extend(['-mem', self.cycNum])

        return args


@dataclass
class Bilin(UniaxialMaterial):
    """
    OpenSees Bilin Material.

    https://openseespydoc.readthedocs.io/en/latest/src/Bilin.html

    """

    K0: float
    as_Plus: float  # noqa: N815
    as_Neg: float  # noqa: N815
    My_Plus: float
    My_Neg: float
    Lamda_S: float
    Lamda_C: float
    Lamda_A: float
    Lamda_K: float
    c_S: float  # noqa: N815
    c_C: float  # noqa: N815
    c_A: float  # noqa: N815
    c_K: float  # noqa: N815
    theta_p_Plus: float  # noqa: N815
    theta_p_Neg: float  # noqa: N815
    theta_pc_Plus: float  # noqa: N815
    theta_pc_Neg: float  # noqa: N815
    Res_Pos: float
    Res_Neg: float
    theta_u_Plus: float  # noqa: N815
    theta_u_Neg: float  # noqa: N815
    D_Plus: float
    D_Neg: float
    nFactor: float  # noqa: N815

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return [
            'Bilin',
            self.uid,
            self.K0,
            self.as_Plus,
            self.as_Neg,
            self.My_Plus,
            self.My_Neg,
            self.Lamda_S,
            self.Lamda_C,
            self.Lamda_A,
            self.Lamda_K,
            self.c_S,
            self.c_C,
            self.c_A,
            self.c_K,
            self.theta_p_Plus,
            self.theta_p_Neg,
            self.theta_pc_Plus,
            self.theta_pc_Neg,
            self.Res_Pos,
            self.Res_Neg,
            self.theta_u_Plus,
            self.theta_u_Neg,
            self.D_Plus,
            self.D_Neg,
            self.nFactor,
        ]


@dataclass
class IMKBilin(UniaxialMaterial):
    """
    OpenSees IMKBilin Material.

    https://portwooddigital.com/2019/12/08/an-update-of-the-imk-models/

    """

    K0: float
    theta_p_Plus: float  # noqa: N815
    theta_pc_Plus: float  # noqa: N815
    theta_u_Plus: float  # noqa: N815
    My_Plus: float
    as_Plus: float  # noqa: N815
    Res_Pos: float
    theta_p_Neg: float  # noqa: N815
    theta_pc_Neg: float  # noqa: N815
    theta_u_Neg: float  # noqa: N815
    My_Neg: float
    as_Neg: float  # noqa: N815
    Res_Neg: float
    Lamda_S: float
    Lamda_C: float
    Lamda_K: float
    c_S: float  # noqa: N815
    c_C: float  # noqa: N815
    c_K: float  # noqa: N815
    D_Plus: float
    D_Neg: float

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return [
            'IMKBilin',
            self.uid,
            self.K0,
            self.theta_p_Plus,
            self.theta_pc_Plus,
            self.theta_u_Plus,
            self.My_Plus,
            self.as_Plus,
            self.Res_Pos,
            self.theta_p_Neg,
            self.theta_pc_Neg,
            self.theta_u_Neg,
            self.My_Neg,
            self.as_Neg,
            self.Res_Neg,
            self.Lamda_S,
            self.Lamda_C,
            self.Lamda_K,
            self.c_S,
            self.c_C,
            self.c_K,
            self.D_Plus,
            self.D_Neg,
        ]


@dataclass
class Pinching4(UniaxialMaterial):
    """
    OpenSees Pinching4 Material.

    https://openseespydoc.readthedocs.io/en/latest/src/Pinching4.html

    """

    ePf1: float  # noqa: N815
    ePf2: float  # noqa: N815
    ePf3: float  # noqa: N815
    ePf4: float  # noqa: N815
    ePd1: float  # noqa: N815
    ePd2: float  # noqa: N815
    ePd3: float  # noqa: N815
    ePd4: float  # noqa: N815
    eNf1: float  # noqa: N815
    eNf2: float  # noqa: N815
    eNf3: float  # noqa: N815
    eNf4: float  # noqa: N815
    eNd1: float  # noqa: N815
    eNd2: float  # noqa: N815
    eNd3: float  # noqa: N815
    eNd4: float  # noqa: N815
    rDispP: float  # noqa: N815
    fForceP: float  # noqa: N815
    uForceP: float  # noqa: N815
    rDispN: float  # noqa: N815
    fFoceN: float  # noqa: N815
    uForceN: float  # noqa: N815
    gK1: float  # noqa: N815
    gK2: float  # noqa: N815
    gK3: float  # noqa: N815
    gK4: float  # noqa: N815
    gKLim: float  # noqa: N815
    gD1: float  # noqa: N815
    gD2: float  # noqa: N815
    gD3: float  # noqa: N815
    gD4: float  # noqa: N815
    gDLim: float  # noqa: N815
    gF1: float  # noqa: N815
    gF2: float  # noqa: N815
    gF3: float  # noqa: N815
    gF4: float  # noqa: N815
    gFLim: float  # noqa: N815
    gE: float  # noqa: N815
    dmgType: str  # noqa: N815

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return [
            'Pinching4',
            self.uid,
            self.ePf1,
            self.ePf2,
            self.ePf3,
            self.ePf4,
            self.ePd1,
            self.ePd2,
            self.ePd3,
            self.ePd4,
            self.eNf1,
            self.eNf2,
            self.eNf3,
            self.eNf4,
            self.eNd1,
            self.eNd2,
            self.eNd3,
            self.eNd4,
            self.rDispP,
            self.fForceP,
            self.uForceP,
            self.rDispN,
            self.fFoceN,
            self.uForceN,
            self.gK1,
            self.gK2,
            self.gK3,
            self.gK4,
            self.gKLim,
            self.gD1,
            self.gD2,
            self.gD3,
            self.gD4,
            self.gDLim,
            self.gF1,
            self.gF2,
            self.gF3,
            self.gF4,
            self.gFLim,
            self.gE,
            self.dmgType,
        ]


@dataclass
class Hysteretic(UniaxialMaterial):
    """
    OpenSees Bilin Material.

    https://openseespydoc.readthedocs.io/en/latest/src/Bilin.html

    """

    p1: tuple[float, float]
    p2: tuple[float, float]
    p3: tuple[float, float]
    n1: tuple[float, float]
    n2: tuple[float, float]
    n3: tuple[float, float]
    pinchX: float  # noqa: N815
    pinchY: float  # noqa: N815
    damage1: float
    damage2: float
    beta: float

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return [
            'Hysteretic',
            self.uid,
            *self.p1,
            *self.p2,
            *self.p3,
            *self.n1,
            *self.n2,
            *self.n3,
            self.pinchX,
            self.pinchY,
            self.damage1,
            self.damage2,
            self.beta,
        ]


@dataclass
class Fatigue(UniaxialMaterial):
    """
    OpenSees Fatigue Material.

    https://openseespydoc.readthedocs.io/en/latest/src/Fatigue.html

    """

    predecessor: UniaxialMaterial
    e_mod: float = field(default=0.191)
    var_m: float = field(default=-0.458)
    var_min: float = field(default=-1.0e16)
    var_max: float = field(default=+1.0e16)

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return [
            'Fatigue',
            self.uid,
            self.predecessor.uid,
            '-E0',
            self.e_mod,
            '-m',
            self.var_m,
            '-min',
            self.var_min,
            '-max',
            self.var_max,
        ]


@dataclass
class MaxStrainRange(UniaxialMaterial):
    """
    OpenSees MaxStrainRange Material.

    ~not officially added yet~

    """

    predecessor: UniaxialMaterial
    msr_fracture: float
    min_fracture: float | None = field(default=None)
    max_fracture: float | None = field(default=None)
    tangent_ratio: float | None = field(default=None)
    def_coeff: float | None = field(default=None)
    node_tags: tuple[int, int] | None = field(default=None)
    elements_to_remove: list[int] | None = field(default=None)

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        args = [
            'MaxStrainRange',
            self.uid,
            self.predecessor.uid,
            self.msr_fracture,
        ]
        if self.min_fracture:
            args.extend(['-min', self.min_fracture])
        if self.max_fracture:
            args.extend(['-max', self.max_fracture])
        if self.tangent_ratio:
            args.extend(['-tangentRatio', self.tangent_ratio])
        if self.def_coeff:
            args.extend(['-defCoeff', self.def_coeff])
        if self.node_tags:
            args.extend(['-nodeTags', *self.node_tags])
        if self.elements_to_remove:
            args.extend(['-eleTag', *self.elements_to_remove])

        return args


@dataclass
class MinMax(UniaxialMaterial):
    """
    OpenSees MinMax Material.

    https://openseespydoc.readthedocs.io/en/latest/src/MinMax.html

    """

    predecessor: UniaxialMaterial
    min_strain: float
    max_strain: float

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return [
            'MinMax',
            self.uid,
            self.predecessor.uid,
            '-min',
            self.min_strain,
            '-max',
            self.max_strain,
        ]
