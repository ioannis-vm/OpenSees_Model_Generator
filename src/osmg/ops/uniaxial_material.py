"""
Model Generator for OpenSees ~ element
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

# pylint: disable=invalid-name


from dataclasses import dataclass


@dataclass
class UniaxialMaterial:
    """
    OpenSees uniaxialMaterial
    https://openseespydoc.readthedocs.io/en/latest/src/uniaxialMaterial.html
    """
    uid: int
    name: str


@dataclass
class Elastic(UniaxialMaterial):
    """
    OpenSees Elastic
    https://openseespydoc.readthedocs.io/en/latest/src/ElasticUni.html
    """
    e_mod: float

    def ops_args(self):
        """
        Returns the arguments required to define the object in
        OpenSees
        """
        return [
            'Elastic',
            self.uid,
            self.e_mod
        ]


@dataclass
class Steel02(UniaxialMaterial):
    """
    OpenSees Steel02
    https://openseespydoc.readthedocs.io/en/latest/src/steel02.html
    """
    Fy: float
    E0: float
    b: float
    params: tuple[float, float, float]
    a1: float
    a2: float
    a3: float
    a4: float
    sigInit: float
    G: float

    def ops_args(self):
        """
        Returns the arguments required to define the object in
        OpenSees
        """
        return [
            'Steel02',
            self.uid,
            self.Fy,
            self.E0,
            self.b,
            *self.params,
            self.a1,
            self.a2,
            self.a3,
            self.a4,
            self.sigInit
        ]


@dataclass
class Bilin(UniaxialMaterial):
    """
    OpenSees Bilin Material
    https://openseespydoc.readthedocs.io/en/latest/src/Bilin.html
    """
    K0: float
    as_Plus: float
    as_Neg: float
    My_Plus: float
    My_Neg: float
    Lamda_S: float
    Lamda_C: float
    Lamda_A: float
    Lamda_K: float
    c_S: float
    c_C: float
    c_A: float
    c_K: float
    theta_p_Plus: float
    theta_p_Neg: float
    theta_pc_Plus: float
    theta_pc_Neg: float
    Res_Pos: float
    Res_Neg: float
    theta_u_Plus: float
    theta_u_Neg: float
    D_Plus: float
    D_Neg: float
    nFactor: float

    def ops_args(self):
        """
        Returns the arguments required to define the object in
        OpenSees
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
            self.nFactor
        ]


@dataclass
class Pinching4(UniaxialMaterial):
    """
    OpenSees Pinching4 Material
    https://openseespydoc.readthedocs.io/en/latest/src/Pinching4.html?highlight=pinching4
    """
    ePf1: float
    ePf2: float
    ePf3: float
    ePf4: float
    ePd1: float
    ePd2: float
    ePd3: float
    ePd4: float
    eNf1: float
    eNf2: float
    eNf3: float
    eNf4: float
    eNd1: float
    eNd2: float
    eNd3: float
    eNd4: float
    rDispP: float
    fForceP: float
    uForceP: float
    rDispN: float
    fFoceN: float
    uForceN: float
    gK1: float
    gK2: float
    gK3: float
    gK4: float
    gKLim: float
    gD1: float
    gD2: float
    gD3: float
    gD4: float
    gDLim: float
    gF1: float
    gF2: float
    gF3: float
    gF4: float
    gFLim: float
    gE: float
    dmgType: str

    def ops_args(self):
        """
        Returns the arguments required to define the object in
        OpenSees
        """
        return [
            'Pinching4',
            self.uid,
            self.ePf1, self.ePf2, self.ePf3, self.ePf4,
            self.ePd1, self.ePd2, self.ePd3, self.ePd4,
            self.eNf1, self.eNf2, self.eNf3, self.eNf4,
            self.eNd1, self.eNd2, self.eNd3, self.eNd4,
            self.rDispP, self.fForceP, self.uForceP,
            self.rDispN, self.fFoceN, self.uForceN,
            self.gK1, self.gK2, self.gK3, self.gK4, self.gKLim,
            self.gD1, self.gD2, self.gD3, self.gD4, self.gDLim,
            self.gF1, self.gF2, self.gF3, self.gF4, self.gFLim,
            self.gE, self.dmgType
        ]


@dataclass
class Hysteretic(UniaxialMaterial):
    """
    OpenSees Bilin Material
    https://openseespydoc.readthedocs.io/en/latest/src/Bilin.html
    """
    p1: tuple[float, float]
    p2: tuple[float, float]
    p3: tuple[float, float]
    n1: tuple[float, float]
    n2: tuple[float, float]
    n3: tuple[float, float]
    pinchX: float
    pinchY: float
    damage1: float
    damage2: float
    beta: float

    def ops_args(self):
        """
        Returns the arguments required to define the object in
        OpenSees
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
            self.beta
        ]
