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

from dataclasses import dataclass, field

# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes


@dataclass(repr=False)
class uniaxialMaterial:
    """
    OpenSees uniaxialMaterial
    https://openseespydoc.readthedocs.io/en/latest/src/uniaxialMaterial.html
    """
    uid: int
    name: str


@dataclass(repr=False)
class Steel02(uniaxialMaterial):
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
