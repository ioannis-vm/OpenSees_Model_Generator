"""
Common definitions
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from typing import OrderedDict

# very big, very small numbers used for
# comparing floats and hashing
EPSILON = 1.00E-6
ALPHA = 10000000.00

# gravitational acceleration
G_CONST = 386.22  # in/s**2

# quantities to use for extreme stiffnesses
STIFF_ROT = 1.0e12
STIFF = 1.0e12  # note: too high a value causes convergence problems
TINY = 1.0e-12


def methods(obj: object) -> list[str]:
    """
    Returns the names of all methods of an object.
    """
    object_methods = [method_name for method_name in dir(obj)
                      if callable(getattr(obj, method_name))]
    return object_methods


def previous_element(dct: OrderedDict, key):
    """
    Returns the previous object in an OrderedDict
    given a target key, assuming it is in the OrderedDict.
    If it is not, it returns None.
    If the target key is the first object, it returns None.
    """
    if key in dct:
        key_list = list(dct.keys())
        idx = key_list.index(key)
        if idx == 0:
            ans = None
        else:
            ans = dct[key_list[idx-1]]
    else:
        ans = None
    return ans
