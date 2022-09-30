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

from pprint import pprint
from typing import OrderedDict
from typing import Any

# very big, very small numbers used for
# comparing floats and hashing
EPSILON = 1.00E-6
ALPHA = 10000000.00

# gravitational acceleration
G_CONST_IMPERIAL = 386.22  # in/s**2
G_CONST_SI = 9.81  # m/s**2

# quantities to use for extreme stiffnesses
STIFF_ROT = 1.0e15
STIFF = 1.0e15  # note: too high a value causes convergence problems
TINY = 1.0e-12


def methods(obj: object) -> list[str]:
    """
    Returns the names of all methods of an object.
    """
    object_methods = [method_name for method_name in dir(obj)
                      if callable(getattr(obj, method_name))]
    return object_methods


def print_methods(obj: object):
    """
    Prints the methods of an object
    """
    object_methods = methods(obj)
    pprint(object_methods)


def print_dir(obj: object):
    """
    Prints the entire output of `dir()` of an object
    """
    pprint(dir(obj))


def previous_element(dct: OrderedDict[Any, Any], key):
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
