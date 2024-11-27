"""Common definitions."""

from __future__ import annotations

import re
from pprint import pprint
from typing import Hashable, OrderedDict, TypeVar

import numpy as np
import numpy.typing as npt

# very big, very small numbers used for
# comparing floats and hashing
EPSILON = 1.00e-6
ALPHA = 1.00e8

# gravitational acceleration
G_CONST_IMPERIAL = 386.22  # in/s**2
G_CONST_SI = 9.81  # m/s**2

# quantities to use for extreme stiffnesses
STIFF_ROT = 1.0e15
STIFF = 1.0e10
TINY = 1.0e-12

NDM: dict[str, int] = {
    '1D1DOF': 1,
    '2D Truss': 2,
    '2D Frame': 2,
    '3D Truss': 3,
    '3D Frame': 3,
}
NDF: dict[str, int] = {
    '1D1DOF': 1,
    '2D Truss': 2,
    '2D Frame': 3,
    '3D Truss': 3,
    '3D Frame': 6,
}

TWO_DIMENSIONAL = 2
THREE_DIMENSIONAL = 3

numpy_array = npt.NDArray[np.float64]


def methods(obj: object) -> list[str]:
    """
    Get the methods of an object.

    Returns:
      The names of all methods of an object, excluding the dunder
      methods.

    Example:
        >>> class TestClass:
        ...     def method_1(self):
        ...         pass
        ...
        ...     def method_2(self):
        ...         pass
        ...
        >>> obj = TestClass()
        >>> methods(obj)
        ['method_1', 'method_2']
    """
    object_methods = [
        method_name
        for method_name in dir(obj)
        if callable(getattr(obj, method_name))
    ]
    pattern = r'__.*__'
    return [s for s in object_methods if not re.match(pattern, s)]


def print_methods(obj: object) -> None:
    """Print the methods of an object."""
    object_methods = methods(obj)
    pprint(object_methods)  # noqa: T203


def print_dir(obj: object) -> None:
    """Print the entire output of `dir()` of an object."""
    pprint(dir(obj))  # noqa: T203


K = TypeVar('K', bound=Hashable)  # Represents the key type (must be Hashable)
V = TypeVar('V')  # Represents the value type


def previous_element(dct: OrderedDict[K, V], key: K) -> V | None:
    """
    Get the previous element.

    Returns the value of the element that comes before the given key
    in an ordered dictionary.
    If the key is not in the dictionary, or if it is the first element
    in the dictionary, returns None.

    Arguments:
        dct: An ordered dictionary.
        key: The key of the element whose previous element we want to
        find.

    Returns:
        The value of the element that comes before the given key in
        the dictionary, or None if there is no such element.
    """
    if key in dct:
        key_list = list(dct.keys())
        idx = key_list.index(key)
        if idx == 0:
            result = None
        else:
            result = dct[key_list[idx - 1]]
    else:
        result = None
    return result
