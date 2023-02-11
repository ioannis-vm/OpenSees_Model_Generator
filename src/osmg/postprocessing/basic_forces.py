"""
Defines the :func:`~osmg.postprocessing.basic_forces.basic_forces`
method.
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

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import pandas as pd
from ..ops.element import TrussBar
from ..ops.element import ElasticBeamColumn
from ..ops.element import DispBeamColumn
from ..solver import ModalResponseSpectrumAnalysis

if TYPE_CHECKING:
    from ..solver import Analysis


nparr = npt.NDArray[np.float64]

# pylint: disable=no-else-return


def basic_forces(
    anl: Analysis,
    case_name: str,
    step: int,
    elm: TrussBar | ElasticBeamColumn | DispBeamColumn,
    num_points: int,
    as_tuple: bool = False,
) -> object:
    """
    Returns the basic forces of a specified element.

    Arguments:
      anl: Analysis object.
      case_name: Name of loadcase to look for results.
      step: Analysis step to look for results.
      elm: Element of which the basic forces are required.
      num_points: Number of points along the length of the element for
        which to report the basic forces (oftentimes called `stations`
        in analysis software.)
      as_tupe: Whether to return the results in the form of a pandas
        DataFrame or a tuple.

    """

    if isinstance(anl, ModalResponseSpectrumAnalysis):
        forces = anl.combined_basic_forces(elm.uid)
        w_x, w_y, w_z = (0.00, 0.00, 0.00)
    else:
        forces = anl.results[case_name].element_forces[elm.uid][step]
        if isinstance(elm, (ElasticBeamColumn, DispBeamColumn)):
            w_x, w_y, w_z = anl.load_cases[
                case_name].line_element_udl[elm.uid].val
        else:
            w_x, w_y, w_z = (0.00, 0.00, 0.00)

    if isinstance(elm, (ElasticBeamColumn, DispBeamColumn)):
        p_i = np.array(elm.nodes[0].coords) + elm.geomtransf.offset_i
        p_j = np.array(elm.nodes[1].coords) + elm.geomtransf.offset_j
    else:
        p_i = np.array(elm.nodes[0].coords)
        p_j = np.array(elm.nodes[1].coords)

    n_i, qy_i, qz_i = forces[0:3]
    t_i, my_i, mz_i = forces[3:6]

    len_clr = np.linalg.norm(p_i - p_j)

    t_vec = np.linspace(0.00, len_clr, num=num_points)

    nx_vec = -t_vec * w_x - n_i
    qy_vec = t_vec * w_y + qy_i
    qz_vec = t_vec * w_z + qz_i
    tx_vec = np.full(num_points, -t_i)
    mz_vec = t_vec**2 * 0.50 * w_y + t_vec * qy_i - mz_i
    my_vec = t_vec**2 * 0.50 * w_z + t_vec * qz_i + my_i

    if as_tuple:
        return (nx_vec, qy_vec, qz_vec, tx_vec, mz_vec, my_vec)
    else:
        dframe = pd.DataFrame.from_dict(
            {
                "nx": nx_vec,
                "qy": qy_vec,
                "qz": qz_vec,
                "tx": tx_vec,
                "mz": mz_vec,
                "my": my_vec,
            }
        )
        return dframe
