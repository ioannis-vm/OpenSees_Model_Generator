"""
Model Generator for OpenSees ~ basic forces
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from __future__ import annotations
import numpy as np
import numpy.typing as npt
import pandas as pd
from ..solver import Analysis
from ..ops.element import ElasticBeamColumn
from ..ops.element import DispBeamColumn
from ..solver import ModalResponseSpectrumAnalysis

nparr = npt.NDArray[np.float64]

# pylint: disable=no-else-return


def basic_forces(
        anl: Analysis,
        case_name: str,
        step: int,
        elm: ElasticBeamColumn | DispBeamColumn,
        num_points: int,
        as_tuple: bool = False) -> object:
    """
    Returns the basic forces of a specified element
    """
    if isinstance(anl, ModalResponseSpectrumAnalysis):
        forces = anl.combined_basic_forces(elm.uid)
        w_x, w_y, w_z = (0.00, 0.00, 0.00)
    else:
        forces = anl.results[case_name].element_forces[
            elm.uid][step]
        w_x, w_y, w_z = anl.load_cases[case_name].line_element_udl[elm.uid].val

    n_i, qy_i, qz_i = forces[0:3]
    t_i, my_i, mz_i = forces[3:6]

    p_i = np.array(elm.nodes[0].coords) + elm.geomtransf.offset_i
    p_j = np.array(elm.nodes[1].coords) + elm.geomtransf.offset_j
    len_clr = np.linalg.norm(p_i - p_j)

    t_vec = np.linspace(0.00, len_clr, num=num_points)

    nx_vec = - t_vec * w_x - n_i
    qy_vec = t_vec * w_y + qy_i
    qz_vec = t_vec * w_z + qz_i
    tx_vec = np.full(num_points, -t_i)
    mz_vec = t_vec**2 * 0.50 * w_y + t_vec * qy_i - mz_i
    my_vec = t_vec**2 * 0.50 * w_z + t_vec * qz_i + my_i

    if as_tuple:
        return (nx_vec, qy_vec, qz_vec,
                tx_vec, mz_vec, my_vec)
    else:
        dframe = pd.DataFrame.from_dict(
            {'nx': nx_vec,
             'qy': qy_vec,
             'qz': qz_vec,
             'tx': tx_vec,
             'mz': mz_vec,
             'my': my_vec}
        )
        return dframe
