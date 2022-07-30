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
from .. import transformations
from ..solver import Analysis
from ..ops.element import elasticBeamColumn
from ..ops.element import dispBeamColumn
from ..solver import ModalResponseSpectrumAnalysis

nparr = npt.NDArray[np.float64]


def basic_forces(
        anl: Analysis,
        case_name: str,
        step: int,
        elm: elasticBeamColumn | dispBeamColumn,
        num_points: int,
        as_tuple: bool = False) -> nparr:

    if isinstance(anl, ModalResponseSpectrumAnalysis):
        forces = anl.combined_basic_forces(elm.uid)
        wx, wy, wz = (0.00, 0.00, 0.00)
    else:
        forces = anl.results[case_name].element_forces.registry[
            elm.uid][step]
        wx, wy, wz = anl.load_cases[case_name].line_element_udl.registry[elm.uid].val

    ni, qyi, qzi = forces[0:3]
    ti, myi, mzi = forces[3:6]

    
    p_i = np.array(elm.eleNodes[0].coords) + elm.geomtransf.offset_i
    p_j = np.array(elm.eleNodes[1].coords) + elm.geomtransf.offset_j
    len_clr = np.linalg.norm(p_i - p_j)

    t = np.linspace(0.00, len_clr, num=num_points)

    nx_vec = - t * wx - ni
    qy_vec = t * wy + qyi
    qz_vec = t * wz + qzi
    tx_vec = np.full(num_points, -ti)
    mz_vec = t**2 * 0.50 * wy + t * qyi - mzi
    my_vec = t**2 * 0.50 * wz + t * qzi + myi

    if as_tuple:
        return (nx_vec, qy_vec, qz_vec,
                tx_vec, mz_vec, my_vec)
    else:
        df = pd.DataFrame.from_dict(
            {'nx': nx_vec,
             'qy': qy_vec,
             'qz': qz_vec,
             'tx': tx_vec,
             'mz': mz_vec,
             'my': my_vec}
        )
        return df
