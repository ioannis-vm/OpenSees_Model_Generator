"""
Utility functions.

Utility functions for handling ground motion files and response
spectra.
"""

import re
from pathlib import Path

import numpy as np
import numpy.typing as npt

numpy_array = npt.NDArray[np.float64]


def import_PEER(filename: str) -> numpy_array:  # noqa: N802
    """
    Import a PEER ground motion.

    Import a ground motion record from a specified PEER ground
    motion record file.
    Output is a two column matrix of time - acceleration pairs.
    Acceleration is in [g] units.

    Returns:
      The data in the file.
    """
    # Get all data except for the last line, where it may have fewer
    # columns and cause an error
    a_g = np.genfromtxt(filename, skip_header=4, skip_footer=1)
    # Manually read the last line and append
    with Path(filename).open(encoding='utf-8') as file:
        lines = file.readlines()
        last_line = lines[-1]
    last = np.fromstring(last_line, sep='  ')
    a_g = np.append(a_g, last)

    # Read metadata
    with Path(filename).open(encoding='utf-8') as file:
        line_containing_units = 2
        line_containing_number_of_points = 3
        for i, line in enumerate(file):
            if i == line_containing_units:
                # Units
                units = (line.split(sep=' ')[-1]).strip()
            elif i == line_containing_number_of_points:
                # Number of points
                npts = int(re.sub(r'NPTS=\s+', '', line.split(sep=', ')[0]))
                # Time step
                tmp = re.sub(r'DT=\s+', '', line.split(sep=', ')[1])
                tmp = re.sub(r'\s* SEC', '', tmp)
                tmp = tmp.replace('SEC', '')  # some files have no space
                d_t = float(tmp)
            elif i > line_containing_number_of_points:
                break

    # Assert correct number of points and units
    assert npts == len(
        a_g
    ), 'Number of points reported in file does not match recovered points'
    assert units == 'G', "Expected file to be in G units, but it isn't"

    # Obtain the corresponding time values
    t = np.array([x * d_t for x in range(npts)])

    # Return data in the form of a matrix
    return np.column_stack((t, a_g))


def response_spectrum(
    th: numpy_array, dt: float, zeta: float, num_points: int = 200
) -> numpy_array:
    """
    Calculate a  response spectrum.

    Calculate the linear response spectrum of an acceleration
    time history of fixed time interval dt and values given in vector
    th, and damping ratio zeta.
    n_Pts is the number of log-spaced points of the response spectrum

    Returns:
      The response spectrum.
    """
    T = np.logspace(  # noqa: N806
        -2, 1, num_points - 1
    )  # -1 because we also include PGA @ T=0s
    # we may have to upsample the ground motion time history
    # to ensure convergence of the central difference method
    if dt > 0.1 * T[0]:
        t_max = float(len(th)) * dt
        upscale = dt / (0.1 * T[0])
        old_ts = np.linspace(0, t_max, num=len(th))
        new_ts = np.linspace(0, t_max, num=int(upscale + 1.0) * len(th))
        th = np.interp(new_ts, old_ts, th)
        dt = new_ts[1] - new_ts[0]
        assert dt < 0.1 * T[0]
    omega = 2 * np.pi / T
    c = 2 * zeta * omega
    k = omega**2
    n = len(th)
    # Initial calculations
    u = np.full(len(T), 0.00)  # initialize arrays
    u_prev = np.full(len(T), 0.00)
    umax = np.full(len(T), 0.00)
    khut = 1.00 / dt**2 + c / (2.0 * dt)  # initial calcs
    alpha = 1.00 / dt**2 - c / (2.0 * dt)
    beta = k - 2.0 / dt**2
    for i in range(1, n):
        phut = -th[i] - alpha * u_prev - beta * u
        u_prev = u
        u = phut / khut  # update step
        # update maximum displacements
        umax[np.abs(u) > umax] = np.abs(u[np.abs(u) > umax])
    # Determine pseudo-spectral acceleration
    sa = umax * omega**2
    # rs = np.column_stack((T, sa))  # not yet
    # Include T = 0 s ~ PGA
    Ts = np.concatenate((np.array([0.00]), T))  # noqa: N806
    sas = np.concatenate((np.array([np.max(np.abs(th))]), sa))
    return np.column_stack((Ts, sas))


def code_spectrum(
    t_array: numpy_array, s_s: float, s_1: float, t_long: float = 8.00
) -> numpy_array:
    """
    Generate a simplified ASCE code response spectrum.

    Returns:
      The response spectrum.
    """
    num_vals = len(t_array)
    code_sa = np.full(num_vals, 0.00)
    T_short = s_1 / s_s  # noqa: N806
    T_zero = 0.20 * T_short  # noqa: N806
    code_sa[t_array <= T_short] = s_s
    code_sa[t_array >= t_long] = s_1 * t_long / t_array[t_array >= t_long] ** 2
    sel = np.logical_and(t_array > T_short, t_array < t_long)
    code_sa[sel] = s_1 / t_array[sel]
    code_sa[t_array < T_zero] = s_s * (
        0.40 + 0.60 * t_array[t_array < T_zero] / T_zero
    )
    return np.column_stack((t_array, code_sa))


if __name__ == '__main__':
    pass
