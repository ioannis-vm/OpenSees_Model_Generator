"""
Utility functions for handling ground motion files and response
spectra.
"""

import re
import numpy as np
import numpy.typing as npt

nparr = npt.NDArray[np.float64]


def import_PEER(filename):
    """
    Import a ground motion record from a specified PEER ground
    motion record file.
    Output is a two column matrix of time - acceleration pairs.
    Acceleration is in [g] units.
    """

    # Get all data except for the last line, where it may have fewer
    # columns and cause an error
    ag = np.genfromtxt(filename, skip_header=4, skip_footer=1)
    # Manually read the last line and append
    with open(filename) as f:
        for line in f:
            pass
        last_line = line
    last = np.fromstring(last_line, sep='  ')
    ag = np.append(ag, last)

    # Read metadata
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 2:
                # Units
                units = (line.split(sep=' ')[-1]).strip()
            elif i == 3:
                # Number of points
                npts = int(re.sub('NPTS=\s+', '',  # noqa: W605
                                  line.split(sep=', ')[0]))  # noqa: W605
                # Time step
                tmp = re.sub(
                    'DT=\s+', '',  # noqa: W605
                    line.split(sep=', ')[1])
                tmp = re.sub('\s* SEC', '', tmp)  # noqa: W605
                tmp = tmp.replace('SEC', '')  # some files have no space
                dt = float(tmp)
            elif i > 3:
                break

    # Assert correct number of points and units
    assert npts == len(ag), \
        'Number of points reported in file does not match recovered points'
    assert units == 'G', \
        'Expected file to be in G units, but it isn\'t'

    # Obtain the corresponding time values
    t = np.array([x*dt for x in range(npts)])

    # Return data in the form of a matrix
    return np.column_stack((t, ag))


def response_spectrum(th, dt, zeta, n_Pts=200):
    """
    Calculate the linear response spectrum of an acceleration
    time history of fixed time interval dt and values given in vector
    th, and damping ratio zeta.
    n_Pts is the number of log-spaced points of the response spectrum
    """
    T = np.logspace(-2, 1, n_Pts-1)  # -1 becuase we also include PGA @ T=0s
    # we may have to upsample the ground motion time history
    # to ensure convergence of the central difference method
    if dt > 0.1*T[0]:
        t_max = float(len(th)) * dt
        upscale = dt/(0.1*T[0])
        old_ts = np.linspace(0, t_max, num=len(th))
        new_ts = np.linspace(0, t_max, num=int(upscale+1.0) * len(th))
        th = np.interp(new_ts, old_ts, th)
        dt = new_ts[1] - new_ts[0]
        assert (dt < 0.1 * T[0])
    omega = 2 * np.pi / T
    c = 2 * zeta * omega
    k = omega**2
    n = len(th)
    # Initial calculations
    u = np.full(len(T), 0.00)       # initialize arrays
    u_prev = np.full(len(T), 0.00)
    umax = np.full(len(T), 0.00)
    khut = 1.00/dt**2 + c/(2.*dt)   # initial calcs
    alpha = 1.00/dt**2 - c/(2.*dt)
    beta = k - 2./dt**2
    for i in range(1, n):
        phut = -th[i] - alpha*u_prev - beta*u
        u_prev = u
        u = phut/khut  # update step
        # update maximum displacements
        umax[np.abs(u) > umax] = np.abs(u[np.abs(u) > umax])
    # Determine pseudo-spectral acceleration
    sa = umax * omega**2
    # rs = np.column_stack((T, sa))  # not yet
    # Include T = 0 s ~ PGA
    Ts = np.concatenate((np.array([0.00]), T))
    sas = np.concatenate((np.array([np.max(np.abs(th))]), sa))
    rs = np.column_stack((Ts, sas))
    return rs


def code_spectrum(T_vals: nparr, Ss: float, S1: float,
                  Tl: float = 8.00) -> nparr:
    """
    Generate a simplified ASCE code response spectrum.
    """
    num_vals = len(T_vals)
    code_sa = np.full(num_vals, 0.00)
    T_short = S1 / Ss
    T_zero = 0.20 * T_short
    code_sa[T_vals <= T_short] = Ss
    code_sa[T_vals >= Tl] = S1 * Tl / T_vals[T_vals >= Tl]**2
    sel = np.logical_and(T_vals > T_short, T_vals < Tl)
    code_sa[sel] = S1 / T_vals[sel]
    code_sa[T_vals < T_zero] = (
        Ss * (0.40 + 0.60 * T_vals[T_vals < T_zero] / T_zero))
    return np.column_stack((T_vals, code_sa))


if __name__ == '__main__':
    pass
