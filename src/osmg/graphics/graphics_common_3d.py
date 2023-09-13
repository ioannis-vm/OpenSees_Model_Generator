"""
Common utility functions used for 3D data visualization
https://plotly.com/python/reference/
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

import plotly.graph_objects as go  # type: ignore
import numpy as np

# pio.renderers.default = 'browser'

# used for production plots
node_marker = {
    'fixed': ("square", 8),
    'release': ("circle-open", 6),
    'free': ("circle", 3),
    'parent': ("circle-open", 15),
    'internal': ("x", 2),
}

# # used for development plots
# node_marker = {
#     "fixed": ("square", 10),
#     "free": ("circle", 5),
#     "parent": ("circle-open", 20),
#     "internal": ("x", 3),
#     "release": ("circle-open", 15),
# }


def global_layout(mdl, camera=None):
    """
    Some general definitions that are often needed.

    """

    # get a proper boudning box form the model
    ref_len = mdl.reference_length()
    p_min, p_max = mdl.bounding_box(padding=2.0 * ref_len)

    # view_type = "orthographic"
    view_type = "perspective"
    if not camera:
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.0, y=1.00, z=0.0),
            projection={"type": view_type},
        )
    ref_len = np.linalg.norm(p_max - p_min)  # used in aspect ratio calcs
    return go.Layout(
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            bgcolor="white",
            camera=camera,
            xaxis=dict(range=[p_min[0], p_max[0]], autorange=False),
            yaxis=dict(range=[p_min[1], p_max[1]], autorange=False),
            zaxis=dict(range=[p_min[2], p_max[2]], autorange=False),
            # it's interesting that we have to do this to get the
            # axes aspect ratio right.. but hey! it works
            aspectratio=dict(
                x=(p_max[0] - p_min[0]) / (ref_len / 4.0),
                y=(p_max[1] - p_min[1]) / (ref_len / 4.0),
                z=(p_max[2] - p_min[2]) / (ref_len / 4.0),
            )
            # note:
            # aspectmode='data': was causing issues with
            # the camera 'moving' across different animation frames
        )
    )


if __name__ == "__main__":
    pass
