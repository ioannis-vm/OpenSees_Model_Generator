"""
Common utility functions used for 3D data visualization.

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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go  # type: ignore

if TYPE_CHECKING:
    from osmg.model import Model

# pio.renderers.default = 'browser'

# used for production plots
node_marker = {
    'fixed': ('square', 8),
    'release': ('circle-open', 6),
    'free': ('circle', 3),
    'parent': ('circle-open', 15),
    'internal': ('x', 2),
}

# # used for development plots
# node_marker = {
#     "fixed": ("square", 10),
#     "free": ("circle", 5),
#     "parent": ("circle-open", 20),
#     "internal": ("x", 3),
#     "release": ("circle-open", 15),
# }


def global_layout(mdl: Model, camera: dict[str, object] | None = None) -> go.Layout:
    """
    Make some general definitions that are often needed.

    Returns:
      A general layout.
    """
    # get a proper bounding box form the model
    ref_len = mdl.reference_length()
    p_min, p_max = mdl.bounding_box(padding=2.0 * ref_len)

    # view_type = "orthographic"
    view_type = 'perspective'
    if not camera:
        camera = {
            'up': {'x': 0, 'y': 0, 'z': 1},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'eye': {'x': 0.7, 'y': 1.00, 'z': 0.30},
            'projection': {'type': view_type},
        }
    ref_len = np.linalg.norm(p_max - p_min)  # used in aspect ratio calcs
    return go.Layout(
        scene={
            'xaxis_visible': False,
            'yaxis_visible': False,
            'zaxis_visible': False,
            'bgcolor': 'white',
            'camera': camera,
            'xaxis': {'range': [p_min[0], p_max[0]], 'autorange': False},
            'yaxis': {'range': [p_min[1], p_max[1]], 'autorange': False},
            'zaxis': {'range': [p_min[2], p_max[2]], 'autorange': False},
            # it's interesting that we have to do this to get the
            # axes aspect ratio right.. but hey! it works
            'aspectratio': {
                'x': (p_max[0] - p_min[0]) / (ref_len / 4.0),
                'y': (p_max[1] - p_min[1]) / (ref_len / 4.0),
                'z': (p_max[2] - p_min[2]) / (ref_len / 4.0),
            },
            # note:
            # aspectmode='data': was causing issues with
            # the camera 'moving' across different animation frames
        }
    )


if __name__ == '__main__':
    pass
