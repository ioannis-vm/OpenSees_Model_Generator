"""
Common utility functions used for 3D data visualization
https://plotly.com/python/reference/
"""
#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

import plotly.graph_objects as go  # type: ignore
# pio.renderers.default = 'browser'

# # used for production plots
# node_marker = {
#     'fixed': ("square", 8),
#     'pinned': ("circle-open", 2),
#     'free': ("circle", 3),
#     'parent': ("circle-open", 15),
#     'internal': ("x", 2),
#     '111011': ("circle-open", 3),
# }

# used for development plots
node_marker = {
    'fixed': ("square", 10),
    'free': ("circle", 5),
    'parent': ("circle-open", 20),
    'internal': ("x", 3),
    'release': ('circle-open', 15)
}


def global_layout(camera=None):
    """
    Some general definitions needed often
    """
    # view_type = "orthographic"
    view_type = "perspective"
    if not camera:
        camera = dict(
                       up=dict(x=0, y=0, z=1),
                       center=dict(x=0, y=0, z=0),
                       eye=dict(x=1.25, y=1.25, z=1.25),
                       projection={
                           "type": view_type
                       }
                   )
    return go.Layout(
        scene=dict(aspectmode='data',
                   xaxis_visible=False,
                   yaxis_visible=False,
                   zaxis_visible=False,
                   bgcolor='black',
                   camera=camera
                   )
    )


if __name__ == "__main__":
    pass
