"""
Common utility functions used for 3D data visualization
https://plotly.com/python/reference/
"""
#   __                 UC Berkeley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   April 2021
#
# https://github.com/ioannis-vm/OpenSeesPy_Building_Modeler/

from utility.graphics import common
import plotly.graph_objects as go
import plotly.io as pio
# pio.renderers.default = 'browser'

node_marker = {
    'fixed': ("square", 10),
    'pinned': ("circle-open", 10),
    'free': ("circle", 5),
    'master': ("circle-open", 15)
}


def global_layout():
    """
    Some general definitions needed often
    """
    return go.Layout(
        scene=dict(aspectmode='data',
                   xaxis_visible=False,
                   yaxis_visible=False,
                   zaxis_visible=False,
                   ),
        showlegend=False
    )


if __name__ == "__main__":
    pass
