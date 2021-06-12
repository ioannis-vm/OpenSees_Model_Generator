"""
Library for consistent all-purpose 2D plots
"""

#   __                 UC Berkeley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   April 2021
#
# https://github.com/ioannis-vm/OpenSeesPy_Building_Modeler/

#############
# Resources #
#############
#
# https://coolors.co/  <- amazing color palette generator
#

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

C_LINE = "#6F1D1B"
C_AXIS_LINE = "#432818"
C_TITLE_COLOR = "#432818"
C_LABEL_COLOR = "#432818"
C_GRID = "#BB9457"
C_BACKGROUND = "#FFE6A7"
C_HOVERLABEL_BG = "#BB9457"


def line_plot_interactive(title_text, xlab, ylab, xunit, yunit, x, y):
    assert len(x) == len(y), "Dimensions don't match"
    num_points = len(x)
    indices = np.array([i for i in range(num_points)])
    fig = go.Figure(
        data=go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            line_shape='spline',
            line=dict(color=C_LINE, width=4),
            marker=dict(color=C_BACKGROUND,
                        size=10,
                        line=dict(
                            color=C_LINE,
                            width=4)),
            customdata=indices,
            hovertemplate='Step: %{customdata:d}<br>' +
            xlab+' = %{x:.0f} ' + xunit + '<br>' +
            ylab+' = %{y:.0f} ' + yunit +
            '<extra></extra>'
        )
    )
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            gridcolor=C_GRID,
            showticklabels=True,
            linecolor=C_BACKGROUND,
            zerolinecolor=C_AXIS_LINE,
            linewidth=0,
            ticks='outside',
            tickcolor=C_GRID,
            tickfont=dict(
                family='Cambria',
                size=14,
                color=C_AXIS_LINE,
            ),
        ),
        yaxis=dict(
            showline=True,
            showgrid=True,
            gridcolor=C_GRID,
            showticklabels=True,
            linecolor=C_BACKGROUND,
            zerolinecolor=C_AXIS_LINE,
            linewidth=0,
            ticks='outside',
            tickcolor=C_GRID,
            tickfont=dict(
                family='Cambria',
                size=14,
                color=C_AXIS_LINE,
            ),
        ),
        showlegend=False,
        plot_bgcolor=C_BACKGROUND
    )
    annotations = [
        dict(
            xref='paper',
            yref='paper',
            x=0.0,
            y=1.00,
            xanchor='left', yanchor='bottom',
            text=title_text,
            font=dict(family='Cambria',
                      size=28,
                      color=C_TITLE_COLOR),
            showarrow=False
        )
    ]
    fig.update_layout(annotations=annotations)
    fig.update_layout(
        xaxis_title=xlab + ' (' + xunit + ')',
        yaxis_title=ylab + ' (' + yunit + ')',
        font=dict(
            family='Cambria',
            size=18,
            color=C_LABEL_COLOR
        ),
        hoverlabel=dict(
            bgcolor=C_HOVERLABEL_BG,
            font_size=15,
            font_family='Cambria'
        )
    )
    fig.show()


if __name__ == '__main__':
    pass


# # (Test)
# x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# y = [0, 10, 20, 30, 35, 38, 40, 41, 42, 43]

# line_plot_interactive('Pushover\nDirection: Y',
#                       'Displacement', 'Base Shear',
#                       'in', 'lb', x, y)
