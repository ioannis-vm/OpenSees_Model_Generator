"""
Library for consistent all-purpose 2D plots
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ / 
# / /_/ / / / / / / /_/ /_/  
# \____/_/ /_/ /_/\__, (_)   
#                /____/      
#                            
# https://github.com/ioannis-vm/OpenSees_Model_Generator

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


def line_plot_interactive(title_text, x, y, mode,
                          xlab="x", xunit=None, xhoverformat='.0f',
                          ylab="y", yunit=None, yhoverformat='.0f'):
    assert len(x) == len(y), "Dimensions don't match"
    assert mode in ['spline+markers', 'line'], \
        "mode can either be `spline+markers` or `line`"

    if mode == 'line':
        lshape = 'linear'
        lmode = 'lines'
        lwidth = 2
    elif mode == 'spline+markers':
        lshape = 'spline'
        lmode = 'lines+markers'
        lwidth = 4
    else:
        raise ValueError("oops! This should never run. Strange.")

    if xunit:
        xtitle = xlab + ' (' + xunit + ')'
    else:
        xtitle = xlab
    if yunit:
        ytitle = ylab + ' (' + yunit + ')'
    else:
        ytitle = ylab

    num_points = len(x)
    indices = np.array([i for i in range(num_points)])
    my_hovertemplate = \
        'XY value pair: %{customdata:d}<br>' + \
        xlab + ' = %{x: ' + xhoverformat + '} '
    if xunit:
        my_hovertemplate += xunit + '<br>'
    else:
        my_hovertemplate += '<br>'
    my_hovertemplate += ylab+' = %{y:' + yhoverformat + '} '
    if yunit:
        my_hovertemplate += yunit + '<extra></extra>'
    else:
        my_hovertemplate += '<extra></extra>'
    fig = go.Figure(
        data=go.Scatter(
            x=x,
            y=y,
            mode=lmode,
            line_shape=lshape,
            line=dict(color=C_LINE, width=lwidth),
            marker=dict(color=C_BACKGROUND,
                        size=10,
                        line=dict(
                            color=C_LINE,
                            width=4)),
            customdata=indices,
            hovertemplate=my_hovertemplate
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
        xaxis_title=xtitle,
        yaxis_title=ytitle,
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
