"""
Library for consistent all-purpose 2D plots
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

#############
# Resources #
#############
#
# https://coolors.co/  <- amazing color palette generator
#

import sys
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

nparr = npt.NDArray[np.float64]


C_LINE = "#6F1D1B"
C_AXIS_LINE = "#432818"
C_TITLE_COLOR = "#432818"
C_LABEL_COLOR = "#432818"
C_GRID = "#BB9457"
C_BACKGROUND = "#FFE6A7"
C_HOVERLABEL_BG = "#BB9457"


def line_plot_interactive(
    title_text,
    x_vals,
    y_vals,
    mode,
    xlab="x",
    xunit=None,
    xhoverformat=".0f",
    ylab="y",
    yunit=None,
    yhoverformat=".0f",
):
    """
    Gneric polty line plot
    """
    assert len(x_vals) == len(y_vals), "Dimensions don't match"
    assert mode in [
        "spline+markers",
        "line",
    ], "mode can either be `spline+markers` or `line`"

    if mode == "line":
        lshape = "linear"
        lmode = "lines"
        lwidth = 2
    elif mode == "spline+markers":
        lshape = "spline"
        lmode = "lines+markers"
        lwidth = 4
    else:
        raise ValueError("oops! This should never run. Strange.")

    if xunit:
        xtitle = f"{xlab} ({xunit})"
    else:
        xtitle = xlab
    if yunit:
        ytitle = f"{ylab} ({yunit})"
    else:
        ytitle = ylab

    num_points = len(x_vals)
    indices: nparr = np.array([range(num_points)])
    my_hovertemplate = (
        "XY value pair: %{customdata[0]:d}<br>"
        + xlab
        + " = %{x: "
        + xhoverformat
        + "} "
    )
    if xunit:
        my_hovertemplate += xunit + "<br>"
    else:
        my_hovertemplate += "<br>"
    my_hovertemplate += ylab + " = %{y:" + yhoverformat + "} "
    if yunit:
        my_hovertemplate += yunit + "<extra></extra>"
    else:
        my_hovertemplate += "<extra></extra>"
    fig = go.Figure(
        data=go.Scatter(
            x=x_vals,
            y=y_vals,
            mode=lmode,
            line_shape=lshape,
            line=dict(color=C_LINE, width=lwidth),
            marker=dict(
                color=C_BACKGROUND, size=10, line=dict(color=C_LINE, width=4)
            ),
            customdata=np.reshape(indices, (-1, 1)),
            hovertemplate=my_hovertemplate,
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
            ticks="outside",
            tickcolor=C_GRID,
            tickfont=dict(
                family="Cambria",
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
            ticks="outside",
            tickcolor=C_GRID,
            tickfont=dict(
                family="Cambria",
                size=14,
                color=C_AXIS_LINE,
            ),
        ),
        showlegend=False,
        plot_bgcolor=C_BACKGROUND,
    )
    annotations = [
        dict(
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.00,
            xanchor="left",
            yanchor="bottom",
            text=title_text,
            font=dict(family="Cambria", size=28, color=C_TITLE_COLOR),
            showarrow=False,
        )
    ]
    fig.update_layout(annotations=annotations)
    fig.update_layout(
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        font=dict(family="Cambria", size=18, color=C_LABEL_COLOR),
        hoverlabel=dict(
            bgcolor=C_HOVERLABEL_BG, font_size=15, font_family="Cambria"
        ),
    )
    if "pytest" not in sys.modules:
        fig.show()


if __name__ == "__main__":
    pass


# # (Test)
# x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# y = [0, 10, 20, 30, 35, 38, 40, 41, 42, 43]

# line_plot_interactive('Pushover\nDirection: Y',
#                       xlab='Displacement', ylab='Base Shear',
#                       xunit='in', yunit='lb', x_vals=x, y_vals=y,
#                       mode='spline+markers')
