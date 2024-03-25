import jax.numpy as jnp
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_player as dp
import plotly.express as px
import numpy as np
import pandas as pd
from types import SimpleNamespace
import plotly.graph_objs as go

Cfg = SimpleNamespace()
Cfg.template = "simple_white"
Cfg.xaxis = {"showgrid": True}
Cfg.yaxis = {"showgrid": True}
Cfg.markers = False
# Cfg.line_color = "#4dd2ff"
Cfg.line_color = "#0072BD"
# Cfg.text_color = "#7FDBFF"
Cfg.text_color = "#000000"
Cfg.page_background_color = "#ffffff"
Cfg.figure_background_color = "#ffffff"

app = Dash(__name__)

app.layout = html.Div(
    style={"backgroundColor": Cfg.page_background_color },
    children=[
        html.H1(
            children="Experiment Monitor",
            style={"textAlign": "center", "color": Cfg.text_color},
        ),
        dp.DashPlayer(
            url="static/video.webm",
            controls=True,
            loop=True,
            playing=True,
            width="33%",
            height="33%",
        ),
        dcc.Graph(figure={}, id="controls-and-graph"),
        dcc.Interval(id="interval-component", interval=1 * 1000, n_intervals=0),
    ],
)


@callback(
    Output(component_id="controls-and-graph", component_property="figure"),
    Input(component_id="interval-component", component_property="n_intervals"),
)
def update_graph(col_chosen):
    df_mean = pd.read_csv("mean.csv")
    df_std = pd.read_csv("std.csv")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
            x=np.arange(0,len(df_mean)),
            y=df_mean['# Mean'],
            mode='lines',
        )
    )

    fig.add_trace(go.Scatter(
            x=np.arange(0,len(df_mean)),
            y=df_mean['# Mean']-df_std['# Std'],
            showlegend=False,
            mode='lines',
            line=dict(width=0),
        )
    )

    fig.add_trace(go.Scatter(
            x=np.arange(0,len(df_mean)),
            y=df_mean['# Mean']+df_std['# Std'],
            showlegend=False,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0, 114, 189, 0.3)',
            fill='tonexty',
        )
    )

    # fig.update_layout(
    #     plot_bgcolor=Cfg.figure_background_color ,
    #     paper_bgcolor=Cfg.figure_background_color ,
    #     font_color=Cfg.text_color,
    #     showlegend=False,
    #     xaxis=Cfg.xaxis,
    #     yaxis=Cfg.yaxis,
    # )
    return fig


app.run(debug=True)
