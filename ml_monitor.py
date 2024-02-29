import jax.numpy as jnp
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_player as dp
import plotly.express as px
import numpy as np
import pandas as pd
from types import SimpleNamespace

Cfg = SimpleNamespace()
Cfg.template = "simple_white"
Cfg.xaxis = {"showgrid": True}
Cfg.yaxis = {"showgrid": True}
Cfg.markers = False
Cfg.line_color = "#4dd2ff"

app = Dash(__name__)

colors = {"background": "#111111", "text": "#7FDBFF"}
app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1(
            children="Experiment Monitor",
            style={"textAlign": "center", "color": colors["text"]},
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
    df = pd.read_csv("data.csv")

    fig = px.line(
        df,
        title="Reward",
        template=Cfg.template,
        markers=Cfg.markers,
        color_discrete_map={"# Reward": Cfg.line_color},
        width=400,
        height=400,
    )

    fig.update_layout(
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
        font_color=colors["text"],
        showlegend=False,
        xaxis=Cfg.xaxis,
        yaxis=Cfg.yaxis,
    )  # , x='continent', y=col_chosen, histfunc='avg')
    return fig


app.run(debug=True)
