import jax.numpy as jnp
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_player as dp
import plotly.express as px
import pandas as pd
from types import SimpleNamespace

Cfg = SimpleNamespace()
Cfg.template = "simple_white"
Cfg.xaxis = {"showgrid": False}
Cfg.yaxis = {"showgrid": False}
Cfg.show_grid = False
Cfg.markers = False
Cfg.line_color = "RebeccaPurple"

app = Dash(__name__)


app.layout = html.Div(
    children=[
        dp.DashPlayer(
            url="static/video.webm",
            controls=True,
            loop=True,
            playing=True,
            width="33%",
            height="33",
        ),
        dcc.Graph(figure={}, id="controls-and-graph"),
        dcc.Interval(id="interval-component", interval=0.1 * 1000, n_intervals=0),
    ]
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
    )
    fig.update_layout(
        showlegend=False,
        xaxis=Cfg.xaxis,
        yaxis=Cfg.yaxis,
    )  # , x='continent', y=col_chosen, histfunc='avg')
    return fig


app.run(debug=True)
