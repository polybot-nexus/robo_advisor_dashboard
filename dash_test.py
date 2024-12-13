import base64
from io import BytesIO

import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash import dash_table, dcc, html

# import imageio
from dash.dependencies import Input, Output
from PIL import Image

app = dash.Dash(__name__)


# Helper functions to process images
def find_film_image(id):
    try:
        img = imageio.imread(
            f'C:/Users/Public/robot/OECT_demo/samples/media/{id}_raw_annealed_film.jpg'
        )
        img = Image.fromarray(img)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return buffered.getvalue()
    except:
        print(id)
        return None


def image_base64(im):
    if im is None:
        return None
    return base64.b64encode(im).decode()


def image_formatter(id):
    image_data = find_film_image(id)
    if image_data:
        return f'<img src="data:image/jpeg;base64,{image_base64(image_data)}" width="150"/>'
    else:
        return "No image"


# Sample DataFrame (replace with actual data)
df = pd.DataFrame(
    {
        'ID': [1, 2, 3],
        'thickness': [10, 20, 30],
        'coating_on_top.sol_label': ['Sol1', 'Sol2', 'Sol3'],
        'coating_on_top.vel': [5, 10, 15],
        'coating_on_top.T': [100, 200, 300],
        'coating_on_top.substrate_label': ['Sub1', 'Sub2', 'Sub3'],
        'transconductance': [0.5, 0.7, 0.9],
        '#': [1, 2, 3],
    }
)


# Style Plot Function
def style_plot(fig):
    fig.update_traces(marker={'size': 7})
    fig.update_layout(plot_bgcolor='white')
    fig.update_layout(
        legend={
            'orientation': 'h',
            'yanchor': "bottom",
            'y': 1.02,
            'xanchor': "left",
            'x': 0,
        }
    )
    fig.update_layout(legend_title_text='')
    axis_setup = {
        'title_font': {'size': 18},
        'mirror': True,
        'ticks': 'outside',
        'showline': True,
        'linecolor': 'grey',
        'gridcolor': 'lightgrey',
        'zeroline': False,
    }
    fig.update_xaxes(**axis_setup)
    fig.update_yaxes(**axis_setup)


# Layout for Dash App
app.layout = html.Div(
    children=[
        html.H1("OECT Data Dashboard"),
        # Table of All Samples
        html.Div(
            [
                html.H2("Table of All Samples"),
                dash_table.DataTable(
                    id='sample-table',
                    columns=[
                        {"name": 'ID', "id": 'ID'},
                        {"name": 'Thickness', "id": 'thickness'},
                        {"name": 'Transconductance', "id": 'transconductance'},
                        {
                            "name": 'Film Image',
                            "id": 'image',
                            "presentation": "markdown",
                        },
                    ],
                    data=df.to_dict('records'),
                    style_cell={'textAlign': 'left'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold',
                    },
                ),
            ]
        ),
        # Graphs
        html.Div(
            [
                html.Div(
                    [
                        dcc.Markdown("#### Transconductance"),
                        dcc.Graph(id='transconductance-plot'),
                    ],
                    style={'width': '48%', 'display': 'inline-block'},
                ),
                html.Div(
                    [
                        dcc.Markdown("#### Transconductance of Devices"),
                        dcc.Graph(id='device-plot'),
                    ],
                    style={'width': '48%', 'display': 'inline-block'},
                ),
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Markdown("#### Coating Speed"),
                        dcc.Graph(id='coating-speed-plot'),
                    ],
                    style={'width': '48%', 'display': 'inline-block'},
                ),
                html.Div(
                    [
                        dcc.Markdown("#### Coating Temperature"),
                        dcc.Graph(id='coating-temp-plot'),
                    ],
                    style={'width': '48%', 'display': 'inline-block'},
                ),
            ]
        ),
    ]
)


# Callbacks for dynamic updating
@app.callback(
    Output('transconductance-plot', 'figure'),
    Output('device-plot', 'figure'),
    Output('coating-speed-plot', 'figure'),
    Output('coating-temp-plot', 'figure'),
    Input('sample-table', 'data'),
)
def update_graphs(rows):
    df = pd.DataFrame(rows)

    # Transconductance Plot
    fig_trans = px.scatter(
        data_frame=df,
        x='#',
        y='transconductance',
        color='coating_on_top.substrate_label',
    )
    style_plot(fig_trans)

    # Device Plot
    fig_device = px.scatter(
        data_frame=df,
        x='thickness',
        y='transconductance',
        color='coating_on_top.substrate_label',
    )
    style_plot(fig_device)

    # Coating Speed Plot
    fig_speed = px.scatter(
        data_frame=df,
        x='coating_on_top.vel',
        y='transconductance',
        color='coating_on_top.substrate_label',
    )
    style_plot(fig_speed)

    # Coating Temperature Plot
    fig_temp = px.scatter(
        data_frame=df,
        x='coating_on_top.T',
        y='transconductance',
        color='coating_on_top.substrate_label',
    )
    style_plot(fig_temp)

    return fig_trans, fig_device, fig_speed, fig_temp


if __name__ == '__main__':
    app.run_server(debug=True)
