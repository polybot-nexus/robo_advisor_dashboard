import dash
import matplotlib
import pandas as pd
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State

matplotlib.use('Agg')
import base64
import json
from io import BytesIO

import dash_bootstrap_components as dbc
import imageio
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

from ml_models import (
    evaluate_ml_models,
    plot_whiskerplots,
    shapley_analysis_plotly,
)
from utils import (  # , create_plotly_trendline_fit, get_trendline_slope
    create_plotly_stock_market_plot,
)


def update_message_box(oect_data):
    import pandas_ta as ta

    df = oect_data.copy()
    df['SMA_10'] = ta.sma(df['transconductance'], length=10)
    df['EMA_10'] = ta.ema(df['transconductance'], length=10)
    df['EMA_20'] = ta.ema(df['transconductance'], length=20)
    df['SMA_200'] = ta.sma(df['transconductance'], length=200)
    df['SMA_50'] = ta.sma(df['transconductance'], length=50)
    df['SMA_20'] = ta.sma(df['transconductance'], length=20)
    df['RSI'] = ta.rsi(df['transconductance'], length=14)

    # slopes = get_trendline_slope(oect_data)
    if oect_data is None or len(oect_data) == 0:
        return "No data available."
    try:
        macd = ta.macd(df['transconductance'], fast=10, slow=20, signal=8)
        df = pd.concat([df, macd], axis=1)
        # print('MACD', df['MACDh_10_20_8'].values)
        if df['MACDh_10_20_8'].values[-1] < 0:
            return "The MACD indicator is showing a downward trajectory. Consider changing strategy by selecting different ML model or modify the parameter range."
    # if consecutive_declines(slopes) >= 3:
    #     return "The transconductance trendline slope is showing a downward trajectory. Consider Changing strategy."
    except:
        return "AI-advisor will display a message if any workflow modification is required."

    return "AI-advisor will display a message if any workflow modification is required."


def consecutive_declines(data):
    # Get the last 5 data of the slopes and check if they are keep descreasing
    if len(data) < 5:
        pass

    last_5 = data[-5:]
    declines = 0

    for i in range(1, len(last_5)):
        if last_5[i] < last_5[i - 1]:
            declines += 1

    return declines


# Helper functions to process images
def find_film_image(id):
    try:
        img = imageio.imread(f'assets/images/{id}_annealed_film.jpg')
        img = Image.fromarray(img)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return buffered.getvalue()
    except:
        # print(id)
        return None


# Helper functions to handle images
def find_film_image_path(id):
    """Finds the image path based on the given ID."""
    return f'assets/images/{id}_annealed_film.jpg'


def image_base64(image_path):
    """Encodes the image to Base64 format."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        return None


def create_image_link(id):
    """Creates a direct link to the image."""
    image_path = f'assets/images/{id}_annealed_film.jpg'
    return f'[View Image]({image_path})'


def image_formatter(id):
    image_data = find_film_image(id)
    if image_data:
        return f'data:image/jpeg;base64,{image_base64(image_data)}'
        # return f'<img src="data:image/jpeg;base64,{image_base64(image_data)}" width="150"/>'
    else:
        return "No image"


def encode_image(image_file):
    with open(image_file, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('ascii')
    return f'data:image/jpeg;base64,{encoded}'


data_file_path = 'datasets/oect_summary_posted_rf__plus_ml_combined.csv'
oect_data = pd.read_csv(
    'datasets/oect_summary_posted_rf__plus_ml_combined.csv'
)
oect_data['coating_on_top.sol_label'] = oect_data[
    'coating_on_top.sol_label'
].map(lambda x: x.lstrip('mg/ml').rstrip('mg/ml'))
oect_data['coating_on_top.substrate_label'] = oect_data[
    'coating_on_top.substrate_label'
].map(lambda x: x.lstrip('nm').rstrip('nm'))
oect_data['coating_on_top.sol_label'] = pd.to_numeric(
    oect_data['coating_on_top.sol_label']
)
oect_data['coating_on_top.substrate_label'] = pd.to_numeric(
    oect_data['coating_on_top.substrate_label']
)
oect_data['coating_on_top.vel'] = pd.to_numeric(
    oect_data['coating_on_top.vel']
)
oect_data['coating_on_top.T'] = pd.to_numeric(oect_data['coating_on_top.T'])
df = oect_data.copy()
df['image'] = df['ID'].apply(create_image_link)


# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Increase the timeout limit
server.config['PROPAGATE_EXCEPTIONS'] = True
server.config['WTF_CSRF_TIME_LIMIT'] = 3600

tabs_styles = {'height': '44px'}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
}

app.layout = html.Div(
    [
        dcc.Tabs(
            id="tabs-styled-with-inline",
            value='tab-1',
            children=[
                dcc.Tab(
                    label='AI advisor board',
                    value='tab-1',
                    style=tab_style,
                    selected_style=tab_selected_style,
                ),
                dcc.Tab(
                    label='Data visualization',
                    value='tab-2',
                    style=tab_style,
                    selected_style=tab_selected_style,
                ),
                # dcc.Tab(label='ML Model Evaluation and Feature Importance', value='tab-3', style=tab_style, selected_style=tab_selected_style),
            ],
            style=tabs_styles,
        ),
        html.Div(id='tabs-content-inline'),
        html.Div(id='hover-data-output', style={'display': 'none'}),
        dcc.Interval(
            id='page-load', interval=1, n_intervals=0, max_intervals=1
        ),
        # dcc.Loading(
        #     children=[
        #         html.Div(id='results-table-div'),
        #         dcc.Graph(id='feature-importance-plot')
        #     ],
        #     type="circle"
        # )
        # dcc.Interval(
        #     id='interval-refresh',
        #     interval=5*1000000,  # adjust the intervals
        #     n_intervals=0
        # )
    ]
)


@app.callback(
    Output('tabs-content-inline', 'children'),
    Input('page-load', 'n_intervals'),
    #   Input('interval-refresh', 'n_intervals'),
    Input('tabs-styled-with-inline', 'value'),
)
def render_content(n_intervals, tab):
    oect_data = pd.read_csv(
        'datasets/oect_summary_posted_rf__plus_ml_combined.csv'
    )
    oect_data['coating_on_top.sol_label'] = oect_data[
        'coating_on_top.sol_label'
    ].map(lambda x: x.lstrip('mg/ml').rstrip('mg/ml'))
    oect_data['coating_on_top.substrate_label'] = oect_data[
        'coating_on_top.substrate_label'
    ].map(lambda x: x.lstrip('nm').rstrip('nm'))
    oect_data['coating_on_top.sol_label'] = pd.to_numeric(
        oect_data['coating_on_top.sol_label']
    )
    oect_data['coating_on_top.substrate_label'] = pd.to_numeric(
        oect_data['coating_on_top.substrate_label']
    )
    oect_data['coating_on_top.vel'] = pd.to_numeric(
        oect_data['coating_on_top.vel']
    )
    oect_data['coating_on_top.T'] = pd.to_numeric(
        oect_data['coating_on_top.T']
    )
    oect_data['image'] = oect_data['ID'].apply(create_image_link)

    if tab == 'tab-1':
        return html.Div(
            [
                # Message box
                html.Div(
                    [
                        html.H3("Message Box", style={'textAlign': 'center'}),
                        html.Div(
                            id='message-box',
                            style={
                                'width': '60%',
                                'border': '2px solid blue',
                                'padding': '20px',
                                'border-radius': '5px',
                                'box-shadow': '2px 2px 10px rgba(0, 0, 0, 0.1)',
                                'background-color': '#f9f9f9',
                                'color': 'black',
                                'font-weight': 'bold',
                                'font-size': '18px',
                                'text-align': 'center',
                                'margin': 'auto',
                            },
                        ),
                        # Stockmarket trendline
                        html.H3(
                            "Trendline monitoring",
                            style={'textAlign': 'center', 'marginTop': '30px'},
                        ),  # , 'marginTop': '3px'
                        dcc.Graph(
                            id='stock-market-plot',
                            style={
                                'width': '100%',
                                'height': '600px',
                                'marginTop': '10px',
                            },  # 'marginBottom': '10px'
                        ),
                    ]  # , #style={'width': '100%'} #, 'display': 'flex'
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                # Title for the ML comparison table
                                html.H3(
                                    "ML Model Evaluation",
                                    style={'textAlign': 'center'},
                                ),
                                html.Div(
                                    id='results-table-div',
                                    style={
                                        'width': '100%',
                                        'display': 'block',
                                        'margin-top': '50px',
                                        'margin-left': '30px',
                                        'height': '500px',
                                        'overflow': 'auto',
                                    },
                                ),
                            ],
                            style={
                                'width': '48%',
                                'display': 'inline-block',
                                'verticalAlign': 'top',
                                'padding-left': '20px',
                            },
                        ),
                        # Feature importance plot with a title
                        html.Div(
                            [
                                html.H3(
                                    "SHAP Summary Plot",
                                    style={'textAlign': 'center'},
                                ),
                                dcc.Graph(
                                    id='feature-importance-plot',
                                    style={
                                        'width': '100%',
                                        'display': 'inline-block',
                                        'verticalAlign': 'top',
                                    },
                                ),
                            ],
                            style={
                                'width': '48%',
                                'display': 'inline-block',
                                'verticalAlign': 'top',
                            },
                        ),
                    ],
                    style={
                        'width': '100%',
                        'display': 'flex',
                        'justify-content': 'space-between',
                    },
                ),
            ]
        )

    elif tab == 'tab-2':
        return html.Div(
            [
                html.Div(
                    [
                        html.H2("Table of All Samples"),
                        dash_table.DataTable(
                            id='sample-table',
                            columns=[
                                {"name": 'ID', "id": 'ID'},
                                {
                                    "name": 'Coating Speed (mm/s)',
                                    "id": 'coating_on_top.vel',
                                },
                                {
                                    "name": 'Coating Temperature (oC)',
                                    "id": 'coating_on_top.T',
                                },
                                {
                                    "name": 'Concentration (mg/ml)',
                                    "id": 'coating_on_top.sol_label',
                                },
                                {
                                    "name": 'Substrate (nm)',
                                    "id": 'coating_on_top.substrate_label',
                                },
                                {"name": "μC*", "id": 'transconductance'},
                                {
                                    "name": 'Film Image',
                                    "id": 'image',
                                    "presentation": "markdown",
                                },
                            ],
                            data=oect_data.to_dict('records'),
                            style_table={
                                'height': '200px',
                                'width': '100%',
                                'overflowY': 'auto',
                                'overflowX': 'auto',
                            },
                            style_cell={
                                'textAlign': 'left',
                                'fontSize': '12px',
                                'padding': '5px',
                            },
                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold',
                                'fontSize': '12px',
                                'padding': '5px',
                            },
                            style_data_conditional=[
                                {
                                    'if': {'column_id': 'image'},
                                    'textAlign': 'center',
                                    'whiteSpace': 'normal',
                                    'height': 'auto',
                                }
                            ],
                        ),
                    ],
                    style={'width': '100%', 'display': 'inline-block'},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Markdown("#### Concentration"),
                                dcc.Graph(id='concentration-plot'),
                            ],
                            style={'width': '48%', 'display': 'inline-block'},
                        ),
                        html.Div(
                            [
                                dcc.Markdown("#### Substrate"),
                                dcc.Graph(id='substrate-plot'),
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


############################################################################################################


# Callbacks for dynamic updating of the scatterplots
@app.callback(
    Output('concentration-plot', 'figure'),
    Output('substrate-plot', 'figure'),
    Output('coating-speed-plot', 'figure'),
    Output('coating-temp-plot', 'figure'),
    Input('sample-table', 'data'),
)
def update_graphs(rows):
    df = pd.DataFrame(rows)

    # Concentration Plot
    fig_concentration = px.scatter(
        data_frame=df,
        x='coating_on_top.sol_label',
        y='transconductance',
        color='transconductance',
        color_continuous_scale='Blues',
    )
    style_plot(fig_concentration)
    fig_concentration.update_layout(
        xaxis_title="Concentration (mg/ml)",
        yaxis_title="μC* (F cm⁻¹ V⁻¹ s⁻¹)",
        plot_bgcolor='white',
        coloraxis_colorbar=dict(title="μC*"),
    )

    # Substrate Plot
    fig_substrate = px.scatter(
        data_frame=df,
        x='coating_on_top.substrate_label',
        y='transconductance',
        color='transconductance',
        color_continuous_scale='Blues',
        hover_data=[
            'ID',
            'coating_on_top.substrate_label',
            'transconductance',
        ],
    )
    style_plot(fig_substrate)
    fig_substrate.update_layout(
        xaxis_title="Substrate (nm)",
        yaxis_title="μC* (F cm⁻¹ V⁻¹ s⁻¹)",
        plot_bgcolor='white',
        coloraxis_colorbar=dict(title="μC*"),
    )

    # Coating Speed Plot
    fig_speed = px.scatter(
        data_frame=df,
        x='coating_on_top.vel',
        y='transconductance',
        color='transconductance',
        color_continuous_scale='Blues',
    )
    style_plot(fig_speed)
    fig_speed.update_layout(
        xaxis_title="Coating Speed (mm/s)",
        yaxis_title="μC* (F cm⁻¹ V⁻¹ s⁻¹)*",
        plot_bgcolor='white',
        coloraxis_colorbar=dict(title="μC*"),
    )

    # Coating Temperature Plot
    fig_temp = px.scatter(
        data_frame=df,
        x='coating_on_top.T',
        y='transconductance',
        color='transconductance',
        color_continuous_scale='Blues',
    )
    style_plot(fig_temp)
    fig_temp.update_layout(
        xaxis_title="Coating Temperature (°C)",
        yaxis_title="μC* (F cm⁻¹ V⁻¹ s⁻¹)",
        plot_bgcolor='white',
        coloraxis_colorbar=dict(title="μC*"),
    )

    return fig_concentration, fig_substrate, fig_speed, fig_temp


#############################################################################################################
# Callback to generate and display the ML models comparison plot


def style_plot(fig):
    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey'))
    )
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


# Image modal display callback
@app.callback(
    [Output("image-modal", "is_open"), Output("modal-image", "src")],
    [Input("sample-table", "active_cell"), Input("close-modal", "n_clicks")],
    [State("sample-table", "data"), State("image-modal", "is_open")],
)
def toggle_modal(active_cell, close_clicks, data, is_open):
    ctx = dash.callback_context

    if not ctx.triggered:
        return is_open, None

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "sample-table" and active_cell:
        row = active_cell["row"]
        image_id = data[row]["ID"]
        image_path = find_film_image_path(image_id)
        image_data = image_base64(image_path)
        if image_data:
            return True, f"data:image/jpeg;base64,{image_data}"

    if triggered_id == "close-modal":
        return False, None

    return is_open, None


def update_ml_models_table(filename):
    # Load results from the JSON file
    with open(filename, 'r') as f:
        results = json.load(f)

    model_order = [
        "Gaussian Process",
        "Linear Regression",
        "SVR",
        "Neural Net",
        "Random Forest",
        "AdaBoost",
    ]

    # Prepare table data
    table_data = [
        {
            "Model": model,
            "Average Test RMSE": f"{metrics['test_rmse_average']:.4f}",
            "Test RMSE Std Dev": f"{metrics['test_rmse_std']:.4f}",
            "Train-Test RMSE Diff": f"{abs(metrics['train_rmse_average'] - metrics['test_rmse_average']):.4f}",
        }
        for model, metrics in results.items()
    ]

    table_data.sort(key=lambda x: model_order.index(x["Model"]))

    min_rmse = min(table_data, key=lambda x: float(x["Average Test RMSE"]))[
        "Average Test RMSE"
    ]
    min_std_dev = min(table_data, key=lambda x: float(x["Test RMSE Std Dev"]))[
        "Test RMSE Std Dev"
    ]

    algorithm_colors = {
        "Random Forest": "mediumseagreen",
        "AdaBoost": "mediumseagreen",
        "Gaussian Process": "lightyellow",
        "SVR": "teal",
        "Linear Regression": "lightblue",
        "Neural Net": "teal",
    }

    style_data_conditional = [
        {
            'if': {'filter_query': f'{{Model}} = "{algo}"'},
            'backgroundColor': color,
            'color': 'black',
        }
        for algo, color in algorithm_colors.items()
    ]

    results_table = dash_table.DataTable(
        data=table_data,
        columns=[
            {"name": "Model", "id": "Model"},
            {"name": "Average Test RMSE", "id": "Average Test RMSE"},
            {"name": "Test RMSE Std Dev", "id": "Test RMSE Std Dev"},
            {"name": "Train-Test RMSE Diff", "id": "Train-Test RMSE Diff"},
        ],
        style_table={'overflowX': 'auto', 'width': '100%'},
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'fontSize': '16px',
        },
        style_header={'fontWeight': 'bold', 'fontSize': '18px'},
        style_data_conditional=style_data_conditional,
    )
    return [results_table]


def update_feature_importance_plot():
    # # Load precomputed SHAP plot from JSON
    with open("shap_plot.json", "r") as f:
        feature_importance_fig = pio.from_json(f.read())

    feature_importance_fig.update_traces(marker=dict(size=16, opacity=0.8))
    feature_importance_fig.add_vline(
        x=0,
        line=dict(color="black", width=2, dash="dash"),
    )
    feature_importance_fig.update_layout(
        title="",
        title_font=dict(size=1),
        xaxis=dict(
            title="SHAP Value",
            title_font=dict(size=16),
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            title="Feature", title_font=dict(size=16), tickfont=dict(size=16)
        ),
        margin=dict(l=50, r=50, t=20, b=40),
        height=500,
        width=800,
    )

    feature_importance_fig.update_layout(
        plot_bgcolor='rgba(245, 245, 245, 0.8)',
        paper_bgcolor='rgba(255, 255, 255, 0.8)',
        xaxis=dict(
            title="SHAP Value",
            title_font=dict(size=20),
            tickfont=dict(size=20),
            gridcolor='white',
        ),
        yaxis=dict(
            title="Feature",
            title_font=dict(size=20),
            tickfont=dict(size=20),
            gridcolor='white',
        ),
        coloraxis_colorbar=dict(
            title="Feature Value",
            title_font=dict(size=18),
            tickfont=dict(size=16),
        ),
        margin=dict(l=50, r=50, t=20, b=40),
        height=500,
        width=800,
    )

    feature_importance_fig.update_traces(marker=dict(size=20, opacity=0.8))

    feature_importance_fig.add_vline(
        x=0,
        line=dict(color="black", width=2, dash="dash"),
    )
    return feature_importance_fig


@app.callback(
    [
        Output('stock-market-plot', 'figure'),
        Output('results-table-div', 'children'),
        Output('feature-importance-plot', 'figure'),
        Output('message-box', 'children'),
    ],
    [Input('stock-market-plot', 'relayoutData')],
    prevent_initial_call=True,
)
def update_charts(relayout_data):
    print("relayout_data:", relayout_data)  # Debug print
    if relayout_data and 'xaxis2.range' in relayout_data:
        start_idx = int(float(relayout_data['xaxis2.range'][0]))
        end_idx = int(float(relayout_data['xaxis2.range'][1]))
        filtered_data = oect_data.iloc[start_idx:end_idx]
        print('Range selected:', start_idx, '-', end_idx)
        #    print('Filtered data shape:', filtered_data)

        fig = create_plotly_stock_market_plot(filtered_data)
        fig.update_layout(xaxis_range=[start_idx, end_idx])

        model_results = update_ml_models_table(
            f"ml_model_weights/results_{start_idx}_{end_idx}.json"
        )
        #    model_results = evaluate_ml_models(filtered_data)
        importance_fig = shapley_analysis_plotly(filtered_data)
        return (
            fig,
            model_results,
            importance_fig,
            update_message_box(filtered_data),
        )  # update_ml_models_table(model_results),

    return (
        create_plotly_stock_market_plot(oect_data),
        update_ml_models_table('ml_model_weights/results_0_64.json'),
        update_feature_importance_plot(),
        update_message_box(oect_data),
    )


if __name__ == '__main__':
    app.run_server(debug=True)
