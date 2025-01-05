import dash
import matplotlib
import pandas as pd
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State
import os
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
from dash.exceptions import PreventUpdate

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
    
    if oect_data is None or len(oect_data) == 0:
        return "No data available."
        
    df = oect_data.copy()
    
    # Calculate technical indicators
    df['SMA_10'] = ta.sma(df['transconductance'], length=10)
    df['EMA_10'] = ta.ema(df['transconductance'], length=10)
    df['EMA_20'] = ta.ema(df['transconductance'], length=20)
    df['SMA_200'] = ta.sma(df['transconductance'], length=200)
    df['SMA_50'] = ta.sma(df['transconductance'], length=50)
    df['SMA_20'] = ta.sma(df['transconductance'], length=20)
    df['RSI'] = ta.rsi(df['transconductance'], length=14)
    
    try:
        # Calculate MACD
        macd = ta.macd(df['transconductance'], fast=10, slow=20, signal=8)
        df = pd.concat([df, macd], axis=1)
        
        macd_values = df['MACDh_10_20_8'].values
        current_index = len(macd_values) - 1  # Get the latest point index
        
        # Look back from the current point to find the last negative value
        last_negative_index = None
        for i in range(current_index, max(-1, current_index - 6), -1):
            if macd_values[i] < 0:
                last_negative_index = i
                break

        # print('last_negative_index', last_negative_index)
        # If we found a negative value in the last 5 points
        if last_negative_index is not None and (current_index - last_negative_index) <= 5:
            return f"The MACD indicator is showing a downward trajectory at sample number {df.loc[last_negative_index, '#']}. Consider changing strategy by selecting different ML model or modify the parameter range."
                
    except Exception as e:
        #print(f"Error in MACD calculation: {e}")
        return None
    
    return None


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

# Helper functions to process film images
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


# DataManager to handdle the high-throughput data and detect new datapoints added in the file
class DataManager:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.last_update = None
        self.data = None
        self.previous_data = None  # Cache for previous data
        self._load_data()

    def _load_data(self):
        """Load and process the CSV data"""
        try:
            df = pd.read_csv(self.csv_path)
            
            # Process the data
            df['coating_on_top.sol_label'] = df['coating_on_top.sol_label'].map(
                lambda x: x.lstrip('mg/ml').rstrip('mg/ml'))
            df['coating_on_top.substrate_label'] = df['coating_on_top.substrate_label'].map(
                lambda x: x.lstrip('nm').rstrip('nm'))
                
            # Convert columns to numeric
            numeric_columns = [
                'coating_on_top.sol_label',
                'coating_on_top.substrate_label',
                'coating_on_top.vel',
                'coating_on_top.T'
            ]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
                
            # Add image links
            df['image'] = df['ID'].apply(create_image_link)
            
            # Store previous data before updating
            if self.data is not None:
                self.previous_data = self.data.copy()
            
            self.data = df
            self.last_update = os.path.getmtime(self.csv_path)
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return self.previous_data if self.previous_data is not None else pd.DataFrame()

    def needs_update(self):
        """Check if the CSV file has been modified"""
        try:
            if self.last_update is None:
                return True
            current_time = os.path.getmtime(self.csv_path)
            return current_time > self.last_update
        except OSError as e:
            print(f"Error checking file modification time: {str(e)}")
            return False

    def get_data(self, force_refresh=False):
        """Get the data, refreshing if needed or forced"""
        try:
            if force_refresh or self.needs_update():
                new_data = self._load_data()
                return new_data if not new_data.empty else self.previous_data
            return self.data if self.data is not None else self.previous_data
        except Exception as e:
            print(f"Error getting data: {str(e)}")
            return self.previous_data if self.previous_data is not None else pd.DataFrame()


# Initialize the global data manager
data_manager = DataManager('datasets/oect_summary_posted_rf__plus_ml_combined.csv')

# Dash app initialization
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Timeout limit
server.config['PROPAGATE_EXCEPTIONS'] = True
server.config['WTF_CSRF_TIME_LIMIT'] = 3600

# Container style for the entire layout
container_style = {
    'display': 'flex',
    'flexDirection': 'row',
    'height': '100vh',
    'width': '100%',
}

sidebar_style = {
    'display': 'flex',
    'flexDirection': 'column',
    'width': '200px',
    'borderRight': '1px solid #d6d6d6',
    'backgroundColor': '#f8f9fa',
}

# Style for individual tabs
tabs_styles = {
    #'width': '200px',
    'flexGrow': 1,
    #'borderRight': '1px solid #d6d6d6',
    'backgroundColor': '#f8f9fa',
}
tab_style = {
    'padding': '12px 16px',
    'fontWeight': 'bold',
    'fontSize': '20px',
    'color': '#495057',
    'borderBottom': '1px solid #dee2e6',
    'cursor': 'pointer',
    'transition': 'all 0.2s ease-in-out',
}

# Style of the tabs 1&2
tab_selected_style = {
    'padding': '12px 16px',
    'fontWeight': 'bold',
    'fontSize': '20px',
    'color': 'white',
    'backgroundColor': '#119DFF',
    'borderLeft': '4px solid #0066cc',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
}

# Style of the content area
content_style = {
    'flex': '1',
    'padding': '20px',
}

# Style of the logo container
logo_style = {
    'padding': '10px',
    'backgroundColor': '#f8f9fa',  
    'borderBottom': '1px solid #d6d6d6',
    'display': 'flex',
    'justifyContent': 'center',
    'alignItems': 'center'
}

app.layout = html.Div([
    html.Div([
        # Logo container
        html.Div([
            html.Img(
                src='/assets/polybot_logo.png', 
                style={
                    'height': '200px',  
                    'width': 'auto',
                }
            )
        ], style=logo_style),
        
        # Tab container
        dcc.Tabs(
            id="tabs-styled-with-inline",
            value='tab-1',
            vertical=True,
            children=[
                dcc.Tab(
                    label='AI Advisor Board',
                    value='tab-1',
                    style=tab_style,
                    selected_style=tab_selected_style,
                ),
                dcc.Tab(
                    label='Data Visualization',
                    value='tab-2',
                    style=tab_style,
                    selected_style=tab_selected_style,
                ),
            ],
            style=tabs_styles,
        ),
        ], style=sidebar_style),
        html.Div(id='tabs-content-inline', style=content_style),
        html.Div(id='hover-data-output', style={'display': 'none'}),
        dcc.Interval(
            id='page-load', interval=1, n_intervals=0, max_intervals=1  #interval=18000, n_intervals=0, disabled=False# interval=1, n_intervals=0, max_intervals=1  #interval=50000, n_intervals=0#, max_intervals=-1  # interval=1, n_intervals=0, max_intervals=1 
        ), 
    ], style=container_style)


@app.callback(
    Output('tabs-content-inline', 'children'),
    Input('page-load', 'n_intervals'),
    #   Input('interval-refresh', 'n_intervals'),
    Input('tabs-styled-with-inline', 'value'),
)
def render_content(n_intervals, tab):
    oect_data = data_manager.get_data()
    
    if tab == 'tab-1':
        return html.Div([
            
            # Message Box Card
            dbc.Card(
                dbc.CardBody([
                    html.H3("AI Advisor Message", style={'textAlign': 'center', 'fontSize': '24px'}), 
                    html.Div(
                        id='message-box',
                        style={
                            'background-color': '#F9F1A5',
                            'padding': '15px',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'fontSize': '20px',
                            'fontWeight': 'bold',
                            'color': 'black',
                        }
                    ),
                ]),
                className="mb-4",
                style={
                    'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
                    'width': '80%', 
                    'margin': '20px auto'  
                }
            ),
            
            # Stock Market Plot Card
            dbc.Card(
                dbc.CardBody([
                    html.H3(
                        "Trendline Monitoring: Transconductance and Moving Averages",
                        style={'textAlign': 'center', 'marginBottom': '20px','marginTop': '10px', 'fontSize': '24px'}
                    ),
                    dcc.Graph(
                        id='stock-market-plot',
                        style={
                            'height': '600px', 
                        },
                        config={
                            'displaylogo': False,
                            'modeBarButtonsToRemove': [
                                'select2d', 'lasso2d', 'zoom', 'zoomIn2d',
                                'zoomOut2d', 'autoScale2d', 'resetScale2d', 'pan2d'
                            ],
                        }
                    ),
                ]),
                className="mb-4",
                style={
                    'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
                    'width': '90%', 
                    'margin': '30px auto'  
                }
            ),            
            html.Div([
                # ML Evaluation Card
                dbc.Card(
                    dbc.CardBody([
                        html.H3(
                            "ML Model Evaluation",
                            style={'textAlign': 'center', 'marginBottom': '20px','marginTop': '10px', 'fontSize': '24px'}
                        ),

                        dcc.Loading(
                            id="loading-results",
                            type="circle",  # Spinner type (circle, default, etc.)
                            children=[
                                html.Div(
                                    id='results-container',  # Static container (important for Loading)
                                    children=html.Div(id='results-table-div'),  # Dynamic content inside
                                    style={
                                        'height': '450px',
                                        'overflow': 'auto',
                                        'fontSize': '18px',  
                                        'marginTop': '20px',  
                                    }
                                )
                            ],
                        ),
                    ]),
                    style={
                        'width': '48%',
                        'display': 'inline-block',
                        'verticalAlign': 'top',
                        'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
                        'margin': '0 10px'
                    }
                ),
                
                # SHAP Plot Card
                dbc.Card(
                    dbc.CardBody([
                        html.H3(
                            "SHAP Summary Plot",
                            style={'textAlign': 'center', 'marginBottom': '20px','marginTop': '10px', 'fontSize': '24px'}
                        ),
                        dcc.Graph(
                            id='feature-importance-plot',
                            style={
                                'height': '450px',
                                'marginTop': '20px'
                            }
                        ),
                    ]),
                    style={
                        'width': '48%',  
                        'display': 'inline-block',
                        'verticalAlign': 'top',
                        'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
                        'margin': '0 10px'
                    }
                ),
            ], style={
                'width': '95%',  
                'display': 'flex',
                'justifyContent': 'center',
                'marginTop': '20px',
                'margin': '30px auto', 
                'paddingBottom': '30px'
            })
        ], 
        style={
        'padding': '20px', 
        'backgroundColor': '#f8f9fa',  
        'minHeight': '100vh'  
        }),

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

    min_rmse = min(table_data, key=lambda x: float(x["Average Test RMSE"]))["Average Test RMSE"]
    min_std_dev = min(table_data, key=lambda x: float(x["Test RMSE Std Dev"]))["Test RMSE Std Dev"]

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

    style_data_conditional.extend([
        {
            'if': {
                'filter_query': f'{{Average Test RMSE}} = "{min_rmse}"',
                'column_id': 'Average Test RMSE'
            },
            'fontWeight': 'bold',
        },
        {
            'if': {
                'filter_query': f'{{Test RMSE Std Dev}} = "{min_std_dev}"',
                'column_id': 'Test RMSE Std Dev'
            },
            'fontWeight': 'bold',
        }
    ])

    results_table = dash_table.DataTable(
        data=table_data,
        columns=[
            {
                "name": ["Model\n"],
                "id": "Model"
            },
            {
                "name": ["Average\nTest RMSE"],
                "id": "Average Test RMSE"
            },
            {
                "name": ["Test RMSE\nStd Dev"],
                "id": "Test RMSE Std Dev"
            },
            {
                "name": ["Train-Test\nRMSE Diff"],
                "id": "Train-Test RMSE Diff"
            },
        ],
        style_table={
            'overflowX': 'auto',
            'minWidth': '100%',
            'width': '100%',
            'maxHeight': '500px',
        },
        style_cell={
            'textAlign': 'center',
            'padding': '15px',
            'fontSize': '20px',
            'font-family': 'Arial',
            'minWidth': '150px',
            'width': '25%',
            'maxWidth': '25%',
        },
        style_header={
            'backgroundColor': '#f4f4f4',
            'fontWeight': 'bold',
            'fontSize': '16px',
            'height': 'auto',
            'whiteSpace': 'pre-line',
            'padding': '10px',
            'textAlign': 'center',
            'lineHeight': '1.2'
        },
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_data_conditional=style_data_conditional,
        merge_duplicate_headers=True,
        fixed_rows={'headers': True},
    )
    
    return [results_table]

def update_feature_importance_plot():
    # Load precomputed SHAP plot from JSON
    with open("shap_plot.json", "r") as f:
        feature_importance_fig = pio.from_json(f.read())

    feature_importance_fig.update_layout(
        title="",
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title="SHAP Value",
            title_font=dict(size=20),
            tickfont=dict(size=20),
            gridcolor='lightgrey',
        ),
        yaxis=dict(
            title="Feature",
            title_font=dict(size=20),
            tickfont=dict(size=20),
            gridcolor='lightgrey',
        ),
        coloraxis_colorbar=dict(
            title="Feature Value",
            title_font=dict(size=18),
            tickfont=dict(size=16),
        ),
        margin=dict(l=50, r=50, t=20, b=40),
        height=None,
        width=None,
        autosize=True
    )

    feature_importance_fig.update_traces(marker=dict(size=12, opacity=0.7))

    feature_importance_fig.add_vline(
        x=0,
        line=dict(color="black", width=1, dash="dash"),
    )
    return feature_importance_fig

# @app.callback(
#     [
#         Output('stock-market-plot', 'figure'),
#         Output('results-table-div', 'children'),
#         Output('feature-importance-plot', 'figure'),
#         Output('message-box', 'children'),
#     ],
#     [Input('stock-market-plot', 'relayoutData')],
#     prevent_initial_call=True,
# )
# def update_charts(relayout_data):
#     # data_file_path = 'datasets/oect_summary_posted_rf__plus_ml_combined.csv'
#     oect_data = pd.read_csv(
#         'datasets/oect_summary_posted_rf__plus_ml_combined.csv'
#     )
#     oect_data['coating_on_top.sol_label'] = oect_data[
#         'coating_on_top.sol_label'
#     ].map(lambda x: x.lstrip('mg/ml').rstrip('mg/ml'))
#     oect_data['coating_on_top.substrate_label'] = oect_data[
#         'coating_on_top.substrate_label'
#     ].map(lambda x: x.lstrip('nm').rstrip('nm'))
#     oect_data['coating_on_top.sol_label'] = pd.to_numeric(
#         oect_data['coating_on_top.sol_label']
#     )
#     oect_data['coating_on_top.substrate_label'] = pd.to_numeric(
#         oect_data['coating_on_top.substrate_label']
#     )
#     oect_data['coating_on_top.vel'] = pd.to_numeric(
#         oect_data['coating_on_top.vel']
#     )
#     oect_data['coating_on_top.T'] = pd.to_numeric(oect_data['coating_on_top.T'])
#     df = oect_data.copy()
#     df['image'] = df['ID'].apply(create_image_link)

#     # print("relayout_data:", relayout_data)  # Debug print
#     if relayout_data and 'xaxis.range' in relayout_data:
#         start_idx = 0 #int(float(relayout_data['xaxis.range'][0]))
#         end_idx = int(float(relayout_data['xaxis.range'][1]))
#         filtered_data = oect_data.iloc[start_idx:end_idx]
#         # print('Range selected:', start_idx, '-', end_idx)
#         #    print('Filtered data shape:', filtered_data)

#         fig = create_plotly_stock_market_plot(filtered_data)
#         fig.update_layout(xaxis_range=[start_idx, end_idx])

#         model_results = update_ml_models_table(
#             f"ml_model_weights/results_{start_idx}_{end_idx}.json"
#         )
#         #print('model_results', model_results)
#         #    model_results = evaluate_ml_models(filtered_data)
#         importance_fig = shapley_analysis_plotly(filtered_data)
#         return (
#             fig,
#             model_results,
#             importance_fig,
#             update_message_box(filtered_data),
#         )  # update_ml_models_table(model_results),

#     return (
#         create_plotly_stock_market_plot(oect_data),
#         update_ml_models_table('ml_model_weights/results_0_64.json'),
#         update_feature_importance_plot(),
#         update_message_box(oect_data),
#     )


# @app.callback(
#     [
#         Output('stock-market-plot', 'figure'),
#         Output('results-table-div', 'children'),
#         Output('feature-importance-plot', 'figure'),
#         Output('message-box', 'children'),
#     ],
#     [Input('stock-market-plot', 'relayoutData'),
#      Input('page-load', 'n_intervals')],  # Add this input
#     prevent_initial_call=False,  # Change this to False
# )
# def update_charts(relayout_data, n_intervals):  # Add n_intervals parameter
#     # data_file_path = 'datasets/oect_summary_posted_rf__plus_ml_combined.csv'
#     oect_data = pd.read_csv(
#         'datasets/oect_summary_posted_rf__plus_ml_combined.csv'
#     )
#     oect_data['coating_on_top.sol_label'] = oect_data[
#         'coating_on_top.sol_label'
#     ].map(lambda x: x.lstrip('mg/ml').rstrip('mg/ml'))
#     oect_data['coating_on_top.substrate_label'] = oect_data[
#         'coating_on_top.substrate_label'
#     ].map(lambda x: x.lstrip('nm').rstrip('nm'))
#     oect_data['coating_on_top.sol_label'] = pd.to_numeric(
#         oect_data['coating_on_top.sol_label']
#     )
#     oect_data['coating_on_top.substrate_label'] = pd.to_numeric(
#         oect_data['coating_on_top.substrate_label']
#     )
#     oect_data['coating_on_top.vel'] = pd.to_numeric(
#         oect_data['coating_on_top.vel']
#     )
#     oect_data['coating_on_top.T'] = pd.to_numeric(oect_data['coating_on_top.T'])
#     df = oect_data.copy()
#     df['image'] = df['ID'].apply(create_image_link)
    
#     # print("relayout_data:", relayout_data)  # Debug print
#     # if relayout_data and 'xaxis.range' in relayout_data:
#     start_idx = 0 #int(float(relayout_data['xaxis.range'][0]))
#     end_idx = df.shape[0] #int(float(relayout_data['xaxis.range'][1]))
#     filtered_data = oect_data.iloc[start_idx:end_idx]
#         # print('Range selected:', start_idx, '-', end_idx)
#         #    print('Filtered data shape:', filtered_data)

#     fig = create_plotly_stock_market_plot(filtered_data)
#     fig.update_layout(xaxis_range=[start_idx, end_idx])

#     model_results = update_ml_models_table(
#             f"ml_model_weights/results_{start_idx}_{end_idx}.json"
#         )
#         #print('model_results', model_results)
#         #    model_results = evaluate_ml_models(filtered_data)
#     importance_fig = shapley_analysis_plotly(filtered_data)
#     return (
#             fig,
#             model_results,
#             importance_fig,
#             update_message_box(filtered_data),
#         )  # update_ml_models_table(model_results),

#     # return (
#     #     create_plotly_stock_market_plot(oect_data),
#     #     update_ml_models_table('ml_model_weights/results_0_64.json'),
#     #     update_feature_importance_plot(),
#     #     update_message_box(oect_data),
#     # )



@app.callback(
    [
        Output('stock-market-plot', 'figure'),
        Output('results-table-div', 'children'),
        Output('feature-importance-plot', 'figure'),
        Output('message-box', 'children'),
    ],
    [
        Input('stock-market-plot', 'relayoutData'),
        Input('page-load', 'n_intervals')
    ],
    prevent_initial_call=False,
)
def update_charts(relayout_data, n_intervals):
    ctx = dash.callback_context
    
    # Get the current data
    current_data = data_manager.get_data()
    
    # If there's no data, return the previous state
    if current_data is None or current_data.empty:
        raise PreventUpdate
        
    # Calculate indices
    if relayout_data and 'xaxis.range' in relayout_data:
        start_idx = max(0, int(float(relayout_data['xaxis.range'][0])))
        end_idx = min(current_data.shape[0], int(float(relayout_data['xaxis.range'][1])))
    else:
        start_idx = 0
        end_idx = current_data.shape[0]
        
    filtered_data = current_data.iloc[start_idx:end_idx]

    # Create stock market plot
    fig = create_plotly_stock_market_plot(filtered_data)
    
    # Update layout only if we have data
    if not filtered_data.empty:
        fig.update_layout(
            xaxis_range=[start_idx, end_idx],
            showlegend=True,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )

    # Get model results
    try:
        model_results = update_ml_models_table(
            f"ml_model_weights/results_{start_idx}_{end_idx}.json"
        )
    except Exception as e:
        print(f"Error loading model results: {str(e)}")
        #model_results = create_default_table()

    # Create SHAP plot
    importance_fig = shapley_analysis_plotly(filtered_data)
    
    # Get message
    message = update_message_box(filtered_data)
    if not message:
        message = "AI advisor is monitoring the workflow. No modifications required at this time."

    return fig, model_results, importance_fig, message


if __name__ == '__main__':
    app.run_server(debug=True)

# For render deployment
# server = app.server

# if __name__ == '__main__':
#     app.run_server(host='0.0.0.0', port=int(os.environ.get('PORT', 3000)))