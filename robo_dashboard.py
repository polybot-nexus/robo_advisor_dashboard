import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import dash_table
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from ml_models import evaluate_ml_models, plot_whiskerplots, shapley_analysis_plotly
from utils import create_plotly_stock_market_plot, create_plotly_trendline_fit, get_trendline_slope
import plotly.express as px
from PIL import Image
from io import BytesIO
import base64
import imageio
import dash_bootstrap_components as dbc

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
        if df['MACDh_10_20_8'].values[-1]<= -30:
            return "The transconductance MACD indicator is showing a downward trajectory. Consider Changing strategy."
    # if consecutive_declines(slopes) >= 3:
    #     return "The transconductance trendline slope is showing a downward trajectory. Consider Changing strategy."
    except:   
        return "Robo-advisor will display a message if any workflow modification is required."
    
    return "Robo-advisor will display a message if any workflow modification is required."


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
        img = imageio.imread(f'images/{id}_raw_annealed_film.jpg')
        img = Image.fromarray(img)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return buffered.getvalue()
    except:
        # print(id)
        return None
    
def create_image_link(id):
    image_path = f'images/{id}_raw_annealed_film.jpg'
    return f'[View Image]({image_path})'

def image_base64(im):
    if im is None:
        return None
    return base64.b64encode(im).decode()

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
oect_data = pd.read_csv('datasets/oect_summary_posted_rf__plus_ml_combined.csv')
oect_data['coating_on_top.sol_label'] = oect_data['coating_on_top.sol_label'].map(lambda x: x.lstrip('mg/ml').rstrip('mg/ml'))
oect_data['coating_on_top.substrate_label'] = oect_data['coating_on_top.substrate_label'].map(lambda x: x.lstrip('nm').rstrip('nm'))
oect_data['coating_on_top.sol_label'] = pd.to_numeric(oect_data['coating_on_top.sol_label'])
oect_data['coating_on_top.substrate_label'] = pd.to_numeric(oect_data['coating_on_top.substrate_label'])
oect_data['coating_on_top.vel'] = pd.to_numeric(oect_data['coating_on_top.vel'])
oect_data['coating_on_top.T'] = pd.to_numeric(oect_data['coating_on_top.T'])
df = oect_data.copy()
df['image'] = df['ID'].apply(create_image_link)


# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

app.layout = html.Div([
    dcc.Tabs(id="tabs-styled-with-inline", value='tab-1', children=[
        dcc.Tab(label='Data visualization', value='tab-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Trendline monitoring', value='tab-2', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='ML Model Evaluation and Feature Importance', value='tab-3', style=tab_style, selected_style=tab_selected_style),
    ], style=tabs_styles),
    html.Div(id='tabs-content-inline'),
    html.Div(id='hover-data-output', style={'display': 'none'}),
    dcc.Interval(
        id='interval-refresh',
        interval=5*1000,  # 60 minutes in milliseconds
        n_intervals=0
    )    
])
    
@app.callback(Output('tabs-content-inline', 'children'),
              Input('interval-refresh', 'n_intervals'),
              Input('tabs-styled-with-inline', 'value'))
def render_content(n_intervals, tab):
    oect_data = pd.read_csv('datasets/oect_summary_posted_rf__plus_ml_combined.csv')   
    oect_data['coating_on_top.sol_label'] = oect_data['coating_on_top.sol_label'].map(lambda x: x.lstrip('mg/ml').rstrip('mg/ml'))
    oect_data['coating_on_top.substrate_label'] = oect_data['coating_on_top.substrate_label'].map(lambda x: x.lstrip('nm').rstrip('nm'))
    oect_data['coating_on_top.sol_label'] = pd.to_numeric(oect_data['coating_on_top.sol_label'])
    oect_data['coating_on_top.substrate_label'] = pd.to_numeric(oect_data['coating_on_top.substrate_label'])
    oect_data['coating_on_top.vel'] = pd.to_numeric(oect_data['coating_on_top.vel'])
    oect_data['coating_on_top.T'] = pd.to_numeric(oect_data['coating_on_top.T'])
    oect_data['image'] = oect_data['ID'].apply(create_image_link) 
    if tab == 'tab-1':
        return html.Div([
            html.Div([
                html.H2("Table of All Samples"),
                dash_table.DataTable(
                    id='sample-table',
                    columns=[
                        {"name": 'ID', "id": 'ID'},
                        {"name": 'Coating Speed', "id": 'coating_on_top.vel'},
                        {"name": 'Coating Temperature', "id": 'coating_on_top.T'},
                        {"name": 'Concentration', "id": 'coating_on_top.sol_label'},
                        {"name": 'Substrate', "id": 'coating_on_top.substrate_label'},
                        {"name": 'Transconductance', "id": 'transconductance'},
                        {"name": 'Film Image', "id": 'image', "presentation": "markdown"},
                    ],
                    data=oect_data.to_dict('records'),
                    style_table={
                    'height': '200px', 
                    'width': '100%',  
                    'overflowY': 'auto',  
                    'overflowX': 'auto'  
                    },
                    style_cell={
                    'textAlign': 'left',
                    'fontSize': '12px',  
                    'padding': '5px'    
                    },
                    style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold',
                    'fontSize': '12px', 
                    'padding': '5px' 
                    }, style_data_conditional=[
                    {
                        'if': {'column_id': 'image'}, 
                        'textAlign': 'center',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    }
                ],
                )], style={'width': '100%', 'display': 'inline-block'}) ,

            html.Div([
                html.Div([
                    dcc.Markdown("#### Concentration"),
                    dcc.Graph(id='concentration-plot')
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Markdown("#### Substrate"),
                    dcc.Graph(id='substrate-plot')
                ], style={'width': '48%', 'display': 'inline-block'}),
            ]),
            
            html.Div([
                html.Div([
                    dcc.Markdown("#### Coating Speed"),
                    dcc.Graph(id='coating-speed-plot')
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Markdown("#### Coating Temperature"),
                    dcc.Graph(id='coating-temp-plot')
                ], style={'width': '48%', 'display': 'inline-block'}),
            ])
        ])
    
    elif tab == 'tab-2':
        fig_stock_trends = create_plotly_stock_market_plot(oect_data)
        fig_trendline = create_plotly_trendline_fit(oect_data)
        response = update_message_box(oect_data)

        return html.Div([
            # First section: Trendlines graph 
            html.Div([
                # html.H3('Spline Trendline'),
                dcc.Graph(figure=fig_trendline)
            ], style={'width': '100%', 'margin-bottom': '20px'}),

            # Second section: Stock Market Trendline plot and message box side by side
            html.Div([
                # Trendlines graph
                html.Div([
                    # dcc.Markdown("#### Stock market trendlines"),
                    dcc.Graph(figure=fig_stock_trends)
                ], style={'width': '35%', 'display': 'inline-block', 'vertical-align': 'top'}), 

                # Message box
                html.Div([dcc.Markdown("#### Message Box"),
                        html.Div(id='message-box', children=f"{response}" , style={
                        'border': '2px solid blue', 
                        'padding': '10px', 
                        'border-radius': '3px', 
                        'box-shadow': '2px 2px 10px rgba(0, 0, 0, 0.1)', 
                        'background-color': '#f9f9f9',
                        'color': 'black',
                        'font-weight': 'bold'
                    })
                ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top'})  
            ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'})
        ])

    elif tab == 'tab-3':
        return html.Div([
            html.H3("ML Model Evaluation"),
            html.Button('Evaluate ML Models', id='evaluate-models-btn', n_clicks=0),
            dcc.Graph(id='ml-models-comparison-plot'),
            html.Div(id='feature-importance-btn-div'),
            dcc.Graph(id='feature-importance-plot')
        ])
############################################################################################################

# Callbacks for dynamic updating
@app.callback(
    Output('concentration-plot', 'figure'),
    Output('substrate-plot', 'figure'),
    Output('coating-speed-plot', 'figure'),
    Output('coating-temp-plot', 'figure'),
    Input('sample-table', 'data')
)
def update_graphs(rows):
    df = pd.DataFrame(rows)
    # Transconductance Plot
    fig_conc = px.scatter(data_frame=df, x='coating_on_top.sol_label', y='transconductance', color='transconductance')
    style_plot(fig_conc)
    # Device Plot
    fig_substr = px.scatter(data_frame=df, x='coating_on_top.substrate_label', y='transconductance', color='transconductance')
    style_plot(fig_substr)
    # Coating Speed Plot
    fig_speed = px.scatter(data_frame=df, x='coating_on_top.vel', y='transconductance', color='transconductance')
    style_plot(fig_speed)
    # Coating Temperature Plot
    fig_temp = px.scatter(data_frame=df, x='coating_on_top.T', y='transconductance', color='transconductance')
    style_plot(fig_temp)
    return fig_conc, fig_substr, fig_speed, fig_temp

#############################################################################################################
# Callback to generate and display the ML models comparison plot
@app.callback(
    [Output('ml-models-comparison-plot', 'figure'),
     Output('feature-importance-btn-div', 'children')],
    [Input('evaluate-models-btn', 'n_clicks')]
)
def update_ml_models_comparison(n_clicks):

    ml_comparison_fig = {}
    print(f"Button clicked {n_clicks} times")  
    if n_clicks > 0:
        results = evaluate_ml_models()
        ml_comparison_fig = plot_whiskerplots(results)
        return ml_comparison_fig, html.Button('Feature Importance', id='feature-importance-btn', n_clicks=0)

    return ml_comparison_fig, None

# Callback to display feature importance when the "Feature Importance" button is clicked
@app.callback(
    Output('feature-importance-plot', 'figure'),
    [Input('feature-importance-btn', 'n_clicks')]
)
def update_feature_importance(feature_importance_clicks):
    feature_importance_fig = {}
    # If the "Feature Importance" button is clicked, generate the feature importance plot
    if feature_importance_clicks > 0:
        feature_importance_fig = shapley_analysis_plotly()

    return feature_importance_fig

def style_plot(fig):
    fig.update_traces(marker={'size': 7})
    fig.update_layout(plot_bgcolor='white')
    fig.update_layout(legend={
        'orientation': 'h',
        'yanchor': "bottom",
        'y': 1.02,
        'xanchor': "left",
        'x': 0,
    })
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

if __name__ == '__main__':
    app.run_server(debug=True)