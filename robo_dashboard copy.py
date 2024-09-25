import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import base64
import matplotlib as mpl
import pandas_ta as ta
import plotly.express as px
from ml_models import evaluate_ml_models, plot_ml_comparison

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

oect_data = pd.read_csv('datasets/oect_summary_posted_rf__plus_ml_combined.csv')
df = oect_data.iloc[:,:].copy()
df['SMA_10'] = ta.sma(df['transconductance'], length=10)
df['EMA_10'] = ta.ema(df['transconductance'], length=10)
df['EMA_20'] = ta.ema(df['transconductance'], length=20)
df['SMA_200'] = ta.sma(df['transconductance'], length=200)
df['SMA_50'] = ta.sma(df['transconductance'], length=50)
df['SMA_20'] = ta.sma(df['transconductance'], length=20)
# Calculate RSI
df['RSI'] = ta.rsi(df['transconductance'], length=14)
# Calculate MACD
macd = ta.macd(df['transconductance'], fast=10, slow=20, signal=8)
df = pd.concat([df, macd], axis=1)
datetime_index = pd.date_range(start='2023-01-01', periods=len(df), freq='H')

# Set this datetime index to the DataFrame
df['#'] = datetime_index
colors = []
momentum = df['MACDh_10_20_8'].diff()

for i in range(len(df)):
    if df['MACDh_10_20_8'].iloc[i] >= 0:
        if momentum.iloc[i] > 0:
            colors.append('#96ae8d')  # Bright Green - Increasing Buying Momentum '#789b73', '#a03623'
        else:
            colors.append('#526525')  # Dark Green - Decreasing Buying Momentum
    else:
        if momentum.iloc[i] < 0:
            colors.append('#d9544d')  # Bright Red - Increasing Selling Momentum
        else:
            colors.append('#a03623')  # Dark Red - Decreasing Selling Momentum


# def evaluate_ml_models():
#     # Simulate evaluating multiple ML models and returning a comparison plot
#     df = pd.DataFrame({
#         'Model': ['RandomForest', 'SVM', 'LogisticRegression', 'KNN'],
#         'Accuracy': [0.85, 0.82, 0.78, 0.80]
#     })
#     fig = px.bar(df, x='Model', y='Accuracy', title='ML Models Comparison')
#     return fig

def display_feature_importance():
    # Simulate a feature importance plot
    df = pd.DataFrame({
        'Feature': ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'],
        'Importance': [0.3, 0.25, 0.15, 0.10]
    })
    fig = px.bar(df, x='Feature', y='Importance', title='Feature Importance')
    return fig


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


def create_plot(df):
    mpl.rc('axes', lw=2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # Plot transconductance and moving averages
    ax1.scatter(df.index[21:], df['transconductance'][21:], color='white', label='GPR + RF', edgecolors='black', s=50, lw=2, alpha=0.8)
    ax1.plot(df.index[21:], df['EMA_10'][21:], label='10-points EMA', color='black', linewidth=2, zorder=3, linestyle=':')
    ax1.plot(df.index[21:], df['EMA_20'][21:], label='20-points EMA', color='black', linewidth=2, zorder=4, linestyle='-')
    ax1.legend(facecolor='white')

    # Bin the data points every 5 data points and calculate statistics
    bin_width = 5
    for i in range(21, len(df['transconductance']+1), bin_width):
        bin_data = df['transconductance'][i:i + bin_width]
        if len(bin_data) == bin_width:
            box_color = '#D6B5B3' if bin_data.iloc[0] < bin_data.iloc[-1] else '#B2C8D8'
            ax1.boxplot(bin_data, positions=[i + bin_width // 2], widths=1, patch_artist=True,
                        boxprops=dict(facecolor=box_color, color='black'),
                        whiskerprops=dict(color='black', linestyle='-', linewidth=1.5),
                        capprops=dict(color='black', linewidth=2),
                        medianprops=dict(color='white', linewidth=0.1),
                        whis=[0, 100])

    # Plot MACD
    ax2.plot(df.index, df['MACD_10_20_8'], label='MACD Line', color='black', linewidth=2, linestyle=':')
    ax2.plot(df.index, df['MACDs_10_20_8'], label='Signal Line', color='black', linewidth=2)
    colors = np.where(df['MACDh_10_20_8'] >= 0, '#B2C8D8', '#D6B5B3')
    ax2.bar(df.index.values, df['MACDh_10_20_8'], width=1,  label='Histogram', color=colors, edgecolor='black')

    ax2.legend(facecolor='white')

    # Aesthetic and tick formatting
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    ax1.tick_params(axis='y', labelsize=20)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)

    ax1.yaxis.set_tick_params(width=2)
    ax2.yaxis.set_tick_params(width=2)
    ax2.xaxis.set_tick_params(width=2)

    ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax2.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax2.set_xlim(20, 68)

    # Convert the plot to a PNG image and encode it as base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


def create_plotly_figure(df):
    fig = px.scatter(df, x=df.index, y='transconductance', 
                     labels={'x': 'Index', 'y': 'Transconductance'}, 
                     title="Transconductance with Moving Averages")

    # Add EMA lines to the plot
    fig.add_scatter(x=df.index, y=df['EMA_10'], mode='lines', name='10-points EMA', line=dict(dash='dot', color='black'))
    fig.add_scatter(x=df.index, y=df['EMA_20'], mode='lines', name='20-points EMA', line=dict(dash='solid', color='black'))

    # Add MACD (this could be on a secondary y-axis if needed)
    fig.update_layout(xaxis_title='Index', yaxis_title='Transconductance', hovermode='closest')
    
    return fig

app.layout = html.Div([
    dcc.Tabs(id="tabs-styled-with-inline", value='tab-1', children=[
        dcc.Tab(label='Data visualization', value='tab-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Trendline monitoring', value='tab-2', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='ML Model Evaluation and Feature Importance', value='tab-3', style=tab_style, selected_style=tab_selected_style),
        # dcc.Tab(label='Tab 4', value='tab-4', style=tab_style, selected_style=tab_selected_style),
    ], style=tabs_styles),

    html.Div(id='tabs-content-inline'),

    # Hidden div to store hover information
    html.Div(id='hover-data-output', style={'display': 'none'})
])


@app.callback(Output('tabs-content-inline', 'children'),
              Input('tabs-styled-with-inline', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('OECT data table'),
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in oect_data.columns],
                data=oect_data.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '5px',
                    'fontFamily': 'Arial',
                    'fontSize': 12,
                    'minWidth': '100px',
                    'width': '150px',
                    'maxWidth': '200px',
                }
            )
        ])
    elif tab == 'tab-2':
        img_src = create_plot(df)
        return html.Div([
            html.H3('Dynamic Plot'),
            html.Img(src=img_src, style={'width': '100%'})
        ])
    
    elif tab == 'tab-3':
        # Tab 3: Display the button for ML evaluation
        return html.Div([
            html.H3("ML Model Evaluation"),
            
            # Button to evaluate ML models
            html.Button('Evaluate ML Models', id='evaluate-models-btn', n_clicks=0),

            # Div to display the ML models comparison plot
            dcc.Graph(id='ml-models-comparison-plot'),

            # Div for the "Feature Importance" button (only appears after ML evaluation)
            html.Div(id='feature-importance-btn-div'),

            # Div to display the feature importance plot
            dcc.Graph(id='feature-importance-plot')
        ])
# Callback for Tab 3 to handle button clicks and display plots
@app.callback(
    [Output('ml-models-comparison-plot', 'figure'),
     Output('feature-importance-btn-div', 'children')],
    [Input('evaluate-models-btn', 'n_clicks')]
)
def update_ml_model_evaluation(evaluate_clicks):
    # Initialize empty figure for the model comparison plot
    ml_models_fig = {}

    # If the "Evaluate ML Models" button is clicked, generate the ML models comparison plot
    if evaluate_clicks > 0:
        ml_models_fig = evaluate_ml_models()
        # Return the plot and also create the "Feature Importance" button under the plot
        return ml_models_fig, html.Button('Feature Importance', id='feature-importance-btn', n_clicks=0)

    return ml_models_fig, None  # Return empty until the button is clicked

# Callback to display feature importance when the "Feature Importance" button is clicked
@app.callback(
    Output('feature-importance-plot', 'figure'),
    [Input('feature-importance-btn', 'n_clicks')]
)
def update_feature_importance(feature_importance_clicks):
    # Initialize empty figure for the feature importance plot
    feature_importance_fig = {}

    # If the "Feature Importance" button is clicked, generate the feature importance plot
    if feature_importance_clicks > 0:
        feature_importance_fig = display_feature_importance()

    return feature_importance_fig

if __name__ == '__main__':
    app.run_server(debug=True)