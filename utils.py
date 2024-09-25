import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


def create_plotly_stock_market_plot(oect_data):
    df = oect_data.iloc[:,:].copy()
    df['SMA_10'] = ta.sma(df['transconductance'], length=10)
    df['EMA_10'] = ta.ema(df['transconductance'], length=10)
    df['EMA_20'] = ta.ema(df['transconductance'], length=20)
    df['SMA_200'] = ta.sma(df['transconductance'], length=200)
    df['SMA_50'] = ta.sma(df['transconductance'], length=50)
    df['SMA_20'] = ta.sma(df['transconductance'], length=20)
    df['RSI'] = ta.rsi(df['transconductance'], length=14)


    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Transconductance and Moving Averages", "MACD Histogram"),
                        row_heights=[0.5, 0.5])

    # Scatter plot for transconductance
    fig.add_trace(go.Scatter(x=df.index[21:], y=df['transconductance'][21:], 
                             mode='markers', marker=dict(color='white', size=10, line=dict(width=2, color='black')), 
                             name='GPR + RF'), row=1, col=1)

    # Line plots for EMA
    fig.add_trace(go.Scatter(x=df.index[21:], y=df['EMA_10'][21:], mode='lines', line=dict(color='black', dash='dot'), 
                             name='10-points EMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[21:], y=df['EMA_20'][21:], mode='lines', line=dict(color='black'), 
                             name='20-points EMA'), row=1, col=1)

    # Boxplots for binned data
    bin_width = 5
    for i in range(21, len(df['transconductance']+1), bin_width):
        bin_data = df['transconductance'][i:i + bin_width]
        if len(bin_data) == bin_width:
            box_color = '#D6B5B3' if bin_data.iloc[0] < bin_data.iloc[-1] else '#B2C8D8'
            # Statistics for the boxplot
            q1 = np.percentile(bin_data, 25)
            median = np.percentile(bin_data, 50)
            q3 = np.percentile(bin_data, 75)
            lower_whisker = np.min(bin_data)
            upper_whisker = np.max(bin_data)
            
            x_position = i + bin_width // 2  

            fig.add_trace(go.Scatter(
                x=[x_position, x_position], 
                y=[q1, q3], 
                mode='lines', 
                line=dict(color=box_color, width=8), 
                name=f'Bin {i}',
                showlegend=False, 
            ))

            fig.add_trace(go.Scatter(
                x=[x_position], 
                y=[median], 
                mode='markers', 
                marker=dict(color='white', size=10, line=dict(width=2, color='black')), 
                name=f'Median {i}',
                showlegend=False, 
            ))

            fig.add_trace(go.Scatter(
                x=[x_position, x_position], 
                y=[lower_whisker, upper_whisker], 
                mode='lines', 
                line=dict(color='black', width=2), 
                showlegend=False, 
            ))

    try:
        macd = ta.macd(df['transconductance'], fast=10, slow=20, signal=8)
        df = pd.concat([df, macd], axis=1)
        # MACD Line and Signal Line
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_10_20_8'], mode='lines', line=dict(color='black', dash='dot'), 
                                name='MACD Line'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_10_20_8'], mode='lines', line=dict(color='black'), 
                                name='Signal Line'), row=2, col=1)

        # MACD Histogram
        colors = np.where(df['MACDh_10_20_8'] >= 0, '#B2C8D8', '#D6B5B3')
        fig.add_trace(go.Bar(x=df.index, y=df['MACDh_10_20_8'], marker=dict(color=colors, line=dict(color='black')), 
                            name='Histogram'), row=2, col=1)
        fig.update_layout(height=600, width=1000, #title_text="OECT Data Visualization", 
                        showlegend=True, legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.5)"))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', row=2, col=1)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=1, col=1)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=2, col=1)

    except:
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='black'), 
                            name='Empty Plot'), row=2, col=1)
        fig.update_layout(height=600, width=1000,# title_text="OECT Data Visualization - Partial Empty Plot", 
                        showlegend=True)
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', row=2, col=1)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=2, col=1)

    return fig

def create_plotly_trendline_fit(oect_data):
    # Get the relevant data for plotting
    x = oect_data['#']
    y = oect_data['transconductance']
    # x = oect_data[oect_data.source == 'gpr'].iloc[:21, :]['#']
    # y = oect_data[oect_data.source == 'gpr'].iloc[:21, :]['transconductance']

    # Create a scatter plot for the original data points
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=12, color='#5a7d9a', line=dict(width=2, color='black')),
        name='Data Points'
    ))

    # Create and smooth the spline curve
    spl = UnivariateSpline(x, y, s=0.1, k=3)
    xs = np.linspace(min(x), max(x), x.shape[0]*5)
    ys = spl(xs)
    smoothed_ys = savgol_filter(ys, 70, 4)

    # Plot the smoothed spline
    fig.add_trace(go.Scatter(
        x=xs, y=smoothed_ys,
        mode='lines',
        line=dict(color='#87ae73', width=2),
        name='Smoothed Spline'
    ))

    # Calculate the slope for the tangent line at the last point
    slopes = np.gradient(smoothed_ys, xs)
    slope_at_last_smoothed = slopes[-1]

    # Calculate the tangent line at the last point
    tangent_x_smoothed = np.array([xs[-1]-2, xs[-1]+2])
    tangent_y_smoothed = smoothed_ys[-1] + slope_at_last_smoothed * (tangent_x_smoothed - xs[-1])

    # Plot the tangent line
    fig.add_trace(go.Scatter(
        x=tangent_x_smoothed, y=tangent_y_smoothed,
        mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name=f'Tangent Line (Slope: {slope_at_last_smoothed:.2f})'
    ))

    # Add labels and title
    fig.update_layout(
        xaxis_title='Number of iterations',
        yaxis_title="μC*",
        title='Trendline Fit with Spline and Tangent Line',
        font=dict(size=14)
    )

    return fig


def get_trendline_slope(oect_data):
    x = oect_data['#']
    y = oect_data['transconductance']

    # Create and smooth the spline curve
    spl = UnivariateSpline(x, y, s=0.1, k=3)
    xs = np.linspace(min(x), max(x), x.shape[0]*5)
    ys = spl(xs)
    smoothed_ys = savgol_filter(ys, 70, 4)

    # Calculate the slope for the tangent line at the last point
    slopes = np.gradient(smoothed_ys, xs)
    slope_at_last_smoothed = slopes[-1]

    # Calculate the tangent line at the last point
    tangent_x_smoothed = np.array([xs[-1]-2, xs[-1]+2])
    tangent_y_smoothed = smoothed_ys[-1] + slope_at_last_smoothed * (tangent_x_smoothed - xs[-1])

    return slopes 