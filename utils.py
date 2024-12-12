import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


def create_plotly_stock_market_plot(oect_data):
    df = oect_data.copy()
    
    # Calculate stock market related indicators
    df['SMA_10'] = ta.sma(df['transconductance'], length=10)
    df['EMA_10'] = ta.ema(df['transconductance'], length=10)
    df['EMA_20'] = ta.ema(df['transconductance'], length=20)
    df['SMA_200'] = ta.sma(df['transconductance'], length=200)
    df['SMA_50'] = ta.sma(df['transconductance'], length=50)
    df['SMA_20'] = ta.sma(df['transconductance'], length=20)
    df['RSI'] = ta.rsi(df['transconductance'], length=14)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Transconductance and Moving Averages", "MACD Histogram"),
                        )
    
    # Top plot traces
    fig.add_trace(go.Scatter(x=df.index[21:], y=df['transconductance'][21:], 
                            mode='markers', marker=dict(color='white', size=10, line=dict(width=2, color='black')), 
                            name='GPR + RF'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index[21:], y=df['EMA_10'][21:], mode='lines', 
                            line=dict(color='black', dash='dot'), name='10-points EMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[21:], y=df['EMA_20'][21:], mode='lines', 
                            line=dict(color='black'), name='20-points EMA'), row=1, col=1)
    
    # Update layout for full width and add spike lines
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=False),
            showline=True,
            linewidth=2,
            linecolor='black',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='black',
            spikethickness=2
        ),
        xaxis2=dict(
            rangeslider=dict(visible=True),
            showline=True,
            linewidth=2,
            linecolor='black',
            rangeslider_thickness=0.1,
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='black',
            spikethickness=2
        ),
        yaxis=dict(
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='black',
            spikethickness=1
        ),
        xaxis_range=[df.index[21], df.index[-1]],
        height=600,
        # Make plot fill the container width
        width=None,
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Boxplots
    bin_width = 5
    for i in range(21, len(df['transconductance']+1), bin_width):
        bin_data = df['transconductance'][i:i + bin_width]
        if len(bin_data) == bin_width:
            box_color = '#D6B5B3' if bin_data.iloc[0] < bin_data.iloc[-1] else '#B2C8D8'
            stats = {
                'q1': np.percentile(bin_data, 25),
                'median': np.percentile(bin_data, 50),
                'q3': np.percentile(bin_data, 75),
                'whisker_low': np.min(bin_data),
                'whisker_high': np.max(bin_data)
            }
            
            x_position = i + bin_width // 2
            
            for trace in [
                go.Scatter(x=[x_position, x_position], y=[stats['q1'], stats['q3']], 
                          mode='lines', line=dict(color=box_color, width=8), showlegend=False),
                go.Scatter(x=[x_position], y=[stats['median']], mode='markers',
                          marker=dict(color='white', size=10, line=dict(width=2, color='black')), showlegend=False),
                go.Scatter(x=[x_position, x_position], y=[stats['whisker_low'], stats['whisker_high']],
                          mode='lines', line=dict(color='black', width=2), showlegend=False)
            ]:
                fig.add_trace(trace, row=1, col=1)

    # MACD subplot
    try:
        macd = ta.macd(df['transconductance'], fast=10, slow=20, signal=8)
        df = pd.concat([df, macd], axis=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_10_20_8'], mode='lines',
                                line=dict(color='black', dash='dot'), name='MACD Line'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_10_20_8'], mode='lines',
                                line=dict(color='black'), name='Signal Line'), row=2, col=1)
        
        colors = np.where(df['MACDh_10_20_8'] >= 0, '#B2C8D8', '#D6B5B3')
        fig.add_trace(go.Bar(x=df.index, y=df['MACDh_10_20_8'],
                            marker=dict(color=colors, line=dict(color='black')), name='Histogram'), row=2, col=1)
        fig.update_layout(showlegend=True, legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.5)"))
    except:
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='black'),
                                name='Empty Plot'), row=2, col=1)

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', row=2, col=1)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=1, col=1)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=2, col=1)

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