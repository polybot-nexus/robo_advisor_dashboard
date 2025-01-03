import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


def create_plotly_stock_market_plot(oect_data):
    df = oect_data.copy()

    X = df.index
    Y = df['transconductance']

    # Calculate stock market related indicators
    df['SMA_10'] = ta.sma(Y, length=10)
    df['SMA_20'] = ta.sma(Y, length=20)
    df['SMA_50'] = ta.sma(Y, length=50)
    df['SMA_200'] = ta.sma(Y, length=200)
    df['EMA_10'] = ta.ema(Y, length=10)
    df['EMA_20'] = ta.ema(Y, length=20)
    df['RSI'] = ta.rsi(Y, length=14)

    try:
        macd = ta.macd(Y, fast=10, slow=20, signal=8)
        df = pd.concat([df, macd], axis=1)
        macd_line = df['MACD_10_20_8']
        macd_signal = df['MACDs_10_20_8']
        macd_hist = df['MACDh_10_20_8']
    except Exception:
        macd = None

    # Exclude training data
    num_train = 21  # TODO: auto determine this
    x = X[num_train:]
    y = Y[num_train:]
    short_trend = df['EMA_10'][num_train:]
    long_trend = df['EMA_20'][num_train:]

    # Initialize plots
    h_slider = 0.07
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.0,
        specs=[[{}], [{'t': h_slider + 0.05}], [{}]],
        row_heights=[1e-3, 0.7, 0.2],
        # subplot_titles=("", "", ""),
    )
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True

    # Data points
    color_src = {'train': 'gray', 'gpr': '#FFF2CC', 'rf': '#3CB371'}
    csrc = df['source'].map(color_src)
    for row, mode, size, cl, c, showlegend in [
        (1, 'markers+lines', 4, 0.5, csrc, False),
        (2, 'markers', 10, 2, 'white', True),
    ]:
        fig.add_trace(
            go.Scatter(
                x=X,
                y=Y,
                mode=mode,
                marker={
                    'color': c,
                    'size': size,
                    'line': {'color': 'black', 'width': cl},
                },
                line={'color': 'black', 'width': 1},
                name='Data',
                showlegend=showlegend,
                legendgroup='1',
            ),
            row=row,
            col=1,
        )

    fig.add_vline(
        x=num_train - 0.5, line_width=1, line_dash="dashdot", line_color="gray"
    )

    # Moving average lines
    for trendline, name, style in [
        (short_trend, 'Short-term', 'dot'),
        (long_trend, 'Long-term', 'solid'),
    ]:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=trendline,
                mode='lines',
                line={'color': 'black', 'dash': style},
                name=name,
                legendgroup='1',
            ),
            row=2,
            col=1,
        )

    # Boxplots
    bin_width = 5
    for i in range(num_train, len(Y) + 1, bin_width):
        bin_data = Y[i : i + bin_width]

        if len(bin_data) != bin_width:
            break

        negative = bin_data.iloc[0] < bin_data.iloc[-1]
        box_color = '#D6B5B3' if negative else '#B2C8D8'

        stats = {
            'q1': np.percentile(bin_data, 25),
            'median': np.percentile(bin_data, 50),
            'q3': np.percentile(bin_data, 75),
            'whisker_low': np.min(bin_data),
            'whisker_high': np.max(bin_data),
        }

        x_position = i + bin_width * 0.5  # // 2

        for trace in [
            go.Scatter(
                x=[x_position, x_position],
                y=[stats['q1'], stats['q3']],
                mode='lines',
                line={'color': box_color, 'width': 8},
                showlegend=(i == num_train),
                name='[Q1, Q3]',
                legendgroup='1',
            ),
            go.Scatter(
                x=[x_position, x_position],
                y=[stats['whisker_low'], stats['whisker_high']],
                mode='lines',
                line={'color': 'black', 'width': 1},
                showlegend=(i == num_train),
                name='[Min, Max]',
                legendgroup='1',
            ),
            go.Scatter(
                x=[x_position],
                y=[stats['median']],
                mode='markers',
                marker={'color': 'black', 'size': 6},
                showlegend=(i == num_train),
                name='Median',
                legendgroup='1',
            ),
        ]:
            fig.add_trace(trace, row=2, col=1)

    # MACD subplot
    if macd is not None:
        for trendline, name, style in [
            (macd_line, 'MACD', 'dot'),
            (macd_signal, 'Signal', 'solid'),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=X,
                    y=trendline,
                    mode='lines',
                    line={'color': 'black', 'dash': style},
                    name=name,
                    legendgroup='2',
                ),
                row=3,
                col=1,
            )

        colors = np.where(macd_hist >= 0, '#B2C8D8', '#D6B5B3')
        fig.add_trace(
            go.Bar(
                x=X,
                y=macd_hist,
                marker={'color': colors, 'line': {'color': 'black'}},
                name='Histogram',
                legendgroup='2',
            ),
            row=3,
            col=1,
        )

    if macd is None: 
        # adding empty plot for if macd is None for visual purposes      
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[None] * len(x),
                mode='lines',
                #line={'color': 'black', 'dash': 'dot'},
                #name='MACD',
                legendgroup='2',
                showlegend=False
            ),
            row=3,
            col=1,
        )

    # Update layout for full width and add spike lines
    default_font = {
        'family': 'Arial',
        'size': 17,
        'color': 'black',
    }

    default_values = {
        'showline': True,
        'linewidth': 2,
        'linecolor': 'black',
        'mirror': True,
    }

    add_spike_line = {
        'showspikes': True,
        'spikemode': 'across+marker',
        'spikesnap': 'data',  # cursor
        'spikecolor': '#119DFF',
        'spikethickness': 0.5,
    }

    add_rangeslider = {
        'rangeslider': {
            'visible': True,
            'borderwidth': 1,
            'bgcolor': 'white',
            'thickness': h_slider,
        },
    }

    add_legend = {
        'showlegend': True,
        'legend': {
            'x': 0.0,
            'y': 0.93 - h_slider,
            'tracegroupgap': 200,
            'bgcolor': "rgba(255, 255, 255, 0.9)",
            'font': default_font,
        },
    }

    add_y_zeroline = {
        'zeroline': True,
        'zerolinecolor': 'gray',
        'zerolinewidth': 1,
    }

    # fig.update_traces(xaxis="x3")

    fig.update_layout(
        # Make plot fill the container width
        width=None,
        height=None,
        autosize=True,
        margin=dict(l=10, r=100, t=10, b=50),
        # Plots
        title_text='',
        paper_bgcolor='white',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis={
            **default_values,
            'showline': False,
            **add_rangeslider,
            **add_spike_line,
        },
        xaxis2={
            **default_values,
            'side': 'top',
            'mirror': False,
            **add_spike_line,
        },
        xaxis3={
            **default_values,
            'title_text': 'Sample',
            **add_spike_line,
        },
        yaxis={
            **default_values,
            'showline': False,
            'showticklabels': False,
        },
        yaxis2={
            **default_values,
            'title_text': 'μC* (F cm⁻¹ V⁻¹ s⁻¹)',
            'side': 'right',
            **add_spike_line,
        },
        yaxis3={
            **default_values,
            'showticklabels': False,
            **add_y_zeroline,
        },
        xaxis_range=[x[0], x[-1]],
        **add_legend,
    )

    fig.update_annotations({'font': default_font})
    fig.update_xaxes(**add_spike_line, tickfont=default_font)
    fig.update_yaxes(tickfont=default_font)

    def add_arrow(x0, x1, color, text):
        fig.add_annotation(
            {
                'x': x1 - 0.45,
                'y': 0,
                'xref': 'x',
                'yref': 'y',
                'ax': x0 - 0.7,
                'ay': 0,
                'axref': 'x',
                'ayref': 'y',
                'showarrow': True,
                'arrowhead': 2,
                'arrowsize': 0.7,
                'arrowwidth': 7,
                'arrowcolor': color,
            }
        )
        fig.add_annotation(
            {
                'x': 0.5 * (x1 + x0 - 1),
                'y': 0,
                'yshift': 1.5,
                'xref': 'x',
                'yref': 'y',
                'text': f"   {text}   ",
                'showarrow': False,
                'font': {**default_font, 'size': 19},
                'bordercolor': color,
                'bgcolor': 'white',
                'borderwidth': 2,
            }
        )

    x0, x1, src0 = 0, 0, df['source'].iloc[0] if not df['source'].empty else 0 #df['source'][0]
    for src in df['source'][1:]:
        x1 += 1
        if src != src0:
            add_arrow(x0, x1, color_src[src0], src0)
            x0 = x1
            src0 = src
    add_arrow(x0, x1 + 1, color_src[src0], src0)

    return fig


def get_trendline_slope(oect_data):
    x = oect_data['#']
    y = oect_data['transconductance']

    # Create and smooth the spline curve
    spl = UnivariateSpline(x, y, s=0.1, k=3)
    xs = np.linspace(min(x), max(x), x.shape[0] * 5)
    ys = spl(xs)
    smoothed_ys = savgol_filter(ys, 70, 4)

    # Calculate the slope for the tangent line at the last point
    slopes = np.gradient(smoothed_ys, xs)
    slope_at_last_smoothed = slopes[-1]

    # Calculate the tangent line at the last point
    tangent_x_smoothed = np.array([xs[-1] - 2, xs[-1] + 2])
    tangent_y_smoothed = smoothed_ys[-1] + slope_at_last_smoothed * (
        tangent_x_smoothed - xs[-1]
    )

    return slopes
