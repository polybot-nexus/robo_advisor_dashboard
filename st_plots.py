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


def find_film_image(id):
    try:
        img = imageio.imread(
            f'C:/Users/Public/robot/OECT_demo/samples/media/{id}_raw_annealed_film.jpg'
        )
        buffered = BytesIO()
        img = Image.fromarray(img)
        return img
    except:
        print(id)
        return 0


# link is the column with hyperlinks
df['link'] = df['ID'].apply(find_film_image)


pd.set_option('display.max_colwidth', -1)


def get_thumbnail(path):
    i = Image.open(path)
    i.thumbnail((150, 150), Image.LANCZOS)
    return i


def image_base64(im):
    try:
        if isinstance(im, str):
            im = get_thumbnail(im)
        with BytesIO() as buffer:
            im.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()
    except:
        return 0


def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'


hover_data = [
    'ID',
    'thickness',
    'coating_on_top.sol_label',
    'coating_on_top.vel',
    'coating_on_top.T',
]


def update_board():

    with st.expander("Table of all Samples", True):
        hide_table_row_index = """
                    <style>
                    thead tr th:first-child {display:none}
                    tbody th {display:none}
                    </style>
                    """

        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        # st.dataframe(df.set_index('#').sort_values('transconductance', ascending=False), use_container_width=True)
        st.write(
            df.set_index('#')
            .sort_values('transconductance', ascending=False)
            .to_html(formatters={'link': image_formatter}, escape=False),
            unsafe_allow_html=True,
            use_container_width=True,
        )

    fig_col1a, fig_col2a = st.columns(2)

    with fig_col1a:
        st.markdown("#### Transconductance")
        fig = px.scatter(
            data_frame=df,
            x='#',
            y='transconductance',
            hover_data=hover_data,
            color='coating_on_top.substrate_label',
        )
        style_plot(fig)
        st.write(fig)

    with fig_col2a:
        st.markdown("#### Transconductance of devices")
        fig = px.scatter(
            data_frame=df_devices,
            x='thickness',
            y='transconductance',
            hover_data=['#'],
            color='coating_on_top.substrate_label',
        )
        style_plot(fig)
        st.write(fig)

    fig_col1b, fig_col2b = st.columns(2)

    with fig_col1b:
        st.markdown("#### Coating Speed")
        fig = px.scatter(
            data_frame=df,
            x='coating_on_top.vel',
            y='transconductance',
            hover_data=hover_data,
            color='coating_on_top.substrate_label',
        )
        style_plot(fig)
        st.write(fig)

    with fig_col2b:
        st.markdown("#### Coating Temperature")
        fig = px.scatter(
            data_frame=df,
            x='coating_on_top.T',
            y='transconductance',
            hover_data=hover_data,
            color='coating_on_top.substrate_label',
        )
        style_plot(fig)
        st.write(fig)

    fig_col1c, fig_col2c = st.columns(2)

    from natsort import index_natsorted

    with fig_col1c:
        st.markdown("#### Concentration")
        fig = px.scatter(
            data_frame=df.sort_values(
                by="coating_on_top.sol_label",
                key=lambda s: np.argsort(index_natsorted(s)),
            ),
            x="coating_on_top.sol_label",
            y='transconductance',
            hover_data=hover_data,
            color='coating_on_top.substrate_label',
        )
        style_plot(fig)
        st.write(fig)

    with fig_col2c:
        st.markdown("#### Substrate")
        fig = px.scatter(
            data_frame=df,
            x="coating_on_top.substrate_label",
            y='transconductance',
            hover_data=hover_data,
            color='coating_on_top.substrate_label',
        )
        style_plot(fig)
        st.write(fig)
