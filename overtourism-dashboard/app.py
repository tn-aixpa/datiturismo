import streamlit as st
import pandas as pd
import streamlit_option_menu
from streamlit_option_menu import option_menu
import numpy as np
import plotly.express as px
import json

st.set_page_config(
    page_title="Overtourism",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open('model.json', 'r') as file:
    configuration = json.load(file)

c = {}
constraints = []

def process_config(config):
    for cat_obj in config['categories']:
        if 'constraints' in cat_obj:
            for p in cat_obj['constraints']:
                constraints.append(p)

def compute_line(label, f1, f2, x_max, y_max):
    max = x_max + y_max
    try:
        x = 0
        p1 = [x, eval(f1)]
        if p1[1] > max: raise Exception            
    except:
        y = max
        p1 = [eval(f2), y]
    try:
        y = 0
        p2 = [eval(f2), y]
        if p2[0] > max: raise Exception
    except:
        x = max
        p2 = [x, eval(f1)]    
    return pd.DataFrame({ 'label': label, 'x': [p1[0], p2[0]], 'y': [p1[1], p2[1]]})

def compute_lines(constraints, points):
    frames = []
    for cstr in constraints:
        # y = [eval(cstr['function']) for x in range(int(points))]
        # data[cstr['label']] = y
        frames.append(compute_line(cstr['label'], cstr['fx'], cstr['fy'], c[configuration['x']['id']], c[configuration['y']['id']]))

    df = pd.concat(frames)
    return df

with st.sidebar:
    selected = option_menu(
        menu_title = "Overtourism",
        options = ["Home"],
        icons = ["house"],
        menu_icon = "cast",
        default_index = 0,
    )

if selected == "Home":
    process_config(configuration)
    # Create a row layout
    c_plot, c_form = st.columns([2,1])

    with c_form:
        c_form.subheader("Parametri dei vincoli")
        for cat_obj in configuration['categories']:
            with st.expander(cat_obj['label'], False):
                if 'parameters' in cat_obj:
                    for p in cat_obj['parameters']:            
                        c[p['id']] = st.slider(p['label'], p['min'], p['max'], p['value'], p['step'], key=p['id'])

        c_form.subheader("Parametri di visualizzazione")
        cat_obj = configuration['visualization']
        with st.expander(cat_obj['label'], False):
            if 'parameters' in cat_obj:
                for p in cat_obj['parameters']:            
                    c[p['id']] = st.slider(p['label'], p['min'], p['max'], p['value'], p['step'], key=p['id'])
    with c_plot:
        chart_data = compute_lines(constraints, 10000)
        xl = configuration['x']['label']
        yl = configuration['y']['label']
        chart_data.rename(columns={'x': xl, 'y': yl, 'label': 'Vincolo'}, inplace=True)

        bound = c[configuration['x']['id']] + c[configuration['y']['id']]
        
        fig = px.line(chart_data, x=xl, y=yl, color="Vincolo")
        fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='grey', range=[0, bound])
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='grey', range=[0, bound])
        st.plotly_chart(fig, use_container_width=True)

        c_plot.subheader("Legenda")
        for cat_obj in configuration['categories']:
            if 'constraints' in cat_obj:
                for p in cat_obj['constraints']:
                    c_plot.caption(p['label'])
                    if 'description' in p:
                        c_plot.markdown(p['description'])
                    # st.divider()
        