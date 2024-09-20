import streamlit as st
import pandas as pd
import streamlit_option_menu
from streamlit_option_menu import option_menu
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

def compute_line(label, params, x_max, y_max):
    max = x_max + y_max
    r = list(map(lambda x: eval(x),params))
    try:
        x = 0
        p1 = [x, (r[2] - r[0]*x)/(r[1] if r[1] != 0 else 0.01)]
        if p1[1] > max: raise Exception            
    except:
        y = max
        p1 = [(r[2] - r[1]*y)/(r[0] if r[0] != 0 else 0.01), y]
    try:
        y = 0
        p2 = [(r[2] - r[1]*y)/(r[0] if r[0] != 0 else 0.01), y]
        if p2[0] > max: raise Exception
    except:
        x = max
        p2 = [x, (r[2] - r[0]*x)/(r[1] if r[1] != 0 else 0.01)]    
    return pd.DataFrame({ 'label': label, 'x': [p1[0], p2[0]], 'y': [p1[1], p2[1]]})

def compute_lines(constraints, points):
    frames = []
    for cstr in constraints:
        frames.append(compute_line(cstr['label'], cstr['r'], c[configuration['x']['id']], c[configuration['y']['id']]))

    df = pd.concat(frames)
    return df

def compute_area():
    def intersect(r1, r2, matrix):
        div = (r1[0]*r2[1]-r2[0]*r1[1])
        if div == 0: div = 0.0000000001
        return [round((-r1[1]*r2[2]+r2[1]*r1[2])/div,2), round((-r1[2]*r2[0]+r2[2]*r1[0])/div,2)]
    def compute_matrix(constraints):
        res = []
        for cstr in constraints:
            r = list(map(lambda x: eval(x),cstr['r']))
            res.append(r)
        return res
    matrix = compute_matrix(constraints)
    min = None
    idx = 0
    for i in range(len(matrix)):
        p = intersect([1,0,0],matrix[i],matrix)
        if p[1] >= 0 and (min == None or min > p[1]):
            min = p[1]
            idx = i
    
    points = [[0,min]]
    
    processed = [idx]
    while len(processed) < len(matrix):
        min = None
        r1 = matrix[idx]
        point = None
        for i in range(len(matrix)):
            if i not in processed:
                r2 = matrix[i]
                p = intersect(r1, r2, matrix)
                if p[0] >= 0 and p[0] > points[-1][0] and (min == None or min > p[0]):
                    min = p[0]
                    idx = i
                    point = p
                    print(i, p, min)
        if point is not None: points.append(point)
        processed.append(idx)    
    return points
    
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

        points = compute_area()
        fig.add_trace(go.Scatter(x=list(map(lambda p: p[0], points)), y=list(map(lambda p: p[1], points)), fill='tozeroy', showlegend=False, fillcolor='lightgrey', line_color='rgba(0,0,0,0)'))
        
        
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
        