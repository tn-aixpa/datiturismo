import streamlit as st
import pandas as pd
import streamlit_option_menu
from streamlit_option_menu import option_menu
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

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
        if point is not None: points.append(point)
        processed.append(idx)    
    return points

#################################################################################
#####################          STREAMLIT APP              #######################
#################################################################################
st.set_page_config(
    page_title="Overtourism",
    layout="wide",
    initial_sidebar_state="expanded"
)


with st.sidebar:
    selected = option_menu(
        menu_title = "Overtourism",
        options = ["Home"],
        icons = ["house"],
        menu_icon = "cast",
        default_index = 0,
    )

if selected == "Home":
    # prepare data
    c = {}
    a = {}
    constraints = []
    with open('model.json', 'r') as file: configuration = json.load(file)
    presenze = pd.read_parquet('./data.parquet')
    process_config(configuration)
    
    # Create a row layout
    c_plot, c_form = st.columns([2,1])

    def perform(a_obj, mul):
        def _perform():
            applied = True if mul > 0 else False
            if not a_obj['multiple'] and a_obj['id'] in st.session_state and st.session_state[a_obj['id']] == applied: return
            for ap in a_obj['impact']:
                key = a_obj['id']+ ':' + ap['parameter']
                st.session_state[ap['parameter']] = st.session_state[ap['parameter']] + (a[key]*mul)
            if not a_obj['multiple']: st.session_state[a_obj['id']] = applied
        return _perform
    
    # Parameters
    with c_form:
        show_action_tab = 'show_actions' in st.session_state and st.session_state['show_actions'] == True
        if show_action_tab:
            param_tab, actions_tab, vis_tab = st.tabs([ "Parametri dei vincoli", "Azioni", "Parametri di visualizzazione"])
            with actions_tab:
                for a_obj in configuration['actions']:
                    with actions_tab.expander(a_obj['label'], False):
                        dis = not a_obj['multiple'] and a_obj['id'] in st.session_state and st.session_state[a_obj['id']] == True 
                        for ap in a_obj['impact']:
                            key = a_obj['id']+ ':' + ap['parameter']
                            a[key] = st.number_input(key=key, value = ap['value'], label=ap['label'], disabled=dis)
                        cb1, cb2 = st.columns(2)
                        with cb1:
                            st.button(key=a_obj['id']+":apply", label="Apply", type="primary", on_click=perform(a_obj, 1), disabled=dis)
                        with cb2: 
                            st.button(key=a_obj['id']+":undo", label="Undo", type="secondary", on_click=perform(a_obj, -1), disabled=not dis)
        else:
            param_tab, vis_tab = st.tabs([ "Parametri dei vincoli", "Parametri di visualizzazione"])
        
        with param_tab:                        
            # c_form.subheader("Parametri dei vincoli")
            for cat_obj in configuration['categories']:
                with st.expander(cat_obj['label'], False):
                    if 'parameters' in cat_obj:
                        for p in cat_obj['parameters']:     
                            if p['id'] not in st.session_state: st.session_state[p['id']] = p['value']
                            c[p['id']] = st.slider(p['label'], min_value=p['min'], max_value=p['max'], value=st.session_state[p['id']], step=p['step'], key=p['id'])

        with vis_tab:
            # c_form.subheader("Parametri di visualizzazione")
            cat_obj = configuration['visualization']
            with st.expander(cat_obj['label'], True):
                if 'parameters' in cat_obj:
                    for p in cat_obj['parameters']:            
                        c[p['id']] = st.slider(p['label'], p['min'], p['max'], p['value'], p['step'], key=p['id'])
                
                show_points = st.checkbox("Visualizza presenze", key="show_points")
                show_actions = st.checkbox("Visualizza azioni", key="show_actions")
                

    # Charts and legends
    with c_plot:
        chart_data = compute_lines(constraints, 10000)
    
        xl = configuration['x']['label']
        yl = configuration['y']['label']
        chart_data.rename(columns={'x': xl, 'y': yl, 'label': 'Vincolo'}, inplace=True)

        bound = c[configuration['x']['id']] + c[configuration['y']['id']]
        
        fig = px.line(chart_data, x=xl, y=yl, color="Vincolo")

        points = compute_area()
        fig.add_trace(go.Scatter(x=list(map(lambda p: p[0], points)), 
                                 y=list(map(lambda p: p[1], points)), 
                                 fill='tozeroy', showlegend=False, fillcolor='lightgrey', line_color='rgba(0,0,0,0)'))

        if 'show_points' in st.session_state and st.session_state['show_points'] == True:             
            fig.add_trace(go.Scatter(name="presenze",
                                     x=presenze['value_x']*st.session_state['p_T'],
                                     y=presenze['value_y']*st.session_state['p_E'],
                                     mode='markers', showlegend=False, text=presenze['date']))
        
        fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='grey', range=[0, bound])
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='grey', range=[0, bound])
        st.plotly_chart(fig, use_container_width=True)

        # Lengend
        st.subheader("Legenda")
        for cat_obj in configuration['categories']:
            if 'constraints' in cat_obj:
                for p in cat_obj['constraints']:
                    st.caption(p['label'])
                    if 'description' in p:
                        st.markdown(p['description'])
                    # st.divider()
