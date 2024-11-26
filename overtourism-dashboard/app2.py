import streamlit as st
import pandas as pd
import streamlit_option_menu
from streamlit_option_menu import option_menu
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

from model import ModelViz, Context

def process_config(config):
    constraints = []
    for cat_obj in config['categories']:
        if 'constraints' in cat_obj:
            for p in cat_obj['constraints']:
                constraints.append(p)
    return constraints

def init_values(config):
    values = {}
    cat_obj = config['visualization']
    if 'parameters' in cat_obj:
        for p in cat_obj['parameters']:            
            values[p['id']] =  p['value']
                
    for cat_obj in config['categories']:
        if 'parameters' in cat_obj:
            for p in cat_obj['parameters']:     
                values[p['id']] = p['value']
    return values

def init_parameters(config):
    res = {}
    cat_obj = config['visualization']
    if 'parameters' in cat_obj:
        for p in cat_obj['parameters']:            
            res[p['id']] =  p['value']

    for cat_obj in config['categories']:
        with st.expander(cat_obj['label'], False):
            if 'parameters' in cat_obj:
                for p in cat_obj['parameters']:     
                    res[p['id']] = st.slider(p['label'], min_value=p['min'], max_value=p['max'], value=p['value'], step=p['step'], key=p['id'])
    return res

def add_scenario(context_name, config, constraints, contexts, values):
    original = init_values(config)
    diff = {}
    for k in values: 
        if values[k] != original[k]: diff[k] = values[k] - original[k] 
    mv = ModelViz(context_name, config['x'], config['y'], constraints, contexts, values, diff)
    st.session_state.models.append(mv)

@st.dialog("Aggiungere scenario")
def add_scenario_dialog(config, constraints, contexts):
    c = init_parameters(config)
    context_name = st.text_input("Nome scenario")
    
    if st.button("Aggiungi", type="primary", disabled=context_name == ""):
        if context_name != '':
            add_scenario(context_name, config, constraints, contexts, c)
            st.rerun()

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
    
    with open('model2.json', 'r') as file: 
        configuration = json.load(file)
        constraints = process_config(configuration)

    presenze = pd.read_parquet('./data.parquet')
    contexts = [
        Context("Base", presenze),
        Context("Tempo Buono", presenze),
        Context("Tempo Brutto", presenze),
    ]
    # Create a row layout
    c_plot, c_form = st.columns([2,1])

    # Parameters
    with c_form:
        if st.button("Aggiungere scenario", type="primary"):
            add_scenario_dialog(configuration, constraints, contexts)

    # Charts and legends
    with c_plot:
        context_name = st.selectbox(
            "Contesto",
            ("Base", "Tempo Buono", "Tempo Brutto"),
        )
        if 'models' not in st.session_state:
            st.session_state.models = []        
            values = init_values(configuration)
            add_scenario("Base", configuration, constraints, contexts, values)

        for i, mv in enumerate(st.session_state.models):
            st.text(mv.title())
            if mv.name != "Base":
                if st.button("Canc", type="primary", key=mv.name + "_remove", icon=":material/delete:"):
                    del st.session_state.models[i]
                    st.rerun()
            st.plotly_chart(mv.vis(context_name), use_container_width=True, key = mv.name)
            

        # Lengend
        st.subheader("Legenda")
        for cat_obj in configuration['categories']:
            if 'constraints' in cat_obj:
                for p in cat_obj['constraints']:
                    st.caption(p['label'])
                    if 'description' in p:
                        st.markdown(p['description'])
                    # st.divider()
