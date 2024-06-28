import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import lasio
from io import StringIO

# Functions
# Modified from andymcdo repo
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        stringio = StringIO(bytes_data.decode('Windows-1252'))
        las_file = lasio.read(stringio)
        well_data = las_file.df()

    else:
        las_file = None
        well_data = None

    return las_file, well_data

def las_header(las_file):
    if not las_file:
        st.sidebar.warning("No LAS file loaded")
    else:
        for item in las_file.well:
            st.sidebar.write(f"<b>{item.descr.capitalize()} ({item.mnemonic}):</b> {item.value}", 
            unsafe_allow_html=True)

def curve_info(las_file,well_data):
    if not las_file:
        st.warning("No LAS file loaded")
    else:
        with st.container(height=316):
            for count, curve in enumerate(las_file.curves):
                st.write(f"**{curve.mnemonic}** ({curve.unit}): {curve.descr}",
                         unsafe_allow_html=True)
                
def missing(las_file, well_data):
    
    if not las_file:
        st.warning('No file has been uploaded')
    
    else:

        data_nan = well_data.notnull().astype('int')
        # Need to setup an empty list for len check to work
        curves = []
        columns = list(well_data.columns)
        columns.pop(-1) #pop off depth

        col1_md, col2_md= st.columns(2)

        selection = col1_md.radio('Select all data or custom selection', ('All Data', 'Custom Selection'))

        if selection == 'All Data':
            curves = columns
        else:
            curves = st.multiselect('Select Curves To Plot', columns)

        if len(curves) <= 1:
            st.warning('Please select at least 2 curves.')
        else:
            curve_index = 1
            fig = make_subplots(rows=1, cols= len(curves), subplot_titles=curves, shared_yaxes=True, horizontal_spacing=0.02)

            for curve in curves:
                fig.add_trace(go.Scatter(x=data_nan[curve], y=well_data.index, 
                    fill='tozerox',line=dict(width=0)), row=1, col=curve_index)
                fig.update_xaxes(range=[0, 1], visible=False)
                fig.update_xaxes(range=[0, 1], visible=False)
                curve_index+=1
            
            fig.update_layout(height=700, showlegend=False, yaxis={'title':'DEPTH','autorange':'reversed'})
            # rotate all the subtitles of 90 degrees
            for annotation in fig['layout']['annotations']: 
                    annotation['textangle']=-90
            fig.layout.template='seaborn'
            st.plotly_chart(fig, use_container_width=True)

# Main code
# Initial streamlit page configuration
st.set_page_config(page_title="Vaulted Log Data Explorer", page_icon="📊", 
                   layout="wide",initial_sidebar_state='expanded')

# Set header, subheader inside the sidebar
st.sidebar.header("Vaulted Deep")
st.sidebar.subheader("Log Data Explorer")
# Load the LAS file
uploaded_file = None 
uploaded_file = st.sidebar.file_uploader("Upload your LAS file", type=[".las"])
# Read the LAS file
las_file, well_data = load_data(uploaded_file)
# Display the LAS header
st.sidebar.title(f'Well Header')
las_header(las_file)

# Main page
st.title('Data availability')
# Split in two columns
col1, col2 = st.columns(2)
# First column containts the Curve information
with col1:
    st.subheader('Curves Information')
    curve_info(las_file,well_data)
# Second column contains the well data description statistics
with col2:
    st.subheader('Data Description')
    if well_data is not None:
        st.write(well_data.describe())
    else:
        st.warning("No LAS file loaded")

# Missing data
st.subheader('Data Completeness')
missing(las_file, well_data)