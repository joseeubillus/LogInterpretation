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

        data_not_nan = well_data.notnull().astype('int')
        data_nan = well_data.isnull().astype('int')
        # Need to setup an empty list for len check to work
        curves = []
        columns = list(well_data.columns)
        columns.pop(-1) #pop off depth

        selection = st.radio('Select all data or custom selection', ('All Data', 'Custom Selection'))

        if selection == 'All Data':
            curves = columns
        else:
            curves = st.multiselect('Select Curves To Plot', columns)

        if len(curves) <= 1:
            st.warning('Please select at least 2 curves.')
        else:
            curve_index = 2
            fig = make_subplots(rows=1, cols= len(curves), subplot_titles=curves, shared_yaxes=True, horizontal_spacing=0.02)
            
            fig.add_trace(go.Scatter(x=data_not_nan[curves[0]], y=well_data.index, 
                    fill='tozerox',line=dict(width=0),fillcolor='#D3CBCB',showlegend=True,name='Complete'), row=1, col=1)
            fig.add_trace(go.Scatter(x=data_nan[curves[0]], y=well_data.index,
                    fill='tozerox',line=dict(width=0),fillcolor='#FF807E',showlegend=True,name='Missing'), row=1, col=1)
            for curve in curves[1:]:
                fig.add_trace(go.Scatter(x=data_not_nan[curve], y=well_data.index, 
                    fill='tozerox',line=dict(width=0),fillcolor='#D3CBCB',showlegend=False), row=1, col=curve_index)
                fig.add_trace(go.Scatter(x=data_nan[curve], y=well_data.index,
                    fill='tozerox',line=dict(width=0),fillcolor='#FF807E',showlegend=False), row=1, col=curve_index)
                fig.update_xaxes(range=[0, 1], visible=False)
                fig.update_xaxes(range=[0, 1], visible=False)
                curve_index+=1
            
            fig.update_layout(height=700, showlegend=True, yaxis={'title':'DEPTH','autorange':'reversed'})
            # rotate all the subtitles of 90 degrees
            for annotation in fig['layout']['annotations']: 
                    annotation['textangle']=-90
            fig.layout.template='seaborn'
            st.plotly_chart(fig, use_container_width=True)

def plot(las_file, well_data):
    
    if not las_file:
        st.warning('No file has been uploaded')
    
    else:
        columns = list(well_data.columns)
        st.write('Expand one of the following to visualize your well data.')
        st.write("""Each plot can be interacted with. To change the scales of a plot/track, click on the left hand or right hand side of the scale and change the value as required.""")
        with st.expander('Log Plot'):    
            curves = st.multiselect('Select Curves To Plot', columns)
            if len(curves) <= 1:
                st.warning('Please select at least 2 curves.')
            else:
                curve_index = 1
                fig = make_subplots(rows=1, cols= len(curves), shared_yaxes=True)
                fig.update_xaxes({'side': 'top','ticks':'outside','showline':True,
                                  'showgrid':True,'showticklabels':True,
                                  'linecolor':'lightgrey','gridcolor':'lightgrey',
                                  'linewidth':1,'mirror':True})
                for curve in curves:

                    fig.add_trace(go.Scatter(x=well_data[curve], y=well_data.index), row=1, col=curve_index)
                    fig.update_xaxes(title_text=curve, row=1, col=curve_index)
                    if curve.startswith('RES') ==True:
                        fig.add_trace(go.Scatter(x=well_data[curve], y=well_data.index), row=1, col=curve_index)
                        fig.update_xaxes(type='log',row=1,col=curve_index)
                    curve_index+=1

                fig.update_layout(height=1000,
                                   yaxis={'title':'DEPTH','autorange':'reversed'},showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with st.expander('Histograms'):
            col1_h, col2_h = st.columns(2)
            col1_h.header('Options')

            hist_curve = col1_h.selectbox('Select a Curve', columns)
            log_option = col1_h.radio('Select Linear or Logarithmic Scale', ('Linear', 'Logarithmic'))
            
            if log_option == 'Linear':
                log_bool = False
            elif log_option == 'Logarithmic':
                log_bool = True

            histogram = px.histogram(well_data, x=hist_curve, log_x=log_bool)
            histogram.update_traces(marker_color='#FF807E')
            histogram.layout.template='seaborn'
            col2_h.plotly_chart(histogram, use_container_width=True)

        with st.expander('Crossplot'):
            col1, col2 = st.columns(2)
            col1.write('Options')

            xplot_x = col1.selectbox('X-Axis', columns)
            xplot_y = col1.selectbox('Y-Axis', columns)
            xplot_col = col1.selectbox('Colour By', columns)
            xplot_x_log = col1.radio('X Axis - Linear or Logarithmic', ('Linear', 'Logarithmic'))
            xplot_y_log = col1.radio('Y Axis - Linear or Logarithmic', ('Linear', 'Logarithmic'))

            if xplot_x_log == 'Linear':
                xplot_x_bool = False
            elif xplot_x_log == 'Logarithmic':
                xplot_x_bool = True
            
            if xplot_y_log == 'Linear':
                xplot_y_bool = False
            elif xplot_y_log == 'Logarithmic':
                xplot_y_bool = True

            xplot = px.scatter(well_data, x=xplot_x, y=xplot_y, color=xplot_col, log_x=xplot_x_bool, log_y=xplot_y_bool)
            xplot.layout.template='seaborn'
            col2.plotly_chart(xplot, use_container_width=True)

def post_processing(las_file, well_data):
    if not las_file:
        st.warning('No file has been uploaded')

    else:
        with st.expander('CaLculations'):
            calcs = st.selectbox('Select a Calculation', ('None', 
                                                           'Shale Volume (Vsh)'))
            if calcs == 'Shale Volume (Vsh)':
                GR = st.selectbox('Select a Gamma Ray Curve', well_data.columns, index=None)
                if GR == None:
                    st.warning('Please select a Gamma Ray curve.')
                else:
                    well_data['VSH'] = (well_data[GR] - well_data[GR].min()) / (well_data[GR].max() - well_data[GR].min())
                    
                    # Plot the VSH curve versus depth and the net sand (net sand is estimated with an if condition if Vsh<0.5,sand,else shale)
                    fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
                    fig.add_trace(go.Scatter(x=well_data['VSH'], y=well_data.index,showlegend=False), row=1, col=1)
                    fig.update_xaxes(title_text='Vsh', row=1, col=1)
                    #Update the x-axis ticks
                    fig.update_xaxes({'side': 'top','ticks':'outside','showline':True,
                                    'showgrid':True,'showticklabels':True,
                                    'linecolor':'lightgrey','gridcolor':'lightgrey',
                                    'linewidth':1,'mirror':True})
                    # Create net sand variable
                    well_data['NET_SAND'] = np.where(well_data['VSH'] < 0.5, 1, 0)
                    # Fill the net sand with color
                    fig.add_trace(go.Scatter(x=well_data['NET_SAND'], y=well_data.index, fill='tozeroy',mode='lines', line=dict(width=0), 
                                             fillcolor='#ffbd2e',showlegend=True,name='Sand'), row=1, col=2)
                    fig.update_xaxes(title_text='Net Sand',row=1, col=2)
                    fig.update_layout(height=1000, yaxis={'title':'DEPTH','autorange':'reversed'},showlegend=True)
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

# Data visualization
st.subheader('Data Visualization')
plot(las_file, well_data)

# Post-processing
st.subheader('Post-processing')
post_processing(las_file, well_data)