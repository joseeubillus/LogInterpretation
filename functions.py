import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import lasio
from io import StringIO
import streamlit as st

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
    
    columns = list(well_data.columns)
    st.write('Expand one of the following to visualize your well data.')
    st.write("""Each plot can be interacted with. To change the scales of a plot/track, click on the left hand or right hand side of the scale and change the value as required.""")
    with st.expander('Log Plot'):
        st.write('Select curves to plot:')
        col1, col2, col3 = st.columns(3)
        with col1:
            cal = st.selectbox('Caliper', columns,key='cal',index=None)
            gr = st.selectbox('Gamma Ray', columns,key='gr',index=None)
            sp = st.selectbox('Spontaneous Potential', columns,key='sp',index=None)
            
        with col2:
            rs = st.selectbox('Shallow Resistivity', columns,key='rs',index=None)
            rm = st.selectbox('Medium Resistivity', columns,key='rm',index=None)
            rt = st.selectbox('Deep Resistivity', columns,key='rt',index=None)
        with col3:
            
            rhob = st.selectbox('Bulk Density', 
                                  columns,key='rhob',index=None)
            nphi = st.selectbox('Neutron Porosity', columns,key='nphi',index=None)
            dt = st.selectbox('Sonic', columns,key='dt',index=None)

        curves_selected = st.button('Plot Selected Curves')

        if curves_selected:
            fig = make_subplots(rows=1, cols=4,shared_yaxes=True)
            #Update figure dimensions
            fig.update_layout(height=1000)

            # Update yaxis invert, whole number and title
            fig.update_yaxes(tickformat='.1f',autorange='reversed',domain=[0,0.8])

            # Add traces to track 1
            fig.add_trace(go.Scatter(x=well_data[cal], y=well_data.index, showlegend=False, 
                                     name='Caliper',marker=dict(color='#FE675A')), row=1, col=1)
            fig.add_trace(go.Scatter(x=well_data[gr], y=well_data.index, showlegend=False, name='Gamma Ray',xaxis='x5',marker=dict(color='#707070')))
            # Optional plot
            if sp:
                fig.add_trace(go.Scatter(x=well_data[sp], y=well_data.index, showlegend=False, name='SP',xaxis='x6',marker=dict(color='#D9D9D9')))
            
            fig.update_layout(xaxis=dict(title='CAL',titlefont=dict(color='#FE675A'),range=[0,50],anchor='free',
                                         position=0.8,dtick=5,tickfont=dict(color='#FE675A')),
                              xaxis5=dict(title='GR',anchor='free',titlefont=dict(color='#707070'),
                                          overlaying='x',side='top',position=0.9,range=[0,160],dtick=16,
                                          tickfont=dict(color='#707070')),
                                xaxis6=dict(title='SP',anchor='free',titlefont=dict(color='#D9D9D9'),
                                            overlaying='x',side='top',position=1,range=[-200,200],dtick=40,
                                            tickfont=dict(color='#D9D9D9')))
            # Add traces to track 2
            fig.add_trace(go.Scatter(x=well_data[rs], y=well_data.index, showlegend=False, 
                                     name='Shallow Resistivity',xaxis='x2',marker=dict(color='#FE675A')))
            fig.add_trace(go.Scatter(x=well_data[rm], y=well_data.index, showlegend=False, 
                                     name='Medium Resistivity',xaxis='x7',marker=dict(color='#707070')))
            fig.add_trace(go.Scatter(x=well_data[rt], y=well_data.index, showlegend=False, 
                                     name='Deep Resistivity',xaxis='x8',marker=dict(color='#D9D9D9')))
            
            fig.update_layout(xaxis2=dict(type='log',title='RS',titlefont=dict(color='#FE675A'),range=[0,3],anchor='free',
                                         position=0.8,tickfont=dict(color='#FE675A')),
                              xaxis7=dict(type='log',title='RM',anchor='free',titlefont=dict(color='#707070'),
                                          overlaying='x2',side='top',position=0.9,range=[0,3],
                                          tickfont=dict(color='#707070')),
                                xaxis8=dict(type='log',title='RT',anchor='free',titlefont=dict(color='#D9D9D9'),
                                            overlaying='x2',side='top',position=1,range=[0,3],
                                            tickfont=dict(color='#D9D9D9')))
                                     
            # Add traces to track 3
            fig.add_trace(go.Scatter(x=well_data[rhob], y=well_data.index, showlegend=False, 
                                     name='Bulk Density',xaxis='x3',marker=dict(color='#FE675A')))
            fig.add_trace(go.Scatter(x=well_data[nphi], y=well_data.index, showlegend=False, 
                                     name='Neutron Porosity',xaxis='x9',marker=dict(color='#707070')))
            
            fig.update_layout(xaxis3=dict(title='RHOB',titlefont=dict(color='#FE675A'),range=[1.95,2.95],anchor='free',
                                            position=1,tickfont=dict(color='#FE675A'),tick0=1.95,dtick=0.1),
                                xaxis9=dict(title='NPHI',anchor='free',titlefont=dict(color='#707070'),
                                            overlaying='x3',side='top',position=0.9,range=[0.45,0.15],tick0=0.45,dtick=0.03,
                                            tickfont=dict(color='#707070')))

            # Update all traces with the same ticks in x-axis
            fig.update_xaxes({'side': 'top','ticks':'outside','showline':True,
                            'showgrid':True,'showticklabels':True,
                            'linecolor':'lightgrey','gridcolor':'lightgrey',
                            'linewidth':1,'mirror':True},ticklabelposition='inside')

            # Add traces to track 4
            fig.add_trace(go.Scatter(x=well_data[dt], y=well_data.index, showlegend=False, 
                                     name='Sonic',xaxis='x4',marker=dict(color='#FE675A')))
            
            fig.update_layout(xaxis4=dict(title='DT',titlefont=dict(color='#FE675A'),range=[140,40],anchor='free',
                                         position=0.9,tickfont=dict(color='#FE675A'),dtick=10))
            # Show plot
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
        with st.expander('Calculations'):
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

def streamlit_file_uploader(label,key,accept_multiple_files=False,file_types=['las','LAS']):
    uploaded_files = None
    uploaded_files = st.sidebar.file_uploader(label, key=key, accept_multiple_files=accept_multiple_files)

    return uploaded_files

def Merge_curves(load_data):

    def combine_all_curves(df1, df2):
        available_curves1 = df1.columns
        available_curves2 = df2.columns
        common_curves = set(available_curves1) & set(available_curves2)
        uncommon_curves1 = set(available_curves1) - common_curves
        uncommon_curves2 = set(available_curves2) - common_curves

        # Combine DataFrames on depth index
        combined_data = pd.DataFrame(index=pd.concat([df1, df2], axis=0).sort_index().index.unique())

        # Process common curves
        for curve_name in common_curves:
            df1_sorted = df1[[curve_name]].sort_index()
            df2_sorted = df2[[curve_name]].sort_index()

            # Combine the curves with precedence given to df1
            combined_curve = pd.concat([df1_sorted, df2_sorted[~df2_sorted.index.isin(df1_sorted.index)]], axis=0).sort_index()

            combined_data[f"{curve_name}_combined"] = combined_curve[curve_name]

        # Process uncommon curves
        for curve_name in uncommon_curves1:
            combined_data[curve_name] = df1[curve_name]
        for curve_name in uncommon_curves2:
            combined_data[curve_name] = df2[curve_name]

        return combined_data

    
    uploaded_file = st.sidebar.file_uploader("Upload your First LAS file", type=[".las"])
    uploaded_file2 = st.sidebar.file_uploader("Upload your Second LAS file", type=[".las"])

    # Load the uploaded files
    las_file1, well_data1 = load_data(uploaded_file)
    las_file2, well_data2 = load_data(uploaded_file2)

    # Display the dataframes in the main app
    if well_data1 is not None:
        st.subheader("First LAS File Data")
        st.write(well_data1)

    if well_data2 is not None:
        st.subheader("Second LAS File Data")
        st.write(well_data2)

    # Combine and sum curves if both files are uploaded
    if well_data1 is not None and well_data2 is not None:
        combined_data = combine_all_curves(well_data1, well_data2)
    elif well_data1 is not None:
        combined_data = well_data1.copy()
    else:
        combined_data = None

    if combined_data is not None:
        st.subheader("Combined Data")
        st.write(combined_data)

def Merge_bydepth(load_data):
    def combine_selected_curves(df1, df2, selected_curves, user_depth):
        combined_data = pd.DataFrame(index=pd.concat([df1, df2], axis=0).sort_index().index.unique())

        for curve_name in selected_curves:
            # Ensure depth is sorted
            df1_sorted = df1[[curve_name]].sort_index()
            df2_sorted = df2[[curve_name]].sort_index()

            # Initialize a Series to store the combined curve
            combined_curve = pd.Series(index=combined_data.index, dtype='float64')

            # Iterate over the index to avoid duplication
            for depth in combined_curve.index:
                if depth in df1_sorted.index and depth in df2_sorted.index:
                    if depth <= user_depth:
                        combined_curve[depth] = df1_sorted[curve_name].loc[depth]
                    else:
                        combined_curve[depth] = df2_sorted[curve_name].loc[depth]
                elif depth in df1_sorted.index:
                    combined_curve[depth] = df1_sorted[curve_name].loc[depth]
                elif depth in df2_sorted.index:
                    combined_curve[depth] = df2_sorted[curve_name].loc[depth]

            # Add the combined curve to the final DataFrame
            combined_data[f"{curve_name}_combined"] = combined_curve

        return combined_data

    
    uploaded_file = st.sidebar.file_uploader("Upload your First LAS file", type=[".las"])
    uploaded_file2 = st.sidebar.file_uploader("Upload your Second LAS file", type=[".las"])

    # Load the uploaded files
    las_file1, well_data1 = load_data(uploaded_file)
    las_file2, well_data2 = load_data(uploaded_file2)

    # Get depth ranges if data is available
    if well_data1 is not None:
        min_depth1, max_depth1 = well_data1.index.min(), well_data1.index.max()

    if well_data2 is not None:
        min_depth2, max_depth2 = well_data2.index.min(), well_data2.index.max()

    # Display the dataframes in the main app
    if well_data1 is not None:
        st.subheader("First LAS File Data")
        st.write(well_data1)

    if well_data2 is not None:
        st.subheader("Second LAS File Data")
        st.write(well_data2)

    # Allow user to select curves to merge
    if well_data1 is not None and well_data2 is not None:
        available_curves1 = well_data1.columns
        available_curves2 = well_data2.columns
        common_curves = list(set(available_curves1) & set(available_curves2))

        selected_curves = st.multiselect(
            "Select curves to merge:", options=common_curves, default=common_curves[:2]
        )

        if selected_curves:
            # Input for depth range to merge
            merge_start_depth = st.number_input("Start depth for merging", value=min_depth1 if well_data1 is not None else 0)
            merge_end_depth = st.number_input("End depth for merging", value=max_depth1 if well_data1 is not None else 1000)

            # User input for the depth threshold
            user_depth = st.number_input("Enter the depth at which giving the second LAS file the priorty ", value=(merge_start_depth + merge_end_depth) / 2)

            # Filter the data based on the selected depth range
            if well_data1 is not None:
                well_data1 = well_data1[(well_data1.index >= merge_start_depth) & (well_data1.index <= merge_end_depth)]

            if well_data2 is not None:
                well_data2 = well_data2[(well_data2.index >= merge_start_depth) & (well_data2.index <= merge_end_depth)]

            combined_data = combine_selected_curves(well_data1, well_data2, selected_curves, user_depth)
        else:
            st.warning("Please select at least one curve to merge.")
            combined_data = None
    else:
        combined_data = None

    # Display the combined data
    if combined_data is not None:
        st.subheader("Combined Data")
        st.write(combined_data)
