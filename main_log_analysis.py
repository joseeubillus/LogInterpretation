import streamlit as st
import functions as ft
from plotly.subplots import make_subplots
import plotly.graph_objects as go
    
# Main code - Front end
# Initial streamlit page configuration

st.set_page_config(page_title="Vaulted Log Data Explorer", page_icon="📊", 
                   layout="wide",initial_sidebar_state='expanded')

# Set header, subheader inside the sidebar
st.sidebar.title("Vaulted Deep")
st.sidebar.header("Log Analysis and Visualization Toolkit")
# Select tool required with a radio button
tools = st.sidebar.radio('Select tool required: ',('Visualization and processing','Merge single well log curves',
                                           'Merge single well log curves by depth','Well Correlation', 'Blocking'))
if tools == 'Merge single well log curves':
    st.title(tools)
    combined = ft.merge_single_well_curves()
    if combined is None:
        st.warning('No files have been uploaded')
    else:
        st.subheader('Data Completeness')
        ft.missing(combined)
        st.subheader('Data Visualization')
        ft.plot(combined)
elif tools == 'Merge single well log curves by depth':
    st.title(tools)
    combined = ft.merge_by_depth()
    if combined is None:
        st.warning('No files have been uploaded')
    else:
        st.subheader('Data Completeness')
        ft.missing(combined)
        st.subheader('Data Visualization')
        ft.plot(combined)
elif tools == 'Well Correlation':
    st.title(tools)
    uploaded_files = ft.streamlit_file_uploader("Upload your wells LAS files",
                                            key='correlation',
                                            accept_multiple_files=True)
    
    las_files, wells_data = ft.load_multiple_las_files(uploaded_files)

    zones = []
    ft.well_correlation(las_files, wells_data)
elif tools == 'Blocking':
    st.title(tools)
    uploaded_files = ft.streamlit_file_uploader("Upload your wells LAS files",
                                            key='blocking',
                                            accept_multiple_files=False)
    las_file, well_data = ft.load_data(uploaded_files)
    if las_file:
        # Use PELT algorithm to detect change points
        columns = st.multiselect('Select the column to block', list(well_data.columns),key='block_select')
        penalty = st.slider(f'Penalty Value for PELT Segmentation ', 1, 10, 3, key=f'penalty_well')
        depth_interval = st.number_input(f'Set Depth Interval (ft)', min_value=1, max_value=50, value=5, key=f'depth_interval_well')
        subsample_rate = st.slider(f'Subsample Rate for PELT Segmentation', 1, 20, 10, step=1, key=f'subsample_rate_well')
        # Display the original curves in different subplots using plotly
        # Count number of columns in columns

        # Iterate ft.pelt segmentation for each colum selected
        if len(columns) > 0:
            num_columns = len(columns)
            fig = make_subplots(rows=1, cols=num_columns, shared_yaxes=True)
            for column in columns:
                
                segmented_curve = ft.pelt_segmentation(well_data, column, depth_interval=depth_interval, penalty=penalty, subsample_rate=subsample_rate)
                # Save the segmented curve to the well data
                well_data[column + '_segmented'] = segmented_curve
                # Plot the original and segmented curve
                fig.add_trace(go.Scatter(x=well_data[column], y=well_data.index, name=f'{column} Original'), 
                            row=1, col=columns.index(column) + 1)
                fig.add_trace(go.Scatter(x=segmented_curve, y=well_data.index, name=f'{column} Segmented',marker=dict(color='black')), 
                            row=1, col=columns.index(column) + 1)
                
                # Update name in x-axis
                fig.update_xaxes(title_text=column, row=1, col=columns.index(column) + 1)

            fig.update_layout(height=1500)
            fig.update_yaxes(autorange="reversed",tickformat=".1f",title_text='Depth (ft)', row=1, col=1)
                # Update all traces with the same ticks in x-axis
            fig.update_xaxes({'side': 'top','ticks':'outside','showline':True,
                            'showgrid':True,'showticklabels':True,
                            'linecolor':'lightgrey','gridcolor':'lightgrey',
                            'linewidth':1,'mirror':True},ticklabelposition='inside')
            st.plotly_chart(fig)
        else:
            st.warning('No columns have been selected')
    else:
        st.warning('No file has been uploaded')
    
else:
    st.title(tools)
    uploaded_file = ft.streamlit_file_uploader("Upload your LAS file",
                                            key='viz',
                                            accept_multiple_files=False
                                            ,file_types=['las','LAS'])
    # Read the LAS file
    las_file, well_data = ft.load_data(uploaded_file)
    # Display the LAS header
    st.sidebar.header(f'Well Header')
    ft.las_header(las_file)
    # Main page alert that not LAS file has been uploaded
    if las_file:

        # Main page
        st.header('Data Description')
     
        st.subheader('Curves Information')
        ft.curve_info(las_file,well_data)

        st.subheader('Data Description')
        if well_data is not None:
            st.write(well_data.describe())
        else:
            st.warning("No LAS file loaded")

        # Missing data
        st.header('Data Completeness')
        ft.missing(well_data)

        # Data visualization
        st.header('Data Visualization')
        ft.plot(well_data)

        # Post-processing
        st.header('Post-processing')
        ft.post_processing(las_file, well_data)
    else:
        st.warning('No file has been uploaded')