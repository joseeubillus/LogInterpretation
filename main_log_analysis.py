import streamlit as st
import functions as ft
    
# Main code - Front end
# Initial streamlit page configuration

st.set_page_config(page_title="Vaulted Log Data Explorer", page_icon="📊", 
                   layout="wide",initial_sidebar_state='expanded')

# Set header, subheader inside the sidebar
st.sidebar.title("Vaulted Deep")
st.sidebar.header("Log Analysis and Visualization Toolkit")
# Select tool required with a radio button
tools = st.sidebar.radio('Select tool required: ',('Visualization and processing','Merge single well log curves',
                                           'Merge single well log curves by depth'))
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