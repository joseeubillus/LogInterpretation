import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import lasio
from io import StringIO

# Functions


@st.cache_data
def load_data(uploaded_file):
    """Load the LAS file and convert it to a DataFrame."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        stringio = StringIO(bytes_data.decode("Windows-1252"))
        las_file = lasio.read(stringio)
        well_data = las_file.df()
    else:
        las_file = None
        well_data = None
    return las_file, well_data


def combine_all_curves(df1, df2):
    """Combine all common curves from two DataFrames based on depth."""
    available_curves1 = df1.columns
    available_curves2 = df2.columns
    common_curves = set(available_curves1) & set(available_curves2)

    combined_data = pd.DataFrame(index=pd.concat([df1, df2], axis=0).sort_index().index)

    for curve_name in common_curves:
        # Ensure depth is sorted
        df1_sorted = df1[[curve_name]].sort_index()
        df2_sorted = df2[[curve_name]].sort_index()

        # Combine the DataFrames on depth index
        temp_combined_df = pd.concat([df1_sorted, df2_sorted], axis=0).sort_index()

        # Sum the specified curve
        temp_combined_df[f"{curve_name}_combined"] = (
            temp_combined_df[curve_name].groupby(temp_combined_df.index).sum()
        )

        # Add the combined curve to the final DataFrame
        combined_data[f"{curve_name}_combined"] = temp_combined_df[
            f"{curve_name}_combined"
        ]

    return combined_data


def las_header(las_file):
    """Display the LAS file header information."""
    if not las_file:
        st.sidebar.warning("No LAS file loaded")
    else:
        for item in las_file.well:
            st.sidebar.write(
                f"<b>{item.descr.capitalize()} ({item.mnemonic}):</b> {item.value}",
                unsafe_allow_html=True,
            )


def curve_info(las_file, well_data):
    """Display the curve information from the LAS file."""
    if not las_file:
        st.warning("No LAS file loaded")
    else:
        with st.container():
            for count, curve in enumerate(las_file.curves):
                st.write(
                    f"**{curve.mnemonic}** ({curve.unit}): {curve.descr}",
                    unsafe_allow_html=True,
                )


def missing(las_file, well_data):
    """Visualize missing data in the well log data."""
    if not las_file:
        st.warning("No file has been uploaded")
    else:
        data_not_nan = well_data.notnull().astype("int")
        data_nan = well_data.isnull().astype("int")
        curves = []
        columns = list(well_data.columns)
        columns.pop(-1)  # pop off depth

        selection = st.radio(
            "Select all data or custom selection", ("All Data", "Custom Selection")
        )

        if selection == "All Data":
            curves = columns
        else:
            curves = st.multiselect("Select Curves To Plot", columns)

        if len(curves) <= 1:
            st.warning("Please select at least 2 curves.")
        else:
            curve_index = 2
            fig = make_subplots(
                rows=1,
                cols=len(curves),
                subplot_titles=curves,
                shared_yaxes=True,
                horizontal_spacing=0.02,
            )

            fig.add_trace(
                go.Scatter(
                    x=data_not_nan[curves[0]],
                    y=well_data.index,
                    fill="tozerox",
                    line=dict(width=0),
                    fillcolor="#D3CBCB",
                    showlegend=True,
                    name="Complete",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=data_nan[curves[0]],
                    y=well_data.index,
                    fill="tozerox",
                    line=dict(width=0),
                    fillcolor="#FF807E",
                    showlegend=True,
                    name="Missing",
                ),
                row=1,
                col=1,
            )
            for curve in curves[1:]:
                fig.add_trace(
                    go.Scatter(
                        x=data_not_nan[curve],
                        y=well_data.index,
                        fill="tozerox",
                        line=dict(width=0),
                        fillcolor="#D3CBCB",
                        showlegend=False,
                    ),
                    row=1,
                    col=curve_index,
                )
                fig.add_trace(
                    go.Scatter(
                        x=data_nan[curve],
                        y=well_data.index,
                        fill="tozerox",
                        line=dict(width=0),
                        fillcolor="#FF807E",
                        showlegend=False,
                    ),
                    row=1,
                    col=curve_index,
                )
                fig.update_xaxes(range=[0, 1], visible=False, row=1, col=curve_index)
                curve_index += 1

            fig.update_layout(
                height=700,
                showlegend=True,
                yaxis={"title": "DEPTH", "autorange": "reversed"},
            )
            for annotation in fig["layout"]["annotations"]:
                annotation["textangle"] = -90
            fig.layout.template = "seaborn"
            st.plotly_chart(fig, use_container_width=True)


def calculate_log_average(df, curve, interval=1000):
    """Calculate the average of a log curve over specified intervals."""
    df["Interval"] = (df.index // interval) * interval
    return df.groupby("Interval").mean().reset_index()


def plot(las_file, well_data, zones, cutoffs):
    """Generate different types of plots for the well log data with adjustable axes and zonation."""
    if not las_file:
        st.warning("No file has been uploaded")
    else:
        columns = list(well_data.columns)
        st.write("Expand one of the following to visualize your well data.")
        st.write(
            """Each plot can be interacted with. To change the scales of a plot/track, click on the left hand or right hand side of the scale and change the value as required."""
        )

        with st.expander("Log Plot"):
            st.write(
                "Select the curves you want to plot. You can select up to three curves to be plotted on the same track."
            )

            num_tracks = st.slider("Number of Tracks", 1, 5, 1)
            fig = make_subplots(
                rows=num_tracks, cols=1, shared_xaxes=True, vertical_spacing=0.02
            )

            for i in range(1, num_tracks + 1):
                with st.container():
                    primary_curve = st.selectbox(
                        f"Select Primary Curve for Track {i}",
                        columns,
                        key=f"primary_{i}",
                    )
                    secondary_curve = st.selectbox(
                        f"Select Secondary Curve for Track {i} (optional)",
                        ["None"] + columns,
                        key=f"secondary_{i}",
                    )
                    tertiary_curve = st.selectbox(
                        f"Select Tertiary Curve for Track {i} (optional)",
                        ["None"] + columns,
                        key=f"tertiary_{i}",
                    )

                    x_axis_min = st.number_input(
                        f"Set Minimum X-Axis Value for {primary_curve}",
                        value=well_data[primary_curve].min(),
                        key=f"min_{primary_curve}_{i}",
                    )
                    x_axis_max = st.number_input(
                        f"Set Maximum X-Axis Value for {primary_curve}",
                        value=well_data[primary_curve].max(),
                        key=f"max_{primary_curve}_{i}",
                    )

                    if secondary_curve != "None":
                        x_axis_min_sec = st.number_input(
                            f"Set Minimum X-Axis Value for {secondary_curve}",
                            value=well_data[secondary_curve].min(),
                            key=f"min_{secondary_curve}_{i}",
                        )
                        x_axis_max_sec = st.number_input(
                            f"Set Maximum X-Axis Value for {secondary_curve}",
                            value=well_data[secondary_curve].max(),
                            key=f"max_{secondary_curve}_{i}",
                        )
                    else:
                        x_axis_min_sec = None
                        x_axis_max_sec = None

                    if tertiary_curve != "None":
                        x_axis_min_ter = st.number_input(
                            f"Set Minimum X-Axis Value for {tertiary_curve}",
                            value=well_data[tertiary_curve].min(),
                            key=f"min_{tertiary_curve}_{i}",
                        )
                        x_axis_max_ter = st.number_input(
                            f"Set Maximum X-Axis Value for {tertiary_curve}",
                            value=well_data[tertiary_curve].max(),
                            key=f"max_{tertiary_curve}_{i}",
                        )
                    else:
                        x_axis_min_ter = None
                        x_axis_max_ter = None

                    if primary_curve:
                        fig.add_trace(
                            go.Scatter(
                                x=well_data[primary_curve],
                                y=well_data.index,
                                name=primary_curve,
                                line=dict(color="blue"),
                            ),
                            row=i,
                            col=1,
                        )
                        fig.update_xaxes(
                            title_text=primary_curve,
                            row=i,
                            col=1,
                            range=[x_axis_min, x_axis_max],
                        )

                    if secondary_curve != "None":
                        fig.add_trace(
                            go.Scatter(
                                x=well_data[secondary_curve],
                                y=well_data.index,
                                name=secondary_curve,
                                line=dict(color="red"),
                            ),
                            row=i,
                            col=1,
                        )
                        fig.update_xaxes(
                            title_text=f"{primary_curve} and {secondary_curve}",
                            row=i,
                            col=1,
                            range=[
                                min(x_axis_min, x_axis_min_sec),
                                max(x_axis_max, x_axis_max_sec),
                            ],
                        )

                    if tertiary_curve != "None":
                        fig.add_trace(
                            go.Scatter(
                                x=well_data[tertiary_curve],
                                y=well_data.index,
                                name=tertiary_curve,
                                line=dict(color="green"),
                            ),
                            row=i,
                            col=1,
                        )
                        fig.update_xaxes(
                            title_text=f"{primary_curve}, {secondary_curve}, and {tertiary_curve}",
                            row=i,
                            col=1,
                            range=[
                                min(x_axis_min, x_axis_min_sec, x_axis_min_ter),
                                max(x_axis_max, x_axis_max_sec, x_axis_max_ter),
                            ],
                        )

                    if (
                        primary_curve.startswith("RES")
                        or (
                            secondary_curve != "None"
                            and secondary_curve.startswith("RES")
                        )
                        or (
                            tertiary_curve != "None"
                            and tertiary_curve.startswith("RES")
                        )
                    ):
                        fig.update_xaxes(type="log", row=i, col=1)

            # Add zonation as horizontal bands
            if zones:
                for zone in zones:
                    fig.add_hrect(
                        y0=zone["start"],
                        y1=zone["end"],
                        fillcolor="LightSalmon",
                        opacity=0.5,
                        layer="below",
                        line_width=0,
                        annotation_text=zone["label"],
                        annotation_position="top left",
                    )

            # Add cutoffs as vertical lines
            if cutoffs:
                for cutoff in cutoffs:
                    if cutoff["value"] is not None:
                        fig.add_vline(
                            x=cutoff["value"],
                            line=dict(color="RoyalBlue", width=2, dash="dash"),
                            annotation_text=f"{cutoff['label']}: {cutoff['value']}",
                            annotation_position="top right",
                        )

            fig.update_layout(
                height=800,
                showlegend=True,
                xaxis_title="Value",
                yaxis_title="Depth",
                yaxis_autorange="reversed",
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Average Log Plot"):
            st.write("Select the curve and the interval for averaging.")

            avg_curve = st.selectbox(
                "Select Curve for Averaging", columns, key="avg_curve"
            )
            avg_interval = st.number_input(
                "Set Averaging Interval (Depth units)", value=1000, min_value=1, step=1
            )

            if avg_curve:
                averaged_data = calculate_log_average(
                    well_data, avg_curve, interval=avg_interval
                )
                fig_avg = go.Figure()
                fig_avg.add_trace(
                    go.Scatter(
                        x=averaged_data[avg_curve],
                        y=averaged_data["Interval"],
                        mode="lines+markers",
                        name=f"Average of {avg_curve}",
                    )
                )
                fig_avg.update_layout(
                    height=600,
                    yaxis={"title": "DEPTH", "autorange": "reversed"},
                    xaxis={"title": avg_curve},
                    title=f"Average Log Plot for {avg_curve}",
                )
                st.plotly_chart(fig_avg, use_container_width=True)

        with st.expander("Histograms"):
            col1_h, col2_h = st.columns(2)
            col1_h.header("Options")

            hist_curve = col1_h.selectbox("Select a Curve", columns)
            log_option = col1_h.radio(
                "Select Linear or Logarithmic Scale", ("Linear", "Logarithmic")
            )

            log_bool = log_option == "Logarithmic"

            histogram = px.histogram(well_data, x=hist_curve, log_x=log_bool)
            histogram.update_traces(marker_color="#FF807E")
            histogram.layout.template = "seaborn"
            col2_h.plotly_chart(histogram, use_container_width=True)

        with st.expander("Crossplot"):
            col1, col2 = st.columns(2)
            col1.write("Options")

            xplot_x = col1.selectbox("X-Axis", columns)
            xplot_y = col1.selectbox("Y-Axis", columns)
            xplot_col = col1.selectbox("Colour By", columns)
            xplot_x_log = col1.radio(
                "X Axis - Linear or Logarithmic", ("Linear", "Logarithmic")
            )
            xplot_y_log = col1.radio(
                "Y Axis - Linear or Logarithmic", ("Linear", "Logarithmic")
            )

            xplot_x_bool = xplot_x_log == "Logarithmic"
            xplot_y_bool = xplot_y_log == "Logarithmic"

            xplot = px.scatter(
                well_data,
                x=xplot_x,
                y=xplot_y,
                color=xplot_col,
                log_x=xplot_x_bool,
                log_y=xplot_y_bool,
            )
            xplot.layout.template = "seaborn"
            col2.plotly_chart(xplot, use_container_width=True)


# Streamlit app setup


def main():
    st.title("Well Log Data Visualization")

    # File upload
    uploaded_file1 = st.file_uploader("Upload LAS file 1", type=["las"])
    uploaded_file2 = st.file_uploader("Upload LAS file 2", type=["las"])

    las_file1, well_data1 = load_data(uploaded_file1)
    las_file2, well_data2 = load_data(uploaded_file2)

    if las_file1 and las_file2:
        combined_well_data = combine_all_curves(well_data1, well_data2)
        st.write("Combined LAS file data loaded.")
    else:
        combined_well_data = well_data1 if well_data1 is not None else well_data2

    if combined_well_data is not None:
        las_header(las_file1 or las_file2)
        curve_info(las_file1 or las_file2, combined_well_data)

        zones = []
        if st.checkbox("Add Zonation"):
            num_zones = st.number_input(
                "Number of Zones", min_value=1, max_value=10, value=1
            )
            for i in range(num_zones):
                st.subheader(f"Zone {i+1}")
                label = st.text_input(f"Label for Zone {i+1}", value=f"Zone {i+1}")
                start = st.number_input(
                    f"Start Depth for Zone {i+1}", value=combined_well_data.index.min()
                )
                end = st.number_input(
                    f"End Depth for Zone {i+1}", value=combined_well_data.index.max()
                )
                zones.append({"label": label, "start": start, "end": end})

        cutoffs = []
        if st.checkbox("Add Cutoffs"):
            num_cutoffs = st.number_input(
                "Number of Cutoffs", min_value=1, max_value=10, value=1
            )
            for i in range(num_cutoffs):
                st.subheader(f"Cutoff {i+1}")
                label = st.text_input(f"Label for Cutoff {i+1}", value=f"Cutoff {i+1}")
                value = st.number_input(f"Value for Cutoff {i+1}")
                cutoffs.append({"label": label, "value": value})

        option = st.sidebar.selectbox(
            "Select an option:", ["Missing Data", "Log Plot", "Average Plot"]
        )

        if option == "Missing Data":
            missing(las_file1 or las_file2, combined_well_data)
        elif option == "Log Plot":
            plot(las_file1 or las_file2, combined_well_data, zones, cutoffs)
        elif option == "Average Plot":
            st.write("Select the curve and the interval for averaging.")
            avg_curve = st.selectbox(
                "Select Curve for Averaging",
                combined_well_data.columns,
                key="avg_curve",
            )
            avg_interval = st.number_input(
                "Set Averaging Interval (Depth units)", value=1000, min_value=1, step=1
            )

            if avg_curve:
                averaged_data = calculate_log_average(
                    combined_well_data, avg_curve, interval=avg_interval
                )
                fig_avg = go.Figure()
                fig_avg.add_trace(
                    go.Scatter(
                        x=averaged_data[avg_curve],
                        y=averaged_data["Interval"],
                        mode="lines+markers",
                        name=f"Average of {avg_curve}",
                    )
                )
                fig_avg.update_layout(
                    height=600,
                    yaxis={"title": "DEPTH", "autorange": "reversed"},
                    xaxis={"title": avg_curve},
                    title=f"Average Log Plot for {avg_curve}",
                )
                st.plotly_chart(fig_avg, use_container_width=True)


if __name__ == "__main__":
    main()
