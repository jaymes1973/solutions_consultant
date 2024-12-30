#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:05:00 2024

@author: jaymesmonte
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt
from unidecode import unidecode

st.set_page_config(layout="wide")

# Load the data
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

data = load_data("https://raw.githubusercontent.com/jaymes1973/solutions_consultant/refs/heads/main/final_data.csv")

# Sidebar filters
st.sidebar.image("https://github.com/jaymes1973/solutions_consultant/blob/main/media/image.png?raw=true")
st.sidebar.header("Filters")
teams_list=data["team_name"].unique().tolist()
teams_list.remove("0")
selected_team_name = st.sidebar.selectbox("Select Team", teams_list)
filtered_team_data = data[data["team_name"] == selected_team_name]

if not filtered_team_data.empty:
    selected_player_name = st.sidebar.selectbox("Select Player", filtered_team_data["player_name"].unique())
    filtered_player_data = filtered_team_data[filtered_team_data["player_name"] == selected_player_name]
    
    minimum_mins_played = st.sidebar.slider("Minimum Minutes Played", min_value=0, max_value=90, value=0, step=1)
    filtered_player_data = filtered_player_data[filtered_player_data["player_match_minutes"] >= minimum_mins_played]
else:
    st.error("No players available for the selected team.")
    st.stop()

if not filtered_player_data.empty:
    selected_fixture = st.sidebar.selectbox("Select Fixture", filtered_player_data["fixture"].unique())
else:
    st.error("No data available for the selected player.")
    st.stop()

# Player Position filter
player_positions = ['Forward', 'Midfielder', 'Defender',]  # Example player positions
selected_player_position = st.sidebar.selectbox("Select Positional Filter", player_positions)

# Define a set of metrics for each player position
metrics_dict = {
    'Forward': ["player_match_np_xg_per_shot",
           "player_match_np_xg",
           "player_match_np_shots",
           "xG Assisted",
           "player_match_fhalf_lbp_received",
           "player_match_aerial_ratio",
           "player_match_touches_inside_box",
           "player_match_pressures",
           'total_distance',
           'sprint_distance'],
    'Midfielder': ["xG Assisted",
           "player_match_deep_progressions",
           "player_match_passes",
           "player_match_obv_pass",
           "player_match_obv_dribble_carry",
           "obv_per_pass",
           "player_match_interceptions",
           "player_match_f3_lbp",
           'total_distance',
           'sprint_distance'],
    'Defender': ["player_match_tackles",
           "player_match_interceptions",
           "player_match_aerials",
           "player_match_aerial_ratio",
           "player_match_defensive_actions",
           "player_match_passes",
           "player_match_passing_ratio",
           "ratio_forward_pass",
           'total_distance',
           'sprint_distance'],
}

# Use the selected player position to filter the metrics
metrics = metrics_dict.get(selected_player_position, [])

# Generate a user-friendly list of metric options for the dropdown
metrics_options = [
    metric.replace('player_match_', '').replace('np_', '').replace('_', ' ').replace('ratio', '%').capitalize() for metric in metrics_dict.get(selected_player_position, [])
]

# Sidebar dropdown for bar chart metric
selected_metric_display = st.sidebar.selectbox(
    "Select Metric for Bar Chart", metrics_options, index=0
)

# Map the selected display name back to the actual column name
metrics_mapping = {
    metric.replace('player_match_', '').replace('np_', '').replace('_', ' ').replace('ratio', '%').capitalize(): metric for metric in metrics_dict.get(selected_player_position, [])
}
selected_metric = metrics_mapping[selected_metric_display]

# Sidebar color pickers for team color and color1
team_color = st.sidebar.color_picker("Select Team Color", value="#ff6300")
color1 = st.sidebar.color_picker("Select Secondary Color", value="#b1bcc4")

team_badge=unidecode(selected_team_name)
team_badge=team_badge.replace(' ', '_')

col1,col2 = st.columns([0.1,0.9])
with col1:
    st.image(f"https://github.com/jaymes1973/solutions_consultant/blob/main/media/jubilo_iwata.png?raw=true")
with col2:
    st.title(f"{selected_player_name}")

st.subheader(f"{selected_fixture}")

# Check if data is available after filtering
if filtered_player_data.empty:
    st.error("No data available for the selected filters. Please adjust your filters.")
else:
    # Calculate z-scores for the selected metrics
    for metric in metrics:
    # Calculate z-score and then clip to be between -3 and 3
        z_scores = zscore(filtered_player_data[metric])
        filtered_player_data[f"{metric}_zscore"] = np.clip(z_scores, -3, 3)

    # Split data for season average and fixture-specific highlights
    non_fixture_df = filtered_player_data[filtered_player_data["fixture"] != selected_fixture]
    fixture_df = filtered_player_data[filtered_player_data["fixture"] == selected_fixture]
    
    formatted_titles = [metric.replace('player_match_', '').replace('np_', '').replace('_', ' ').replace('ratio', '%').capitalize() for metric in metrics]

    # Create subplots, one for each metric
    fig = make_subplots(
        rows=len(metrics), cols=1,
        shared_xaxes=False,  # Share x-axis across all subplots
        vertical_spacing=.015,  # Space between subplots
        #subplot_titles=formatted_titles  # Titles for each subplot
    )

    # Loop through each metric and plot on the respective subplot
    for idx, metric in enumerate(metrics):
        # Plot all other fixtures' z-scores
        for z_score_value, group in non_fixture_df.groupby(f"{metric}_zscore"):
            fixture_names = '<br>'.join(group['fixture'].unique())  # Combine fixture names
            fig.add_trace(go.Scatter(
                x=[z_score_value], 
                y=[0], 
                mode='markers',
                marker=dict(color=color1, size=20, opacity=0.5,
                            line=dict(color=team_color, width=2)),
                text=fixture_names,  # Show combined fixture names in the tooltip
                hoverinfo='text',  # Show fixture names on hover
                name=f"Other Fixtures - {metric}",
                showlegend=False  # Hide this trace from the legend
            ), row=idx + 1, col=1)
            
            fig.update_yaxes(showticklabels=False, row=idx + 1, col=1)
            fig.update_xaxes(range=[-4.25, 3.25], row=idx + 1, col=1)

        # Highlight the selected fixture's z-score
        z_score_value = fixture_df[f"{metric}_zscore"].iloc[0]
        metric_value = fixture_df[metric].iloc[0]
        # Format the value conditionally based on whether it has a fractional part
        if metric_value.is_integer():
            formatted_value = f"{int(metric_value)}"  # Display as integer
        else:
            formatted_value = f"{metric_value:.2f}"  # Display with two decimal places
        fig.add_trace(go.Scatter(
            x=[z_score_value], 
            y=[0], 
            mode='markers+text',
            marker=dict(
                color=team_color,
                size=45,
                opacity=1,
                symbol='circle',
                line=dict(color=color1, width=2)
            ),
            text=[formatted_value],  # Display the static text (e.g., metric value)
            textposition="middle center",  # Position the text in the middle of the marker
            hoverinfo="skip",  # Skip hover info since the text is static
            textfont=dict(
                family="Arial",  # Font family for the text
                size=10,  # Font size
                color="black"  # Text color (you can change this to any valid color)
            ),
            name=f"Selected Fixture ({selected_fixture}) - {metric}",
            showlegend=False  # Hide this trace from the legend
        ), row=idx + 1, col=1)

    # Set left-aligned subplot titles using annotations
    for idx, title in enumerate(formatted_titles):
        fig.add_annotation(
            x=-3.5,  # Position the title on the left side of the subplot
            y=0.1,  # Adjust the vertical positioning
            text=title,
            font=dict(size=14, color="black", family="Arial"),
            showarrow=False,
            xref="paper", yref="y",
            row=idx + 1, col=1,
            align="left",
            bgcolor=team_color,  # Background color for the text box
            bordercolor=color1,  # Border color for the text box
            borderwidth=2,  # Border width for the text box
            borderpad=3,  # Padding between text and border
        )
    fig.add_annotation(
        x=0,  # Place the text at x=0
        y=1.05,  # Adjust the vertical position as needed (above the plot)
        text="Season Average",
        font=dict(size=12, color="black", family="Arial"),
        showarrow=False,
        xref="x",  # Reference to the entire figure's x-axis
        yref="paper",  # Use "paper" for y-coordinate to place it above the figure
        align="center",  # Center the text horizontally
        bgcolor="white",  # Background color for the text box (optional)
        bordercolor="black",  # Border color (optional)
        borderwidth=1,  # Border width (optional)
        borderpad=3,  # Padding between text and border (optional)
    )
    fig.update_layout(
        yaxis=dict(showticklabels=False, fixedrange=True, showgrid=False),  # Hide y-axis labels and ticks
        dragmode='zoom',  # Allow zooming but not dragging along the x-axis
        xaxis=dict(fixedrange=True),
        showlegend=False,  # Remove the legend from the entire plot
        height=100 + (len(metrics) * 75),  # Adjust height based on number of metrics
    )

    # Loop through each subplot's x-axis to apply vertical gridlines
    for idx in range(1, len(metrics) + 1):
        fig.update_xaxes(
            tickvals=[-.67, 0, .67],  # Set vertical gridlines at specified x-values
            showticklabels=False,  # Remove x-tick labels
            row=idx, col=1,
            showgrid=True,  # Enable vertical gridlines
            gridcolor='lightgray',  # Gridline color
            gridwidth=0.15,  # Gridline width 
        )
        fig.update_yaxes(showgrid=False)  # Gridline width
        fig.add_shape(
            type="line",
            x0=-3, x1=3,  # Match the x-axis limits
            y0=-.25, y1=-.25,  # Line at y=0
            line=dict(color="black", width=2),  # Line style
            xref=f"x{idx}",  # Reference to the current x-axis
            yref=f"y{idx}"  # Reference to the current y-axis
        )
    fig_bar, ax = plt.subplots(figsize=(18, 8))  # Adjust the height as needed
    fig_bar.set_facecolor("white")
    
    # Extract bar chart data
    filtered_player_data=filtered_player_data.sort_values(by=['match_date'],ascending=True)
    bars = filtered_player_data["fixture"]
    x_pos = np.arange(len(filtered_player_data))
    
    ax.bar(x_pos, filtered_player_data[selected_metric], color=team_color, ec=color1,lw=2, alpha=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bars, rotation=45, ha="right")  
    ax.set_ylabel(selected_metric.replace('player_match_', '').replace('np_', '').replace('_', ' ').replace('ratio', '%').capitalize(),
                  fontsize=14, color="black")
    #ax.set_title(f"{selected_metric.replace('player_match_', '').replace('np_', '').replace('_', ' ').replace('ratio', '%').capitalize()} for {selected_player_name}", 
     #            fontsize=18, color="black")
    # Remove unnecessary spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    # Add gridlines for better readability
    ax.grid(axis="y", linestyle="--", color="gray", alpha=0.5)
    # Display the interactive plot in Streamlit
    st.plotly_chart(fig)
    st.subheader(f"Game-by-game for {selected_metric.replace('player_match_', '').replace('np_', '').replace('_', ' ').replace('ratio', '%').capitalize()}")
    st.pyplot(fig_bar)
