import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st


def plot_cluster_positions(cluster_id, df_clusters):
    """
    Create a bar plot showing the distribution of positions in a specific cluster
    """
    cluster_data = df_clusters[df_clusters['cluster'] == cluster_id]
    pos_counts = cluster_data['pos'].value_counts().reset_index()
    pos_counts.columns = ['Position', 'Count']
    
    # Create color map for positions
    unique_positions = sorted(df_clusters['pos'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_positions)))
    color_map = {
        pos: f'#{int(colors[i][0]*255):02x}{int(colors[i][1]*255):02x}{int(colors[i][2]*255):02x}'
        for i, pos in enumerate(unique_positions)
    }
    
    fig = px.bar(
        pos_counts,
        x='Position',
        y='Count',
        labels={'Count': 'Number of Players', 'Position': 'Position'},
        color='Position',
        color_discrete_map=color_map,
        text='Count'
    )
    fig.update_traces(textposition='auto', marker_line_width=1, marker_line_color='white')
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    return fig


def plot_cluster_league(cluster_id, df_clusters):
    """
    Create a bar plot showing the normalized distribution of leagues in a specific cluster
    (normalized by the total number of players in each league)
    """
    cluster_data = df_clusters[df_clusters['cluster'] == cluster_id]
    
    # Count players by league in this cluster
    league_counts_cluster = cluster_data['league'].value_counts().reset_index()
    league_counts_cluster.columns = ['League', 'Count_in_Cluster']
    
    # Count total players by league in the entire dataset
    league_counts_total = df_clusters['league'].value_counts().reset_index()
    league_counts_total.columns = ['League', 'Total_Count']
    
    # Merge and calculate normalized percentage
    league_data = league_counts_cluster.merge(league_counts_total, on='League')
    league_data['Percentage'] = (league_data['Count_in_Cluster'] / league_data['Total_Count'] * 100).round(1)
    
    # Sort by percentage descending
    league_data = league_data.sort_values('Percentage', ascending=False)
    
    # Define league colors with hex codes
    league_colors = {
        'ESP-La Liga': '#FF4B44',
        'ENG-Premier League': '#04F5FF',
        'GER-Bundesliga': '#777777',
        'FRA-Ligue 1': '#CDFB0A',
        'ITA-Serie A': '#0578FF'
    }
    
    fig = px.bar(
        league_data,
        x='League',
        y='Percentage',
        labels={'Percentage': '% of League Players in Cluster', 'League': 'League'},
        text='Percentage',
        color='League',
        color_discrete_map=league_colors,
        hover_data={'Count_in_Cluster': True, 'Total_Count': True, 'Percentage': ':.1f'}
    )
    fig.update_traces(textposition='auto', marker_line_width=1, marker_line_color='white')
    fig.update_layout(
        xaxis_tickangle=-45, 
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig


def analyze_cluster(cluster_id, df_clusters, df_cluster_profile, glossary_dict, show_metrics=True):
    """
    Display cluster analysis with dominant position, features, and anomalies
    """
    # Extract dominant position
    dominant_role = df_cluster_profile.loc[cluster_id, "dominant_role"]
    dominant_pos = dominant_role.split(" ")[0]
    
    # Get cluster data
    cluster_data = df_clusters[df_clusters['cluster'] == cluster_id]
    
    # Display analysis metrics (optional)
    if show_metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Cluster Size", len(cluster_data), "players")
        
        with col2:
            st.metric("Dominant Position", dominant_pos)
        
        with col3:
            st.metric("Total Positions", cluster_data['pos'].nunique())
    
    # Display profile features
    st.subheader("📊 Cluster Profile Features")
    
    feature_cols = st.columns(2)
    
    with feature_cols[0]:
        st.write("**Top Positive Features:**")
        for col in ['top_pos_1', 'top_pos_2', 'top_pos_3']:
            feature = df_cluster_profile.loc[cluster_id, col]
            if feature != "-":
                feature_name = feature.split(" (")[0]
                if feature_name in glossary_dict:
                    with st.expander(f"ℹ️ {feature}"):
                        st.write(glossary_dict[feature_name])
    
    with feature_cols[1]:
        st.write("**Top Negative Features:**")
        for col in ['top_neg_1', 'top_neg_2', 'top_neg_3']:
            feature = df_cluster_profile.loc[cluster_id, col]
            if feature != "-":
                feature_name = feature.split(" (")[0]
                if feature_name in glossary_dict:
                    with st.expander(f"ℹ️ {feature}"):
                        st.write(glossary_dict[feature_name])
    st.markdown("---")
    # Display scouting report
    col1, col2 = st.columns(2)
    
    with col2:
        st.subheader("🎯 Scouting Report")
        st.info(df_cluster_profile.loc[cluster_id, "scouting_report"])
    
    with col1:
        st.subheader("📈 League Distribution")
        st.plotly_chart(
            plot_cluster_league(cluster_id, df_clusters),
            width='stretch'
    )
    
    st.markdown("---")
    
    # Display position distribution in cluster - side by side with plot
    st.subheader("📍 Position Distribution")
    pos_cols = st.columns(2)
    with pos_cols[0]:
        # Plot
        fig = plot_cluster_positions(cluster_id, df_clusters)
        st.plotly_chart(fig, use_container_width=True)
    
    with pos_cols[1]:
        # Table
        pos_counts = cluster_data['pos'].value_counts().reset_index()
        pos_counts.columns = ['Position', 'Count']
        pos_counts['Percentage'] = (pos_counts['Count'] / len(cluster_data) * 100).round(1)
        st.dataframe(pos_counts, use_container_width=True, hide_index=True)
