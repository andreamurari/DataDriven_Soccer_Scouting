import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


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
        title=f'Position Distribution in Cluster {cluster_id}',
        labels={'Count': 'Number of Players', 'Position': 'Position'},
        color='Position',
        color_discrete_map=color_map,
        text='Count'
    )
    fig.update_traces(textposition='auto')
    fig.update_layout(title_x=0.5)
    fig.show()
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
        title=f'League Distribution in Cluster {cluster_id} (Normalized by League Size)',
        labels={'Percentage': '% of League Players in Cluster', 'League': 'League'},
        text='Percentage',
        color='League',
        color_discrete_map=league_colors,
        hover_data={'Count_in_Cluster': True, 'Total_Count': True, 'Percentage': ':.1f'}
    )
    fig.update_traces(textposition='auto', marker_line_width=2, marker_line_color='white')
    fig.update_layout(title_x=0.5, xaxis_tickangle=-45, showlegend=False)
    fig.show()
    return fig
