import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st



# ============================================
# ============================================
# K-means Clustering Analysis Functions
# ============================================
# ============================================

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



# ============================================
# ============================================
# AutoEncoder Clustering Analysis Functions
# ============================================
# ============================================


def plot_anomalies_per_macropos(df_anomalies, df_merged, sort_by='Percentage'):
    
    macro_pos_order = ['CB', 'Fullback', 'CDM', 'CM', 'Wide Midfielder', 'CAM', 'Winger', 'ST']
    
    total_by_macropos = df_merged.groupby('macro_pos').size().reset_index(name='total_players')
    
    count_by_macropos = df_anomalies.groupby('macro_pos').agg(
        count=('macro_pos', 'count'),
        avg_anomaly_score=('robust_anomaly_score', 'mean')
    ).reset_index()
    
    stats = pd.merge(count_by_macropos, total_by_macropos, on='macro_pos', how='left')
    stats['percentage'] = (stats['count'] / stats['total_players']) * 100
    stats['bar_text'] = stats.apply(lambda x: f"{x['percentage']:.1f}% ({int(x['count'])})", axis=1)
    
    if sort_by == 'Percentage':
        stats = stats.sort_values('percentage', ascending=False)
    elif sort_by == 'Count':
        stats = stats.sort_values('count', ascending=False)
        stats['bar_text'] = stats.apply(lambda x: f"{int(x['count'])} ({x['percentage']:.1f}%)", axis=1)

    fig = px.bar(
        stats,
        x='macro_pos',
        y='percentage',
        color='avg_anomaly_score',
        title='Anomalies by Macro-Position',
        labels={
            'count': 'Number of Anomalies', 
            'macro_pos': 'Macro-Position',
            'avg_anomaly_score': 'Avg Anomaly Score',
            'percentage': '% of Total Position',
            'total_players': 'Total Players in Position'
        },
        text='bar_text', 
        color_continuous_scale='Blues',
        hover_data={
            'avg_anomaly_score': ':.2f', 
            'percentage': ':.2f', 
            'total_players': True,
            'bar_text': False
        }
    )
    
    fig.update_traces(textposition='auto', marker_line_width=0.2, marker_line_color='black')
    
    if sort_by == 'Macro-Position':
        fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray': macro_pos_order})
    
    fig.update_layout(title_x=0.5, xaxis_tickangle=-45)
    
    # MODIFICA STREAMLIT:
    st.plotly_chart(fig, use_container_width=True)


def anomalies_per_league(df, macro_pos=None):
    
    league_colors = {
        'ESP-La Liga': "#FF4B44",
        'ENG-Premier League': '#04F5FF',
        'GER-Bundesliga': "#777777",
        'FRA-Ligue 1': '#CDFB0A',
        'ITA-Serie A': '#0578FF'
    }
    
    if macro_pos is not None:
        df = df[df['macro_pos'] == macro_pos]
    
    count_by_league = df.groupby('league').agg(
        count=('league', 'count'),
        avg_anomaly_score=('robust_anomaly_score', 'mean')
    ).reset_index().sort_values('count', ascending=False)
    
    fig = px.bar(
        count_by_league,
        x='league',
        y='count',
        color='league',
        title='Anomalies by League',
        labels={'count': 'Number of Anomalies', 'league': 'League'},
        text='count',
        color_discrete_map=league_colors,
        hover_data={'avg_anomaly_score': ':.2f'}
    )
    
    fig.update_traces(textposition='auto', marker_line_width=0.2, marker_line_color='black')
    fig.update_layout(title_x=0.5, xaxis_tickangle=-45)
    
    # MODIFICA STREAMLIT:
    st.plotly_chart(fig, use_container_width=True)


def anomalies_per_age(df, macro_pos=None):
    
    if macro_pos is not None:
        df = df[df['macro_pos'] == macro_pos]
    
    count_by_age = df.groupby('born').agg(
        count=('born', 'count'),
        avg_anomaly_score=('robust_anomaly_score', 'mean')
    ).reset_index().sort_values('born')
    
    fig = px.bar(
        count_by_age,
        x='born',
        y='count',
        color='avg_anomaly_score',
        title='Anomalies by Age',
        labels={'count': 'Number of Anomalies', 'born': 'Birth Year'},
        text='count',
        color_continuous_scale='Greens',
        hover_data={'avg_anomaly_score': ':.2f'}
    )
    
    fig.update_traces(textposition='auto', marker_line_width=0.2, marker_line_color='black')
    fig.update_layout(title_x=0.5)
    
    # MODIFICA STREAMLIT:
    st.plotly_chart(fig, use_container_width=True)


def display_single_anomaly_pct(df_anomalies, df_merged, macro_pos):
    """
    Mostra la percentuale di anomalie per una singola macro-posizione.
    """
    anomalies_filtered = df_anomalies[df_anomalies['macro_pos'] == macro_pos]
    merged_filtered = df_merged[df_merged['macro_pos'] == macro_pos]
    
    total_anomalies = len(anomalies_filtered)
    total_players = len(merged_filtered)
    
    if total_players == 0:
        st.warning(f"Nessun dato trovato per la posizione: {macro_pos}")
        return
        
    pct = (total_anomalies / total_players) * 100
    
    st.metric(
        label=f"Anomalie: {macro_pos}", 
        value=f"{pct:.1f}%",
        delta=f"{total_anomalies} giocatori", # OPZIONALE: Sfruttiamo il delta per mostrare il numero assoluto
        delta_color="off" # Spegniamo il colore verde/rosso del delta perché qui è solo informativo
    )