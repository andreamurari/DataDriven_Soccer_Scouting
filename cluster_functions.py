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
    
    fig.update_layout(title_x=0, xaxis_tickangle=-45)
    
    # MODIFICA STREAMLIT:
    st.plotly_chart(fig, use_container_width=True)


def plot_anomalies_per_league(df_anomalies, df_merged, macro_pos=None):
    
    league_colors = {
        'ESP-La Liga': "#FF4B44",
        'ENG-Premier League': '#04F5FF',
        'GER-Bundesliga': "#777777",
        'FRA-Ligue 1': '#CDFB0A',
        'ITA-Serie A': '#0578FF'
    }
    
    if macro_pos is not None:
        df_anomalies = df_anomalies[df_anomalies['macro_pos'] == macro_pos]
        df_merged = df_merged[df_merged['macro_pos'] == macro_pos]
        
    total_by_league = df_merged.groupby('league').size().reset_index(name='total_players')
    
    count_by_league = df_anomalies.groupby('league').agg(
        count=('league', 'count'),
        avg_anomaly_score=('robust_anomaly_score', 'mean')
    ).reset_index()
    
    stats = pd.merge(count_by_league, total_by_league, on='league', how='left')
    stats['percentage'] = (stats['count'] / stats['total_players']) * 100
    
    stats = stats.sort_values('percentage', ascending=False)
    
    stats['bar_text'] = stats.apply(lambda x: f"{x['percentage']:.1f}% ({int(x['count'])})", axis=1)
    
    fig = px.bar(
        stats,
        x='league',
        y='percentage', 
        color='league',
        title='Anomalies by League',
        labels={
            'count': 'Number of Anomalies', 
            'league': 'League',
            'percentage': '% of Anomalies in League',
            'avg_anomaly_score': 'Avg Anomaly Score',
            'total_players': 'Total Players'
        },
        text='bar_text',
        color_discrete_map=league_colors,
        hover_data={
            'avg_anomaly_score': ':.2f',
            'count': True,
            'total_players': True,
            'percentage': ':.2f',
            'league': False, 
            'bar_text': False
        }
    )
    
    fig.update_traces(textposition='auto', marker_line_width=0.2, marker_line_color='black')
    fig.update_layout(title_x=0, xaxis_tickangle=-45)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_anomalies_per_age(df_anomalies, df_merged, macro_pos=None):
    
    if macro_pos is not None:
        df_anomalies = df_anomalies[df_anomalies['macro_pos'] == macro_pos]
        df_merged = df_merged[df_merged['macro_pos'] == macro_pos]
        
    total_by_age = df_merged.groupby('born').size().reset_index(name='total_players')
    
    count_by_age = df_anomalies.groupby('born').agg(
        count=('born', 'count'),
        avg_anomaly_score=('robust_anomaly_score', 'mean')
    ).reset_index()
    
    stats = pd.merge(count_by_age, total_by_age, on='born', how='left')
    stats['percentage'] = (stats['count'] / stats['total_players']) * 100
    
    stats = stats.sort_values('born')
    
    stats['bar_text'] = stats.apply(lambda x: f"{x['percentage']:.1f}% ({int(x['count'])})", axis=1)
    
    fig = px.bar(
        stats,
        x='born',
        y='percentage', 
        color='avg_anomaly_score',
        title='Anomalies by Age',
        labels={
            'count': 'Number of Anomalies', 
            'born': 'Birth Year',
            'percentage': '% of Players in Age Cohort',
            'total_players': 'Total Players',
            'avg_anomaly_score': 'Avg Anomaly Score'
        },
        text='bar_text',
        color_continuous_scale='Greens',
        hover_data={
            'avg_anomaly_score': ':.2f',
            'count': True,
            'total_players': True,
            'percentage': ':.2f',
            'bar_text': False
        }
    )
    
    fig.update_traces(textposition='auto', marker_line_width=0.2, marker_line_color='black')
    fig.update_layout(xaxis_tickangle=-45)
    
    fig.update_layout(
        title_x=0, 
        xaxis=dict(type='category') 
    )
    
    st.plotly_chart(fig, use_container_width=True)

import streamlit as st

def display_single_anomaly_pct(df_anomalies, df_merged, macro_pos=None):
    if macro_pos is None:
        total_anomalies = len(df_anomalies)
        total_players = len(df_merged)
        label_text = "Percentage of anomalies found:"
    else:
        anomalies_filtered = df_anomalies[df_anomalies['macro_pos'] == macro_pos]
        merged_filtered = df_merged[df_merged['macro_pos'] == macro_pos]
        
        total_anomalies = len(anomalies_filtered)
        total_players = len(merged_filtered)
        label_text = f"Anomalies found for {macro_pos}:"
    
    if total_players == 0:
        st.warning(f"No data found for position: {macro_pos}")
        return
        
    pct = (total_anomalies / total_players) * 100
    
    st.metric(
        label=label_text, 
        value=f"{pct:.1f}%",
        delta=f"{total_anomalies} / {total_players} players",
        delta_color="off"
    )
def display_anomaly_scouting_report(df_anomalies, df_glossary, macro_pos=None):
    st.markdown("### 📋 Scouting Roster & Player Deep Dive")
    
    # 1. Filter the dataset based on selection
    if macro_pos is not None and macro_pos != "All Positions":
        df_filtered = df_anomalies[df_anomalies['macro_pos'] == macro_pos].copy()
    else:
        df_filtered = df_anomalies.copy()
        
    if df_filtered.empty:
        st.info(f"No anomalous players found for position: {macro_pos}")
        return
        
    # Sort by the most extreme anomalies first
    df_filtered = df_filtered.sort_values('robust_anomaly_score', ascending=False)
    
    # ==========================================
    # PHASE 1: THE "MASTER" TABLE (Ultra-clean)
    # ==========================================
    st.markdown("Select a profile from the table below to open their analytical card.")
    
    df_display = pd.DataFrame({
        'Player': df_filtered['player'],
        'Team': df_filtered['team'],
        'League': df_filtered['league'],
        'Age': df_filtered['age'],
        'Score': df_filtered['robust_anomaly_score'],
        'Main Positive Deviation': df_filtered['Top_Pos_Feat_1'], 
        'Main Negative Deviation': df_filtered['Top_Neg_Feat_1']   
    })
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn("Anomaly Score", format="%.1f"),
            "Age": st.column_config.NumberColumn("Age", format="%d")
        }
    )
    
    st.markdown("---")
    
    # ==========================================
    # PHASE 2: THE SCOUTING CARD (Detail)
    # ==========================================
    st.markdown("### 🔍 Tactical Profile Explorer")
    
    selected_player = st.selectbox(
        "Select a player for the Deep Dive:",
        options=df_filtered['player'].tolist()
    )
    
    if selected_player:
        # Extract the specific player's data
        player_data = df_filtered[df_filtered['player'] == selected_player].iloc[0]
        
        st.subheader(f"👤 {player_data['player']} ({player_data['age']} yrs) - {player_data['team']}")
        
        # Build the Glossary Dictionary FIRST so we can use it inside the loop
        try:
            glossary_dict = dict(zip(df_glossary['KPI'], df_glossary['Explanation']))
        except KeyError:
            st.error("Error: Could not find 'KPI' or 'Explanation' columns in the glossary file.")
            glossary_dict = {}

        # Create exactly 2 columns for a balanced layout
        col_pos, col_neg = st.columns(2)
        
        with col_pos:
            st.success("📈 Over-Performing Traits")
            for i in range(1, 4):
                feat = player_data[f'Top_Pos_Feat_{i}']
                pctl = player_data[f'Top_Pos_Pctl_{i}']
                delta = player_data[f'Top_Pos_Delta_{i}']
                
                # Retrieve the definition from the dictionary
                trait_definition = glossary_dict.get(feat, "Definition not available.")
                
                st.metric(
                    label=feat,
                    value=f"{pctl}th Pctl",
                    delta=f"+{delta} vs Avg",
                    delta_color="normal",
                    help=trait_definition # <--- UX MAGIC: This creates the hover tooltip!
                )
                
        with col_neg:
            st.error("📉 Under-Performing Traits")
            for i in range(1, 4):
                feat = player_data[f'Top_Neg_Feat_{i}']
                pctl = player_data[f'Top_Neg_Pctl_{i}']
                delta = player_data[f'Top_Neg_Delta_{i}']
                
                # Retrieve the definition from the dictionary
                trait_definition = glossary_dict.get(feat, "Definition not available.")
                
                st.metric(
                    label=feat,
                    value=f"{pctl}th Pctl",
                    delta=f"{delta} vs Avg", 
                    delta_color="normal",
                    help=trait_definition # <--- UX MAGIC: This creates the hover tooltip!
                )