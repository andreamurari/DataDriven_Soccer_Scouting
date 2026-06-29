import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from cluster_functions import plot_cluster_positions, plot_cluster_league, analyze_cluster
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Football Scouting - Anomaly Detection",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_resource
def load_data():
    """Load the necessary data from CSV files and prepare structures"""
    
    # Load cluster analysis data
    df_clusters = pd.read_csv("resources/df_clusters.csv")
    df_clusters = df_clusters.dropna()
    df_clusters = df_clusters[df_clusters['pos'] != 'GK']
    
    # Load cluster profile
    df_cluster_profile = pd.read_csv("resources/df_cluster_profile.csv")
    
    # Load glossary for feature explanations
    try:
        import openpyxl
        glossary = pd.read_excel("resources/glossary.xlsx")
        glossary_dict = dict(zip(glossary['KPI'], glossary['Explanation']))
    except:
        glossary_dict = {}
    
    return df_clusters, df_cluster_profile, glossary_dict

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("⚽ Football Scouting - Anomaly Detection")
    
    # Load data
    df_clusters, df_cluster_profile, glossary_dict = load_data()
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select Section", ["Overview", "Cluster Analysis"])
    
    # ========================================================================
    # PAGE 1: OVERVIEW
    # ========================================================================
    if page == "Overview":
        
        st.header("👽 The Anomaly Hunter")
        
        st.markdown("""
        **Discovering Tactical Aliens**\n
        Modern football scouting relies heavily on rigid positional labels. Our objective is to challenge these traditional definitions by using unsupervised machine learning to uncover players who break the mold and operate in a completely non-traditional way compared to their peers.
        \n
        By applying $K$-means clustering strictly to performance data, we group players based on how they play, rather than where they are deployed. We then analyze the position distribution within each cluster to isolate anomalies: players who end up in clusters dominated by an entirely different role (e.g., a striker grouped with central defenders).
        
        **The Technical Goal**: Identify statistical outliers by detecting minority positions within highly homogeneous clusters, mapping out players with unique, unconventional tactical behaviors.
        """)
        
        st.header("K-Means Clustering Overview")
        
        with st.expander("🧩 What is K-Means Clustering?", expanded=False):
            st.markdown("""
            **K-Means Clustering** is an unsupervised machine learning algorithm that groups players 
            into clusters based on their statistical profiles (e.g., passing, shooting, dribbling, defense, etc.).
            
            ### How It Works:
            1. **Initialization**: The algorithm starts with k randomly selected cluster centers (centroids).
            2. **Assignment**: Each player is assigned to the nearest cluster. 
            3. **Update**: Cluster centers are recalculated based on the mean of assigned players.
            4. **Iteration**: Repeat until convergence.
            
            ### Why K-Means for Soccer Scouting?
            - **Tactical Profiling**: Players in the same cluster share similar playing styles, regardless of how much possession their team averages.
            - **Anomaly Detection**: Identify players who don't fit their assigned position (e.g., a CB playing like a deep-lying playmaker).
            - **Scouting Shortcuts**: Find similar players to targets across different leagues/teams.
            - **Talent Benchmarking**: Compare player profiles to established tactical templates.
            """)
        
        with st.expander("🛠️ How We Built This Model", expanded=False):
            st.markdown("""
            ### Data Preparation:
            - **Removed Noisy Features**: Excluded playing time (`90s`, `Starts`) and penalty metrics (`PK`) to prevent the model from grouping players by "status" (starter vs. bench) instead of tactics.
            - **Standardization**: Scaled all features to mean=0, std=1 using `StandardScaler`.
            - **L2 Normalization**: Projected all player vectors to a length of 1 using `normalize(norm='l2')`. This eliminates the "Possession Bias", ensuring a player with 100 passes and 10 tackles is grouped with a player with 50 passes and 5 tackles (same 10:1 tactical ratio).
            
            ### Model Parameters:
            - **Number of Clusters (k)**: 20 clusters (Optimized to isolate specific tactical micro-roles).
            - **Initialization**: 50 random seeds (`n_init=50`) to guarantee absolute mathematical stability.
            - **Metric**: Euclidean distance on L2-Normalized data (mathematically equivalent to **Cosine Similarity**).
            
            ### Cluster Profiles:
            Each cluster is characterized by its:
            - **Top 3 Positive Features (📈)**: The highest Z-scores showing what the cluster does **more** compared to the European average.
            - **Top 3 Negative Features (📉)**: The lowest Z-scores showing what the cluster does **less** compared to the European average.
            - **Dominant Role**: Most common nominal position in the cluster.
            - **Scouting Report**: A bespoke, human-readable interpretation of the cluster's pure tactical playing style.
            """)
        
        with st.expander("📊 Cluster Distribution & Position Analysis", expanded=False):
            col1, col2 = st.columns([0.3, 0.7])

            with col1:
                st.subheader("Cluster Overview")
                st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
                cluster_dist = df_clusters.groupby('cluster').size().reset_index(name='Player Count')
                cluster_dist = cluster_dist.merge(
                    df_cluster_profile[['cluster', 'dominant_role']],
                    on='cluster'
                )
                st.table(cluster_dist)


            with col2:
                st.subheader("Position Distribution Across Clusters")
                # Create subplots (4 rows x 5 columns for 20 clusters)
                fig = make_subplots(
                    rows=4, cols=5,
                    subplot_titles=[f"Cluster {i}" for i in sorted(df_clusters['cluster'].unique())],
                    specs=[[{"secondary_y": False}] * 5 for _ in range(4)],
                    vertical_spacing=0.12,
                    horizontal_spacing=0.08
                )

                # Color map for positions
                unique_positions = sorted(df_clusters['pos'].unique())
                colors = px.colors.qualitative.Set3
                pos_colors = {pos: colors[i % len(colors)] for i, pos in enumerate(unique_positions)}

                # Create a bar chart for each cluster
                for cluster_idx, cluster_id in enumerate(sorted(df_clusters['cluster'].unique())):
                    row = (cluster_idx // 5) + 1
                    col = (cluster_idx % 5) + 1

                    cluster_data = df_clusters[df_clusters['cluster'] == cluster_id]
                    pos_dist = cluster_data['pos'].value_counts()

                    for pos in pos_dist.index:
                        fig.add_trace(
                            go.Bar(
                                x=[pos],
                                y=[pos_dist[pos]],
                                name=pos,
                                marker_color=pos_colors[pos],
                                showlegend=(cluster_idx == 0),
                                hovertemplate=f"{pos}: %{{y}}<extra></extra>"
                            ),
                            row=row, col=col
                        )

                fig.update_layout(
                    height=710,
                    showlegend=False,
                    barmode='stack',
                    margin=dict(l=0, r=0, t=50, b=0)
                )

                fig.update_traces(textposition='auto', marker_line_width=0.2, marker_line_color='white')

                st.plotly_chart(fig, width='stretch')
                st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)        
        
        with st.expander("🔍 Cluster Profiles & Scouting Reports", expanded=False):
        
            clusters = sorted(df_cluster_profile['cluster'].unique())
            cols_per_row = 4

            for i in range(0, len(clusters), cols_per_row):
                cols = st.columns(cols_per_row)
                cluster_batch = clusters[i:i+cols_per_row]

                for col, cluster_id in zip(cols, cluster_batch):
                    with col:
                        with st.container(border=True):
                            st.markdown(f"**📋 Cluster {cluster_id}**")
                            st.markdown(f"*{df_cluster_profile.loc[cluster_id, 'dominant_role']}*")
                            st.divider()
                            st.write(df_cluster_profile.loc[cluster_id, 'scouting_report'])
        
        with st.expander("🔗 Tactical Similarity Matrix (Positions)", expanded=False):
            st.markdown("""
            This heatmap shows how **tactically similar** different positions are based on their **cluster distribution patterns**.
            
            - **Red**: Positions with negative correlation (players in one position rarely share cluster patterns with players in another)
            - **Blue**: Positions with strong positive correlation (players in these positions often cluster together, suggesting similar playing styles)
            - **Diagonal = 1.0**: Perfect correlation (a position with itself)
            """)
            
            # Calculate position correlation based on cluster distribution
            pivot_pos = pd.crosstab(df_clusters['cluster'], df_clusters['pos'])
            pivot_norm = pivot_pos.div(pivot_pos.sum(axis=1), axis=0)
            pos_corr = pivot_norm.corr()
            
            # Sort the correlation matrix by the specified position order
            position_order = ['CB', 'RB', 'LB', 'CDM', 'CM', 'RM', 'LM', 'CAM', 'RW', 'LW', 'ST']
            # Filter to keep only positions that exist in the data
            position_order = [pos for pos in position_order if pos in pos_corr.columns]
            pos_corr = pos_corr.loc[position_order, position_order]
            
            fig = px.imshow(
                pos_corr,
                labels=dict(x="Position", y="Position", color="Correlation"),
                x=pos_corr.columns,
                y=pos_corr.columns,
                color_continuous_scale='RdBu_r',
                zmin=-0.5,
                zmax=1,
                text_auto='.2f',
                aspect='auto',
            )
            
            fig.update_layout(
                width=900,
                height=800,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Key Insights**:
            - ***The "Wingback" Symmetry*** (RB ↔ LB):
                - The strongest correlation in the entire dataset (excluding self-correlation) is between RB and LB (0.97).
                - Insight: This suggests that in modern football, the requirements for fullbacks are almost identical regardless of the flank. They are likely being clustered based on their involvement in progression and defensive volume rather than side-specific traits.

            - ***The Creative/Wide Block*** (RM, LM, RW, LW, CAM):
                - There is a massive "warm" zone in the bottom-right quadrant.
                - Insight: There is a high degree of interchangeability between wide midfielders (RM/LM) and wingers (RW/LW), with correlations ranging from 0.75 to 0.92.
                - Observation: Interestingly, LM/LW (0.86) and RM/RW (0.76) are very strong, but LM and RW (0.88) are even stronger. This suggests your clustering is picking up on the "Inverted Winger" or "Wide Playmaker" profile that dominates the current era, where the specific side matters less than the functional role.

            - ***The Central Engine Room*** (CDM ↔ CM)
                - There is a solid correlation (0.71) between Defensive Midfielders and Central Midfielders.
                - Insight: While distinct, these roles often bleed into each other. However, notice the sharp drop-off when moving from CM to CAM (-0.04).
                
            - ***The "Lone Wolf" Striker*** (ST)
                - The ST (Striker) position shows almost zero or negative correlation with every other position on the pitch.
                - Analysis: Your clustering suggests that the statistical output of a Striker (likely high shots, low touches in the buildup, high xG) is so unique that they almost never share a cluster with even the most attacking wingers or CAMs. In a modern "False 9" era, you might expect more overlap, but your data suggests a very traditional separation of the #9 role.

            - ***The Defensive Island*** (CB)
                - The Center Back (CB) is the most isolated role defensively.
                - Analysis: It has negative correlations with almost every other position, especially the midfield engine (CM at -0.44). This confirms that the statistical "DNA" of a CB—heavy on aerials, clearances, and low-risk passing—is fundamentally different from the "active" defending seen in CDMs.""")
            
    # ========================================================================
    # PAGE 2: CLUSTER ANALYSIS
    # ========================================================================
    elif page == "Cluster Analysis":
        st.header("🔍 Interactive Cluster Analysis")
        
        # Cluster selector and metrics in one row
        selector_col, metric_col1, metric_col2, metric_col3 = st.columns([1.5, 1, 1, 1])
        
        with selector_col:
            cluster_id = st.selectbox(
                "Select a Cluster to Analyze",
                options=sorted(df_clusters['cluster'].unique()),
                format_func=lambda x: f"Cluster {x}: {df_cluster_profile.loc[x, 'dominant_role']}"
            )
        
        # Get cluster data for metrics
        cluster_data = df_clusters[df_clusters['cluster'] == cluster_id]
        dominant_role = df_cluster_profile.loc[cluster_id, "dominant_role"]
        dominant_pos = dominant_role.split(" ")[0]
        
        with metric_col1:
            st.metric("Cluster Size", len(cluster_data), "players")
        
        with metric_col2:
            st.metric("Dominant Position", dominant_pos)
        
        with metric_col3:
            st.metric("Total Positions", cluster_data['pos'].nunique())
        
        # Run analysis (without repeating metrics)
        analyze_cluster(cluster_id, df_clusters, df_cluster_profile, glossary_dict, show_metrics=False)
        
        st.markdown("---")
        
        # Player list for selected cluster
        st.subheader("👥 Players in This Cluster")
        
        # Get all players in cluster
        all_cluster_players = df_clusters[df_clusters['cluster'] == cluster_id]
        
        # Create filters
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            # Define position order
            position_order = ['CB', 'RB', 'LB', 'CDM', 'CM', 'RM', 'LM', 'CAM', 'RW', 'LW', 'ST']
            available_positions = all_cluster_players['pos'].unique()
            sorted_positions = [pos for pos in position_order if pos in available_positions]
            
            st.markdown("**Filter by Position**")
            selected_positions = st.multiselect(
                "Positions",
                options=sorted_positions,
                default=sorted_positions,
                label_visibility="collapsed"
            )
            selected_positions = selected_positions if selected_positions else sorted_positions
        
        with filter_col2:
            st.markdown("**Filter by League**")
            available_leagues = sorted(all_cluster_players['league'].unique())
            selected_leagues = st.multiselect(
                "Leagues",
                options=available_leagues,
                default=available_leagues,
                label_visibility="collapsed"
            )
            selected_leagues = selected_leagues if selected_leagues else available_leagues
        
        # Apply filters
        cluster_players = all_cluster_players[
            (all_cluster_players['pos'].isin(selected_positions)) &
            (all_cluster_players['league'].isin(selected_leagues))
        ][
            ['player', 'pos', 'team', 'league', 'season', 'age', 'nation']
        ].sort_values(['pos', 'player']).reset_index(drop=True)
        
        st.dataframe(cluster_players, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
