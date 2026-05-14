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
    page_title="Soccer Scouting - Anomaly Detection",
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
    st.title("⚽ Soccer Scouting - Anomaly Detection")
    
    # Load data
    df_clusters, df_cluster_profile, glossary_dict = load_data()
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select Section", ["Overview", "Cluster Analysis", "Compare Clusters"])
    
    # ========================================================================
    # PAGE 1: OVERVIEW
    # ========================================================================
    if page == "Overview":
        st.header("K-Means Clustering Overview")
        
        with st.expander("📚 What is K-Means Clustering?", expanded=True):
            st.markdown("""
            **K-Means Clustering** is an unsupervised machine learning algorithm that groups players 
            into clusters based on their statistical profiles (e.g., passing, shooting, dribbling, defense, etc.).
            
            ### How It Works:
            1. **Initialization**: The algorithm starts with k randomly selected cluster centers (centroids).
            2. **Assignment**: Each player is assigned to the nearest cluster. *(Thanks to our normalization process, this distance effectively acts as Cosine Similarity, grouping players by the "angle" of their playstyle rather than sheer volume).*
            3. **Update**: Cluster centers are recalculated based on the mean of assigned players.
            4. **Iteration**: Repeat until convergence.
            
            ### Why K-Means for Soccer Scouting?
            - **Tactical Profiling**: Players in the same cluster share similar playing styles, regardless of how much possession their team averages.
            - **Anomaly Detection**: Identify players who don't fit their assigned position (e.g., a CB playing like a deep-lying playmaker).
            - **Scouting Shortcuts**: Find similar players to targets across different leagues/teams.
            - **Talent Benchmarking**: Compare player profiles to established tactical templates.
            """)
        
        with st.expander("🔧 How We Built This Model", expanded=False):
            st.markdown("""
            ### Data Preparation:
            - **Removed Noisy Features**: Excluded playing time (`90s`, `Starts`) and penalty metrics (`PK`) to prevent the model from grouping players by "status" (starter vs. bench) instead of tactics.
            - **Standardization**: Scaled all features to mean=0, std=1 using `StandardScaler`.
            - **L2 Normalization (The Secret Sauce)**: Projected all player vectors to a length of 1 using `normalize(norm='l2')`. This eliminates the "Possession Bias", ensuring a player with 100 passes and 10 tackles is grouped with a player with 50 passes and 5 tackles (same 10:1 tactical ratio).
            - **Removed Goalkeepers**: Excluded GK (different statistical universe).
            - **Removed Missing Values**: Only players with complete feature profiles were included.
            
            ### Model Parameters:
            - **Number of Clusters (k)**: 20 clusters (Optimized to isolate specific tactical micro-roles).
            - **Initialization**: 50 random seeds (`n_init=50`) to guarantee absolute mathematical stability.
            - **Random State**: 42 for reproducibility.
            - **Metric**: Euclidean distance on L2-Normalized data (mathematically equivalent to **Cosine Similarity**).
            
            ### Cluster Profiles:
            Each cluster is characterized by its:
            - **Top 3 Positive Features (📈)**: The extreme Z-scores showing what the cluster excels at compared to the European average.
            - **Top 3 Negative Features (📉)**: The lowest Z-scores showing what the cluster actively avoids doing.
            - **Dominant Role**: Most common nominal position in the cluster.
            - **Scouting Report**: A bespoke, human-readable interpretation of the cluster's pure tactical playing style.
            """)
        
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
            
            st.plotly_chart(fig, width='stretch')
            st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)        
            
        # Display scouting reports in expanders
        st.subheader("Cluster Profiles & Scouting Reports")
        for idx in sorted(df_cluster_profile['cluster'].unique()):
            with st.expander(f"📋 Cluster {idx}: {df_cluster_profile.loc[idx, 'dominant_role']}"):
                st.write(df_cluster_profile.loc[idx, 'scouting_report'])
    
    # ========================================================================
    # PAGE 2: CLUSTER ANALYSIS
    # ========================================================================
    elif page == "Cluster Analysis":
        st.header("🔍 Interactive Cluster Analysis")
        
        # Cluster selector
        cluster_id = st.selectbox(
            "Select a Cluster to Analyze",
            options=sorted(df_clusters['cluster'].unique()),
            format_func=lambda x: f"Cluster {x}: {df_cluster_profile.loc[x, 'dominant_role']}"
        )
        
        # Run analysis
        st.subheader(f"Cluster {cluster_id} Analysis")
        analyze_cluster(cluster_id, df_clusters, df_cluster_profile, glossary_dict)
        
        # Display visualizations
        st.subheader("📈 Visualizations")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.plotly_chart(
                plot_cluster_positions(cluster_id, df_clusters),
                width='stretch'
            )
        
        with viz_col2:
            st.plotly_chart(
                plot_cluster_league(cluster_id, df_clusters),
                width='stretch'
            )
        
        # Player list for selected cluster
        st.subheader("👥 Players in This Cluster")
        cluster_players = df_clusters[df_clusters['cluster'] == cluster_id][
            ['player', 'pos', 'team', 'league', 'season', 'age', 'nation']
        ].sort_values(['pos', 'player']).reset_index(drop=True)
        st.dataframe(cluster_players, width='stretch', hide_index=True)
    
    # ========================================================================
    # PAGE 3: Overlapping Positions Analysis
    # ========================================================================
    elif page == "Compare Clusters":
        st.header("🔄 Compare Multiple Clusters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cluster_a = st.selectbox(
                "Select First Cluster",
                options=sorted(df_clusters['cluster'].unique()),
                format_func=lambda x: f"Cluster {x}: {df_cluster_profile.loc[x, 'dominant_role']}",
                key="cluster_a"
            )
        
        with col2:
            cluster_b = st.selectbox(
                "Select Second Cluster",
                options=sorted(df_clusters['cluster'].unique()),
                format_func=lambda x: f"Cluster {x}: {df_cluster_profile.loc[x, 'dominant_role']}",
                key="cluster_b",
                index=1
            )
        
        # Comparison metrics
        st.subheader("Comparison")
        
        comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
        
        cluster_a_data = df_clusters[df_clusters['cluster'] == cluster_a]
        cluster_b_data = df_clusters[df_clusters['cluster'] == cluster_b]
        
        with comp_col1:
            st.metric(
                "Cluster A Size",
                len(cluster_a_data),
                f"vs {len(cluster_b_data)}"
            )
        with comp_col2:
            st.metric(
                "Cluster A Avg Age",
                f"{cluster_a_data['age'].mean():.1f}",
                f"vs {cluster_b_data['age'].mean():.1f}"
            )
        with comp_col3:
            st.metric(
                "Cluster A Positions",
                cluster_a_data['pos'].nunique(),
                f"vs {cluster_b_data['pos'].nunique()}"
            )
        with comp_col4:
            st.metric(
                "Cluster A Leagues",
                cluster_a_data['league'].nunique(),
                f"vs {cluster_b_data['league'].nunique()}"
            )
        
        # Side-by-side visualizations
        comp_viz1, comp_viz2 = st.columns(2)
        
        with comp_viz1:
            st.plotly_chart(
                plot_cluster_positions(cluster_a, df_clusters),
                width='stretch'
            )
        
        with comp_viz2:
            st.plotly_chart(
                plot_cluster_positions(cluster_b, df_clusters),
                width='stretch'
            )
        
        # Feature comparison table
        st.subheader("Feature Comparison")
        
        comparison_data = {
            'Feature': ['Top Positive 1', 'Top Positive 2', 'Top Positive 3',
                       'Top Negative 1', 'Top Negative 2', 'Top Negative 3'],
            f'Cluster {cluster_a}': [
                df_cluster_profile.loc[cluster_a, 'top_pos_1'],
                df_cluster_profile.loc[cluster_a, 'top_pos_2'],
                df_cluster_profile.loc[cluster_a, 'top_pos_3'],
                df_cluster_profile.loc[cluster_a, 'top_neg_1'],
                df_cluster_profile.loc[cluster_a, 'top_neg_2'],
                df_cluster_profile.loc[cluster_a, 'top_neg_3'],
            ],
            f'Cluster {cluster_b}': [
                df_cluster_profile.loc[cluster_b, 'top_pos_1'],
                df_cluster_profile.loc[cluster_b, 'top_pos_2'],
                df_cluster_profile.loc[cluster_b, 'top_pos_3'],
                df_cluster_profile.loc[cluster_b, 'top_neg_1'],
                df_cluster_profile.loc[cluster_b, 'top_neg_2'],
                df_cluster_profile.loc[cluster_b, 'top_neg_3'],
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, width='stretch', hide_index=True)


if __name__ == "__main__":
    main()
