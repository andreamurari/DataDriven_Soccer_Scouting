import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from cluster_functions import plot_cluster_positions, plot_cluster_league
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
# HELPER FUNCTIONS
# ============================================================================

def analyze_cluster(cluster_id, df_clusters, df_cluster_profile, glossary_dict):
    """
    Display cluster analysis with dominant position, features, and anomalies
    """
    # Extract dominant position
    dominant_role = df_cluster_profile.loc[cluster_id, "dominant_role"]
    dominant_pos = dominant_role.split(" ")[0]
    
    # Get cluster data
    cluster_data = df_clusters[df_clusters['cluster'] == cluster_id]
    
    # Display analysis
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
        st.write("**Positive Features (Strengths):**")
        for col in ['top_pos_1', 'top_pos_2', 'top_pos_3']:
            feature = df_cluster_profile.loc[cluster_id, col]
            if feature != "-":
                st.write(f"• {feature}")
                # Add glossary explanation if available
                feature_name = feature.split(" (")[0]
                if feature_name in glossary_dict:
                    with st.expander(f"ℹ️ {feature_name}"):
                        st.write(glossary_dict[feature_name])
    
    with feature_cols[1]:
        st.write("**Negative Features (Weaknesses):**")
        for col in ['top_neg_1', 'top_neg_2', 'top_neg_3']:
            feature = df_cluster_profile.loc[cluster_id, col]
            if feature != "-":
                st.write(f"• {feature}")
                # Add glossary explanation if available
                feature_name = feature.split(" (")[0]
                if feature_name in glossary_dict:
                    with st.expander(f"ℹ️ {feature_name}"):
                        st.write(glossary_dict[feature_name])
    
    # Display scouting report
    st.subheader("🎯 Scouting Report")
    st.info(df_cluster_profile.loc[cluster_id, "scouting_report"])
    
    # Display position distribution in cluster
    st.subheader("📍 Position Distribution")
    pos_counts = cluster_data['pos'].value_counts().reset_index()
    pos_counts.columns = ['Position', 'Count']
    pos_counts['Percentage'] = (pos_counts['Count'] / len(cluster_data) * 100).round(1)
    st.dataframe(pos_counts, use_container_width=True, hide_index=True)
    
    # Display anomalies (players not matching dominant position)
    if len(cluster_data[cluster_data['pos'] != dominant_pos]) > 0:
        st.subheader("⚠️ Tactical Anomalies")
        anomalies = cluster_data[cluster_data['pos'] != dominant_pos][
            ['player', 'pos', 'team', 'season', 'league']
        ].sort_values(['pos', 'player'])
        st.dataframe(anomalies, use_container_width=True, hide_index=True)
    else:
        st.info(f"✅ No anomalies found in this cluster. All {len(cluster_data)} players are {dominant_pos}s.")




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
            into clusters based on their statistical profiles (e.g., pace, shooting, dribbling, defense, etc.).
            
            ### How It Works:
            1. **Initialization**: The algorithm starts with k randomly selected cluster centers (centroids)
            2. **Assignment**: Each player is assigned to the nearest cluster based on Euclidean distance
            3. **Update**: Cluster centers are recalculated based on the mean of assigned players
            4. **Iteration**: Steps 2-3 repeat until convergence
            
            ### Why K-Means for Soccer Scouting?
            - **Tactical Profiling**: Players in the same cluster share similar playing styles
            - **Anomaly Detection**: Identify players who don't fit their assigned position
            - **Scouting Shortcuts**: Find similar players to targets across different leagues/teams
            - **Talent Benchmarking**: Compare player profiles to established tactical templates
            """)
        
        with st.expander("🔧 How We Built This Model", expanded=False):
            st.markdown("""
            ### Data Preparation:
            - **Removed Noisy Features**: Excluded playing time and penalty metrics that introduce noise
            - **Standardization**: Scaled all features to mean=0, std=1 using StandardScaler
            - **Removed Goalkeepers**: Excluded GK (different statistical universe)
            - **Removed Missing Values**: Only players with complete feature profiles
            
            ### Model Parameters:
            - **Number of Clusters (k)**: 20 clusters
            - **Initialization**: 10 random seeds (n_init=10) for stability
            - **Random State**: 42 for reproducibility
            - **Metric**: Euclidean distance
            
            ### Cluster Profiles:
            Each cluster is characterized by its:
            - **Top 3 Positive Features**: What the cluster excels at
            - **Top 3 Negative Features**: What the cluster lacks
            - **Dominant Role**: Most common position in the cluster
            - **Scouting Report**: Human-readable interpretation of the cluster's playing style
            """)
        
        # Display cluster overview statistics
        st.subheader("📊 Cluster Overview")
        
        overview_metrics = st.columns(4)
        with overview_metrics[0]:
            st.metric("Total Clusters", 20)
        with overview_metrics[1]:
            st.metric("Total Players", len(df_clusters))
        with overview_metrics[2]:
            st.metric("Average Cluster Size", f"{len(df_clusters) // 20:.0f}")
        with overview_metrics[3]:
            st.metric("Unique Positions", df_clusters['pos'].nunique())
        
        # Cluster distribution table
        st.subheader("Cluster Distribution")
        cluster_dist = df_clusters.groupby('cluster').size().reset_index(name='Player Count')
        cluster_dist = cluster_dist.merge(
            df_cluster_profile[['cluster', 'dominant_role']],
            on='cluster'
        )
        st.dataframe(cluster_dist, use_container_width=True, hide_index=True)
        
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
                use_container_width=True
            )
        
        with viz_col2:
            st.plotly_chart(
                plot_cluster_league(cluster_id, df_clusters),
                use_container_width=True
            )
        
        # Player list for selected cluster
        st.subheader("👥 Players in This Cluster")
        cluster_players = df_clusters[df_clusters['cluster'] == cluster_id][
            ['player', 'pos', 'team', 'league', 'season', 'age', 'nation']
        ].sort_values(['pos', 'player']).reset_index(drop=True)
        st.dataframe(cluster_players, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # PAGE 3: COMPARE CLUSTERS
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
                use_container_width=True
            )
        
        with comp_viz2:
            st.plotly_chart(
                plot_cluster_positions(cluster_b, df_clusters),
                use_container_width=True
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
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
