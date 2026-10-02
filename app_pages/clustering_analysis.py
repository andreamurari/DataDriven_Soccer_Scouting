import streamlit as st

from cluster_functions import analyze_cluster
from data_loader import load_cluster_data

st.title("⚽ Football Scouting - Anomaly Detection")

df_clusters, df_cluster_profile, glossary_dict = load_cluster_data()

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

st.dataframe(cluster_players, width='stretch', hide_index=True)
