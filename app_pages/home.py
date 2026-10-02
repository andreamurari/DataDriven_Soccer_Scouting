import streamlit as st

st.title("⚽ DataDriven Soccer Scouting")
st.markdown("""
**Deep Player Embeddings: Dimensionality Reduction & Anomaly Detection in European Soccer**

Unsupervised machine learning for tactical scouting. We compress **~115 technical, tactical and physical metrics**
for **~4,000 players** from the Top-5 European leagues into a compact *"Tactical DNA"*, then use it to answer two scouting questions.
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader("🎯 Hidden Gem Engine")
        st.markdown("""
        *Who plays like this player?*

        A PCA + Deep Autoencoder ensemble ranks every player by cosine similarity
        to a chosen target, surfacing affordable alternatives to elite players.
        """)
        st.page_link("app_pages/similarity_overview.py", label="Overview", icon="📊")
        st.page_link("app_pages/similarity_search.py", label="Search Engine", icon="🔍")

with col2:
    with st.container(border=True):
        st.subheader("👽 Anomaly Hunter: K-Means")
        st.markdown("""
        *Who breaks the mould?*

        Players are grouped purely by playing style; positional minorities inside
        homogeneous clusters reveal unconventional tactical profiles.
        """)
        st.page_link("app_pages/clustering_overview.py", label="Overview", icon="🧩")
        st.page_link("app_pages/clustering_analysis.py", label="Cluster Analysis", icon="🔬")

with col3:
    with st.container(border=True):
        st.subheader("🤖 Anomaly Hunter: Autoencoder")
        st.markdown("""
        *Who breaks the mould? (deep view)*

        One autoencoder per macro-position; players it fails to reconstruct are
        flagged as true tactical outliers, with the stats that explain why.
        """)
        st.page_link("app_pages/deep_overview.py", label="Overview", icon="🤖")
        st.page_link("app_pages/deep_anomaly_hub.py", label="Deep Anomaly Hub", icon="🔎")

st.divider()

st.caption(
    "Data: [FBref](https://fbref.com/) via `soccerdata` (performance) and EA Sports FC 24 via Kaggle (physical attributes). "
    "Leagues: Premier League, La Liga, Serie A, Bundesliga, Ligue 1. Seasons 2022/23 to 2024/25. "
    "Educational project, non-commercial use only."
)
