import warnings

import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIGURATION (set once for the whole multipage app)
# ============================================================================
st.set_page_config(
    page_title="DataDriven Soccer Scouting",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# NAVIGATION
# ============================================================================
pages = {
    "": [
        st.Page("app_pages/home.py", title="Home", icon="🏠", default=True),
    ],
    "Hidden Gem Engine": [
        st.Page("app_pages/similarity_overview.py", title="Overview", icon="📊", url_path="similarity-overview"),
        st.Page("app_pages/similarity_search.py", title="Search Engine", icon="🔍", url_path="similarity-search"),
    ],
    "Anomaly Hunter - K-Means": [
        st.Page("app_pages/clustering_overview.py", title="Overview", icon="🧩", url_path="clustering-overview"),
        st.Page("app_pages/clustering_analysis.py", title="Cluster Analysis", icon="🔬", url_path="cluster-analysis"),
    ],
    "Anomaly Hunter - Autoencoder": [
        st.Page("app_pages/deep_overview.py", title="Overview", icon="🤖", url_path="deep-overview"),
        st.Page("app_pages/deep_anomaly_hub.py", title="Deep Anomaly Hub", icon="🔎", url_path="deep-anomaly-hub"),
    ],
}

st.navigation(pages).run()
