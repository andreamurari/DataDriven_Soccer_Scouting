"""Cached data loaders shared by every page of the multipage app."""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESOURCES = ROOT / "resources"
SAVED_MODELS = ROOT / "saved_models"

POS_MAPPING = {
    'RB': 'Fullback',
    'LB': 'Fullback',
    'RW': 'Winger',
    'LW': 'Winger',
    'RM': 'Wide Midfielder',
    'LM': 'Wide Midfielder'
}


@st.cache_data
def load_merged_data():
    """Final joined FBref + EA FC dataset, with the macro-position mapping."""
    df_merged = pd.read_csv(ROOT / "merged_data.csv")
    df_merged['macro_pos'] = df_merged['Position'].replace(POS_MAPPING)
    return df_merged


@st.cache_data
def load_glossary():
    return pd.read_excel(RESOURCES / "glossary.xlsx")


@st.cache_data
def load_similarity_data():
    """Latent spaces exported from the Tanh autoencoder and the PCA."""
    df_tanh = pd.read_csv(SAVED_MODELS / "database_dna_tanh.csv")
    df_pca = pd.read_csv(SAVED_MODELS / "database_dna_pca.csv")
    return df_tanh, df_pca


@st.cache_data
def load_cluster_data():
    """K-Means cluster assignments, cluster profiles and the KPI glossary as a dict."""
    df_clusters = pd.read_csv(RESOURCES / "df_clusters.csv")
    df_clusters = df_clusters.dropna()
    df_clusters = df_clusters[df_clusters['pos'] != 'GK']

    df_cluster_profile = pd.read_csv(RESOURCES / "df_cluster_profile.csv")

    try:
        glossary = load_glossary()
        glossary_dict = dict(zip(glossary['KPI'], glossary['Explanation']))
    except Exception:
        glossary_dict = {}

    return df_clusters, df_cluster_profile, glossary_dict


@st.cache_data
def load_ae_anomalies():
    """Per-position autoencoder anomalies exported from the notebook."""
    return pd.read_csv(RESOURCES / "anomalies_per_pos_AE.csv")
