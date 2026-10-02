import streamlit as st

from cluster_functions import (
    display_anomaly_scouting_report,
    display_single_anomaly_pct,
    plot_anomalies_per_age,
    plot_anomalies_per_league,
    plot_anomalies_per_macropos,
)
from data_loader import load_ae_anomalies, load_merged_data, load_glossary

st.title("🤖 Deep Player Embeddings - Anomaly Detection")

df_anomalies = load_ae_anomalies()
df_merged = load_merged_data()
df_glossary = load_glossary()

st.header("🔎 Deep Anomaly Hub: Tactical Explorer")

st.markdown("""
Use the filters below to isolate specific tactical roles, leagues, or age cohorts. 
The algorithms will dynamically recalculate anomaly frequencies and distributions based on your selection.
""")

col_pos, col_league, col_age, col_kpi = st.columns([2, 2, 2, 3])

with col_pos:
    macro_pos_order = ['CB', 'Fullback', 'CDM', 'CM', 'Wide Midfielder', 'CAM', 'Winger', 'ST']
    selected_pos = st.selectbox("🎯 Macro-Position", ["All Positions"] + macro_pos_order)

with col_league:
    leagues = sorted(df_merged['league'].dropna().unique())
    selected_league = st.selectbox("🌍 League", ["All Leagues"] + leagues)

with col_age:
    # Assicuriamoci che l'anno sia un numero intero per evitare i ".0" nella tendina
    ages = sorted(df_merged['born'].dropna().astype(int).unique())
    selected_age = st.selectbox("⏳ Birth Year", ["All Years"] + ages)

# =========================================================
# 2. IL MOTORE DI FILTRAGGIO (Master Filter)
# =========================================================
df_anomalies_filtered = df_anomalies.copy()
df_merged_filtered = df_merged.copy()

# Applichiamo il filtro Lega se necessario
if selected_league != "All Leagues":
    df_anomalies_filtered = df_anomalies_filtered[df_anomalies_filtered['league'] == selected_league]
    df_merged_filtered = df_merged_filtered[df_merged_filtered['league'] == selected_league]

# Applichiamo il filtro Età se necessario
if selected_age != "All Years":
    df_anomalies_filtered = df_anomalies_filtered[df_anomalies_filtered['born'] == selected_age]
    df_merged_filtered = df_merged_filtered[df_merged_filtered['born'] == selected_age]

# Gestiamo il ruolo per passarlo come parametro (sfruttando la logica già presente nelle tue funzioni)
filtro_pos = None if selected_pos == "All Positions" else selected_pos

# =========================================================
# 3. RENDER DEI GRAFICI CON I DATI FILTRATI
# =========================================================
with col_kpi:
    display_single_anomaly_pct(df_anomalies_filtered, df_merged_filtered, macro_pos=filtro_pos)

st.markdown("---")

if selected_pos == "All Positions":
    plot_anomalies_per_macropos(df_anomalies_filtered, df_merged_filtered)
    st.markdown("---")

plot_anomalies_per_league(df_anomalies_filtered, df_merged_filtered, macro_pos=filtro_pos)
plot_anomalies_per_age(df_anomalies_filtered, df_merged_filtered, macro_pos=filtro_pos)

# Tabella di scouting finale
st.markdown("---")
display_anomaly_scouting_report(df_anomalies_filtered, df_glossary, macro_pos=filtro_pos)
