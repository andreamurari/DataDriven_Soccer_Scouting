import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Soccer Scout", page_icon="⚽", layout="wide")
st.title("⚽ Data-Driven Soccer Scouting")
st.markdown("Search engine based on a Neural Autoencoder Ensemble.")

# --- 2. DATA LOADING (WITH CACHE) ---
@st.cache_data
def load_databases():
    # Loading the directly saved CSV files for Tanh and PCA
    # ATTENZIONE: Sostituisci i nomi dei file qui sotto con i nomi reali 
    # che hai dato ai file CSV estratti dai tuoi due modelli.
    df_tanh = pd.read_csv('saved_models/database_dna_tanh.csv') 
    df_pca = pd.read_csv('saved_models/database_dna_pca.csv')
    data = pd.read_csv('merged_data.csv')
    
    return df_tanh, df_pca, data

# Carichiamo i DataFrame con i nuovi nomi
df_latente_tanh, df_latente_pca, data = load_databases()

# (Nota: La riga 'colonne_ae = ...' è stata rimossa perché ora 
# la rilevazione delle colonne latenti avviene dinamicamente nella funzione)

# List of available seasons for the dropdown menus (using the Tanh DF as reference)
available_seasons = sorted(df_latente_tanh['season'].unique(), reverse=True)
available_players = sorted(df_latente_tanh['player'].unique())
player_to_seasons = (
    df_latente_tanh.groupby('player')['season']
    .apply(lambda seasons: sorted(seasons.unique(), reverse=True))
    .to_dict()
)

# --- 3. SEARCH ENGINE FUNCTION ---

def search_similar_players(
    player_name, 
    player_season, 
    df_latente_tanh, 
    df_latente_pca,
    top_n, 
    max_age, 
    season_filter, 
    same_macro_position, 
    same_league, 
    preferred_foot_filter,
    weight_tanh=0.70,
    weight_pca=0.30
):
    # Find the target player (using Tanh df as baseline structure)
    mask_target = (
        (df_latente_tanh['player'].str.lower() == player_name.lower())
        & (df_latente_tanh['season'] == player_season)
    )
    giocatore_idx = df_latente_tanh[mask_target].index

    if len(giocatore_idx) == 0:
        return None, "Player not found in the database. Please check the spelling and season."

    idx = giocatore_idx[0]
    
    # Extract target player info to show in the UI and use for filters
    target_info = {
        'player': df_latente_tanh.loc[idx, 'player'],
        'team': df_latente_tanh.loc[idx, 'team'],
        'season': df_latente_tanh.loc[idx, 'season'],
        'macro_pos': df_latente_tanh.loc[idx, 'macro_pos'],
        'league': df_latente_tanh.loc[idx, 'league'],
        'preferred_foot': df_latente_tanh.loc[idx, 'preferred_foot']
    }

    # Dynamically detect latent columns for both dataframes (handling different prefixes like AE_ or PCA_)
    cols_tanh = [col for col in df_latente_tanh.columns if col.startswith('AE_') or col.startswith('PCA_')]
    cols_pca = [col for col in df_latente_pca.columns if col.startswith('AE_') or col.startswith('PCA_')]

    # Extract vectors
    vettore_tanh = df_latente_tanh.loc[idx, cols_tanh].values.reshape(1, -1)
    vettore_pca = df_latente_pca.loc[idx, cols_pca].values.reshape(1, -1)
    
    # Calculate raw cosine similarities
    sim_tanh = cosine_similarity(vettore_tanh, df_latente_tanh[cols_tanh].values)[0]
    sim_pca = cosine_similarity(vettore_pca, df_latente_pca[cols_pca].values)[0]
    
    # STANDARDIZATION (Z-SCORE) & WEIGHTED ENSEMBLE
    z_sim_tanh = (sim_tanh - np.mean(sim_tanh)) / np.std(sim_tanh)
    z_sim_pca = (sim_pca - np.mean(sim_pca)) / np.std(sim_pca)
    z_ensemble = (z_sim_tanh * weight_tanh) + (z_sim_pca * weight_pca)

    # Apply similarities to the dataframe
    df_temp = df_latente_tanh.copy()
    df_temp['Z_Score_Ensemble'] = z_ensemble
    
    # Min-Max Scaling to convert Z-Score back into a readable 0-100% format
    min_z = z_ensemble.min()
    max_z = z_ensemble.max()
    df_temp['Match_Score_Scaled'] = (z_ensemble - min_z) / (max_z - min_z)

    # Remove the target player
    df_temp = df_temp.drop(index=idx) 

    # --- APPLY FILTERS ---
    if max_age is not None:
        df_temp = df_temp[df_temp['age'] <= max_age]

    if season_filter is not None and season_filter != "All":
        df_temp = df_temp[df_temp['season'] == season_filter]

    if same_macro_position:
        df_temp = df_temp[df_temp['macro_pos'] == target_info['macro_pos']]

    if same_league:
        df_temp = df_temp[df_temp['league'] == target_info['league']]

    if preferred_foot_filter in ["Left", "Right"]:
        df_temp = df_temp[
            df_temp['preferred_foot'].fillna('').str.lower() == preferred_foot_filter.lower()
        ]

    # Failsafe: Ensure the target player isn't in the list (e.g., from a different season)
    df_temp = df_temp[df_temp['player'] != target_info['player']]

    # Deduplicate: Keep only the best match for each player (across all seasons)
    df_temp = df_temp.loc[df_temp.groupby('player')['Z_Score_Ensemble'].idxmax()]

    # Sort using the TRUE mathematical ensemble (Z-Score) and get top N
    simili = df_temp.sort_values(by='Z_Score_Ensemble', ascending=False).head(top_n).copy()
    
    # Format the UI Match % using the scaled score
    simili['Match %'] = (simili['Match_Score_Scaled'] * 100).round(1).astype(str) + '%'

    return target_info, simili[['player', 'age', 'team', 'macro_pos', 'league', 'preferred_foot', 'season', 'Match %']]# --- 4. USER INTERFACE (SIDEBAR) ---

with st.sidebar:
    st.header("🎯 Target Player")
    input_name = st.selectbox("Player Name:", available_players, index=None, placeholder="Select Player")
    available_player_seasons = player_to_seasons.get(input_name, []) if input_name else []
    input_season = st.selectbox(
        "Reference Season:",
        available_player_seasons,
        index=0 if available_player_seasons else None,
        placeholder="Select player first",
        disabled=input_name is None or len(available_player_seasons) == 0
    )
    
    st.divider()
    
    st.header("⚙️ Search Filters")
    num_results = st.slider("Number of results (Top N):", min_value=1, max_value=15, value=5)
    
    # Filter: Max Age
    val_max_age = st.slider("Max Age:", min_value=16, max_value=40, value=25)
    # Filter: Season
    season_options = ["All"] + list(available_seasons)
    val_season_filter = st.selectbox("Search in Season:", season_options)
    # Filter: Preferred Foot
    val_preferred_foot = st.selectbox("Preferred Foot:", ["All", "Left", "Right"])
    # Filter: Position & League
    val_same_pos = st.checkbox("Same Position Only")
    val_same_league = st.checkbox("Same League Only")
    
    st.divider()
    with st.expander("⚙️Ensemble Weights"):
        val_weight_tanh = st.slider("Tactical Identity (Tanh %)", min_value=0.0, max_value=1.0, value=0.70, step=0.05)
        val_weight_pca = 1.0 - val_weight_tanh
        st.caption(f"Statistical Impact (PCA %) will be: {val_weight_pca:.2f}")
        
    btn_search = st.button(
        "Find Similar Players",
        type="primary",
        use_container_width=True,
        disabled=input_name is None or input_season is None
    )
    

# --- 5. EXECUTE SEARCH ---
if btn_search and input_name is not None and input_season is not None:
    with st.spinner("Scanning the European database..."):
        # Format the season filter variable for the function
        func_season_filter = None if val_season_filter == "All" else val_season_filter
        
        target_data, df_results = search_similar_players(
            player_name=input_name,
            player_season=input_season,
            df_latente_tanh=df_latente_tanh,  
            df_latente_pca=df_latente_pca,    
            top_n=num_results,
            max_age=val_max_age,
            season_filter=func_season_filter,
            same_macro_position=val_same_pos,    
            same_league=val_same_league,
            preferred_foot_filter=val_preferred_foot,
            weight_tanh=val_weight_tanh,                    
            weight_pca=val_weight_pca                      
)
        
        if target_data is None:
            # target_data is None means the second variable contains the error message
            st.error(df_results) 
        else:
            # Display Success Header
            st.success(f"Analysis completed for: **{target_data['player']}**")
            st.info(f"📍 **Role:** {target_data['macro_pos']} | 🛡️ **Team:** {target_data['team']} | 🌍 **League:** {target_data['league']} | 🦶 **Foot:** {target_data['preferred_foot']} | 📅 **Season:** {target_data['season']}")
            
            # Check if filters were too restrictive
            if df_results.empty:
                st.warning("No players found matching your strict filter criteria. Try relaxing the filters (e.g., max age or same league).")
            else:
                # Display the dataframe elegantly
                st.dataframe(
                    df_results,
                    use_container_width=True,
                    hide_index=True
                )