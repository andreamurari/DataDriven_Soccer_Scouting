import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Soccer Scout", page_icon="⚽", layout="wide")
st.title("⚽ Data-Driven Soccer Scouting")
st.markdown("Search engine based on a Neural Autoencoder Ensemble.")

# --- 2. DATA LOADING (WITH CACHE) ---
@st.cache_data
def load_databases():
    # Loading the directly saved CSV files
    df_a = pd.read_csv('saved_models/database_dna_a.csv')
    df_b = pd.read_csv('saved_models/database_dna_b.csv')
    df_c = pd.read_csv('saved_models/database_dna_c.csv')
    data = pd.read_csv('merged_data.csv')
    return df_a, df_b, df_c, data

df_latente_a, df_latente_b, df_latente_c, data = load_databases()

def add_preferred_foot(df_latente, merged_data):
    def normalize_season(value):
        text = str(value).strip()
        if len(text) == 9 and text[4] == '-' and text[:4].isdigit() and text[5:].isdigit():
            return text[2:4] + text[7:9]
        if len(text) == 11 and text[4:7] == ' - ' and text[:4].isdigit() and text[7:].isdigit():
            return text[2:4] + text[9:11]
        return text

    left = df_latente.copy()
    right = merged_data[['league', 'season', 'team', 'player', 'Preferred foot']].copy()

    left['season_norm'] = left['season'].map(normalize_season)
    right['season_norm'] = right['season'].map(normalize_season)

    right = right.drop(columns=['season']).drop_duplicates(
        subset=['league', 'season_norm', 'team', 'player'], keep='first'
    )

    merged = left.merge(
        right,
        on=['league', 'season_norm', 'team', 'player'],
        how='left'
    )
    merged = merged.drop(columns=['season_norm'])
    return merged.rename(columns={'Preferred foot': 'preferred_foot'})

df_latente_a = add_preferred_foot(df_latente_a, data)
df_latente_b = add_preferred_foot(df_latente_b, data)
df_latente_c = add_preferred_foot(df_latente_c, data)
colonne_ae = [f'AE_{i+1}' for i in range(10)]

# List of available seasons for the dropdown menus
available_seasons = sorted(df_latente_a['season'].unique(), reverse=True)
available_players = sorted(df_latente_a['player'].unique())
player_to_seasons = (
    df_latente_a.groupby('player')['season']
    .apply(lambda seasons: sorted(seasons.unique(), reverse=True))
    .to_dict()
)

# --- 3. SEARCH ENGINE FUNCTION ---
def search_similar_players(player_name, player_season, top_n, max_age, season_filter, same_position, same_league, preferred_foot_filter):
    # Find the target player
    mask_target = (
        (df_latente_a['player'].str.lower() == player_name.lower())
        & (df_latente_a['season'] == player_season)
    )
    giocatore_idx = df_latente_a[mask_target].index

    if len(giocatore_idx) == 0:
        return None, "Player not found in the database. Please check the spelling and season."

    idx = giocatore_idx[0]
    
    # Extract target player info to show in the UI and use for filters
    target_info = {
        'player': df_latente_a.loc[idx, 'player'],
        'team': df_latente_a.loc[idx, 'team'],
        'season': df_latente_a.loc[idx, 'season'],
        'pos': df_latente_a.loc[idx, 'pos'],
        'league': df_latente_a.loc[idx, 'league'],
        'preferred_foot': df_latente_a.loc[idx, 'preferred_foot']
    }

    # Average cosine similarity from the three latent spaces
    vettore_a = df_latente_a.loc[idx, colonne_ae].values.reshape(1, -1)
    vettore_b = df_latente_b.loc[idx, colonne_ae].values.reshape(1, -1)
    vettore_c = df_latente_c.loc[idx, colonne_ae].values.reshape(1, -1)
    
    sim_a = cosine_similarity(vettore_a, df_latente_a[colonne_ae].values)[0]
    sim_b = cosine_similarity(vettore_b, df_latente_b[colonne_ae].values)[0]
    sim_c = cosine_similarity(vettore_c, df_latente_c[colonne_ae].values)[0]
    
    sim_ensemble = (sim_a + sim_b + sim_c) / 3

    # Apply similarities to the dataframe
    df_temp = df_latente_a.copy()
    df_temp['Similarity_Score'] = sim_ensemble
    df_temp = df_temp.drop(index=idx) # Remove the target player

    # --- APPLY FILTERS ---
    if max_age is not None:
        df_temp = df_temp[df_temp['age'] <= max_age]

    if season_filter is not None and season_filter != "All":
        df_temp = df_temp[df_temp['season'] == season_filter]

    if same_position:
        df_temp = df_temp[df_temp['pos'] == target_info['pos']]

    if same_league:
        df_temp = df_temp[df_temp['league'] == target_info['league']]

    if preferred_foot_filter in ["Left", "Right"]:
        df_temp = df_temp[
            df_temp['preferred_foot'].fillna('').str.lower() == preferred_foot_filter.lower()
        ]

    # Failsafe: Ensure the target player isn't in the list (e.g., from a different season)
    df_temp = df_temp[df_temp['player'] != target_info['player']]

    # Sort and get top N
    simili = df_temp.sort_values(by='Similarity_Score', ascending=False).head(top_n).copy()
    simili['Match %'] = (simili['Similarity_Score'] * 100).round(1).astype(str) + '%'

    return target_info, simili[['player', 'age', 'team', 'pos', 'league', 'preferred_foot', 'season', 'Match %']]

# --- 4. USER INTERFACE (SIDEBAR) ---
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
    use_max_age = st.checkbox("Filter by Maximum Age")
    val_max_age = None
    if use_max_age:
        val_max_age = st.slider("Max Age:", min_value=16, max_value=40, value=25)
        
    # Filter: Season
    season_options = ["All"] + list(available_seasons)
    val_season_filter = st.selectbox("Search in Season:", season_options)
    
    # Filter: Position & League
    val_same_pos = st.checkbox("Same Position Only")
    val_same_league = st.checkbox("Same League Only")
    val_preferred_foot = st.selectbox("Preferred Foot:", ["All", "Left", "Right"])
    
    st.divider()
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
            top_n=num_results,
            max_age=val_max_age,
            season_filter=func_season_filter,
            same_position=val_same_pos,
            same_league=val_same_league,
            preferred_foot_filter=val_preferred_foot
        )
        
        if target_data is None:
            # target_data is None means the second variable contains the error message
            st.error(df_results) 
        else:
            # Display Success Header
            st.success(f"Analysis completed for: **{target_data['player']}**")
            st.info(f"📍 **Role:** {target_data['pos']} | 🛡️ **Team:** {target_data['team']} | 🌍 **League:** {target_data['league']} | 🦶 **Foot:** {target_data['preferred_foot']} | 📅 **Season:** {target_data['season']}")
            
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