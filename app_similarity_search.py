import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Soccer Scout", page_icon="⚽", layout="wide")
st.title("⚽ Data-Driven Soccer Scouting")
st.markdown("Unsupervised ML for tactical scouting. Mapping the *Tactical DNA* of elite players across Europe.")

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

    return target_info, simili[['player', 'age', 'team', 'macro_pos', 'league', 'preferred_foot', 'season', 'Match %']]

# --- 4. CREATE TABS ---
tab_overview, tab_search = st.tabs(["📊 Overview", "🔍 Search Engine"])

# ===== OVERVIEW TAB =====
with tab_overview:
    st.header("🎯 The Hidden Gem Engine")
    
    # --- OBIETTIVO E TASK (Aggiornato) ---
    st.markdown("""
    **Uncovering "Hidden Gems"**
    Modern soccer is overwhelmed by data. Our objective is to cut through the noise by compressing over 100 physical and technical metrics into a pure, mathematical **"tactical DNA"** for every player. 

    By standardizing this data, we can look past market values and league reputations to perform truly objective scouting. The result? We identify undervalued talent—affordable players in developing leagues who perfectly replicate the playing style and statistical output of world-class superstars.

    **The Technical Goal:** Compress multidimensional player data into a low-dimensional *"latent space"* to mathematically match tactical profiles and find data-backed, cost-effective replacements for elite players."""
    )
    
    st.divider()
    
    # --- METHODOLOGY OVERVIEW (Nuova sezione) ---
    with st.expander("📈 Dataset Overview"):
        try:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Players", f"{len(df_latente_tanh):,}")
            with col2:
                st.metric("Seasons Covered", len(available_seasons))
            with col3:
                st.metric("Features Analyzed", "109")
            
            st.markdown(f"""
    **Coverage:**
    - **Leagues:** Premier League, La Liga, Serie A, Bundesliga, Ligue 1
    - **Positions:** Field players only (excluding Goalkeepers)
    - **Macro Positions:** {', '.join(sorted(set(df_latente_tanh['macro_pos'].unique())))}
    - **Time Period:** {min(available_seasons)} to {max(available_seasons)}
            """)
        except Exception as e:
            st.info("Dataset details will be available once data is loaded.")
            
    with st.expander("🔬 The Methodology") :
        st.markdown("""
        To achieve this goal, we developed and compared two distinct approaches for dimensionality reduction and similarity matching:

        * **Baseline Model: Principal Component Analysis (PCA)** A linear reduction of dimensions to capture the maximum statistical variance, followed by the calculation of Cosine Similarity between the resulting player vectors.

        * **Advanced Model: Deep Autoencoder (Feedforward Neural Network)** Training a neural bottleneck model to capture complex, nonlinear relationships between statistics. We then extract embeddings directly from the bottleneck layer to perform Cosine Similarity.
        """)
    
    with st.expander("📊 Principal Component Analysis (PCA)"):
        st.markdown("""
    *Think of PCA like summarizing a player's game without losing the big picture.*
    
    **How It Works (The Methodology):**
    - **Data Processing:** Analyzes all 109 player statistics simultaneously to identify the main directions where players differ the most.
    - **Dimensionality Reduction:** Compresses the complexity from the original 109 dimensions down to just **27 principal components**.
    - **Variance Target:** These 27 components are precisely configured to retain exactly **95%** of the cumulative explained variance. This effectively reduces statistical noise while perfectly preserving the global statistical "weight" and volume of the player.
    - **Generalization (Validation):** Empirical validation confirms the model's stability, with the Mean Squared Error (MSE) remaining comparable and exceptionally low between the training (**0.0469**) and validation (**0.0526**) datasets.
    
    **Why It Matters (The Advantage):**
    - **Captures Volume & Intensity:** Excellent at identifying how much a player actually plays and their overall game intensity.
    - **Global Perspective:** Sees players' overall contribution to the game, rather than just their positional movement.
    - **Outlier Detection:** Flags unusual or inflated statistical profiles.
    - **Quality Filter:** Ensures similar players have a comparable workload and impact on the pitch.
    
    **Soccer Application:**
    *"Which other players have a similar overall game intensity and statistical volume, proving they can handle the workload of the elite player we want to replace?"*
        """)
                    
    with st.expander("🧠 Deep Autoencoder with Bottleneck"):
        st.markdown("""
    *Think of an Autoencoder as a compression and decompression machine for tactical DNA.*

    **How It Works (The Architecture):**
    Built as a symmetric feedforward neural network (`Input → 64 → 32 → 16 → 32 → 64 → Output`), the process involves:
    1. **Compression Phase (Encoder):** The original 109 statistical dimensions are compressed down through hidden layers (64 nodes, then 32 nodes).
    2. **Latent Space (Bottleneck):** The tightest point of compression. A **16-node linear layer** (`dna_bottleneck`) that extracts a highly compressed, dense tactical signature.
    3. **Decompression Phase (Decoder):** The network rebuilds the 109 original stats from the bottleneck (`16 → 32 → 64 → Output`).
    4. **Learning:** By comparing the input to the output, the model learns to preserve only important non-linear tactical patterns and discard noise.

    **The Technical Advantage:**
    - **Nonlinear Tactical Extraction (Tanh):** The **Tanh** activation function is used across all hidden layers. It captures complex, non-linear tactical relationships while maintaining negative values, ensuring strict positional boundaries and role nuances are respected.
    - **Extreme Value Preservation (Huber Loss):** Explicitly chosen over standard MSE. Soccer statistics naturally contain extreme outliers (e.g., highly skewed metrics for low-playtime players). The **Huber loss** prevents these rare anomalies from disproportionately distorting the network's weights.
    - **Forced Selectivity:** Compressing 109 dimensions into just 16 nodes forces the model to retain only the absolute core of a player's behavior: their pure Tactical DNA.

    **Soccer Application:**
    *"Who has the exact same tactical DNA—role, movement patterns, and pitch positioning—regardless of how famous they are?"*
        """)    
    
    with st.expander("📐 Cosine Similarity in Latent Space"):
        st.markdown("""
        *Once we compress a player's data into a "latent space" (16 dimensions for the Autoencoder, 27 for PCA), we need a mathematical way to compare them. We use **Cosine Similarity**.*

        **How It Works:**
        Instead of measuring the straight-line distance between two players (which is heavily skewed by playing time), Cosine Similarity measures the **angle** between their data vectors. 

        **Why Angles Matter in Scouting:**
        * **Direction = Playing Style:** If two players have the exact same tactical behavior, their vectors point in the exact same direction, even if their overall stat accumulation is different.
        * **Magnitude is Ignored:** This is the secret to finding "hidden gems." A young prospect playing 1,500 minutes can have the exact same "angle" (tactical DNA) as a superstar playing 3,500 minutes.

        **The Scale:**
        * **+1.0 (100%):** Identical tactical profile (vectors point exactly the same way).
        * **0.0 (0%):** Completely unrelated profiles (vectors are perpendicular).
        * **-1.0 (-100%):** Exact opposite profiles (vectors point in opposite directions).
            """)
        
    with st.expander("💡 The Solution: Z-Score Weighted Ensemble"):
        st.markdown("""
    To achieve the perfect balance between tactical fit and player quality, we combined the strengths of both models into a single **Z-Score Weighted Ensemble**.

    **The Winning Architecture:**
    * **70% Tanh Deep Autoencoder (Tactical Precision):** Finds players with identical tactical profiles and pitch behavior.
    * **30% Principal Component Analysis (Volume & Quality):** Ensures similar playing intensity and statistical volume.
    
    **🤝 The Synergy:**
    By blending these approaches, the engine discovers affordable alternatives who play exactly the same way (Autoencoder) while guaranteeing they actually have a comparable and reliable impact on the game (PCA).
        """)
    
    with st.expander("🛠️ Why This Architecture Wins"):
        st.markdown("""
    #### 1️⃣ Discarding Dropout & ReLU
    * **The Problem:** Dropout adds artificial noise (confusing distinct roles), ReLU flattens negative statistics (losing tactical nuances)
    * **The Fix:** Pure **Tanh** without dropout preserves "tactical DNA" and respects pitch position
        """)
        
        st.markdown("""
    #### 2️⃣ Overcoming the Limits of Pure PCA
    * **The Problem:** PCA brilliantly captures global game volume, but is **heavily biased toward expensive superstars** with inflated statistics. A midfielder who plays 2,500 minutes with elite stats will always rank higher than an undervalued talent playing 1,800 minutes with similar tactical patterns. Used alone, PCA fails the "affordable replacement" objective.
    * **The Solution:** Combine with Autoencoder (70%) which prioritizes **tactical patterns over volume**, allowing hidden gems with slightly lower playing time but identical tactical DNA to surface.
        """)
        
        st.markdown("""
    #### 3️⃣ The Power of Z-Score Standardization
    * **The Problem (Raw Averaging):** - Tanh similarities range from -1 to +1 (tight distribution)
      - PCA similarities might range 0.50 to 0.98 (different scale)
      - Simply averaging them (50% + 50%) is mathematically invalid
      
    * **The Solution (Z-Score):**
      - Convert each model's scores to **standard deviations from the mean** (Z-Scores)
      - This puts both models on the **same statistical scale** (mean=0, std=1)
      - Now we can safely weight them: 70% of normalized Tanh + 30% of normalized PCA
      - Result: Fair "apples-to-apples" comparison across both latent spaces
        """)
        
        st.markdown("""
    #### 4️⃣ The 70/30 Strategic Weighting
    * **70% Tanh (The Scout):** Prioritizes **tactical discipline and positional fidelity**. Finds players who behave identically tactically, regardless of playing time.
    * **30% PCA (The Filter):** Acts as **quality assurance**. Ensures the recommended player isn't just tactically similar but also has comparable **game intensity and statistical volume** (doesn't recommend a backup player for a starter).
    * **Why Not 50/50?** Tactical patterns (Tanh) are more predictive of "can this player actually replace this star?" than raw volume (PCA). Hence 70% to Tanh.
        """)
    
    with st.expander("🔄 How It Works"):
        st.markdown("""
    1. **Input:** Select a target player and reference season
    2. **Feature Extraction:** Calculate tactical DNA using both Tanh autoencoder and PCA
    3. **Similarity Computation:** Compute cosine similarities across both latent spaces
    4. **Score Standardization:** Convert to Z-Scores for fair ensemble weighting
    5. **Filtering & Deduplication:** Apply user filters and keep best match per player
    6. **Results:** Display top N similar players ranked by ensemble score
        """)
        
with tab_search:
    st.header("🎯 Target Player Selection")
    
    col1, col2 , col3 = st.columns([2, 0.5, 2])
    with col1:
        input_name = st.selectbox("Player Name:", available_players, index=None, placeholder="Select Player")
    with col3:
        available_player_seasons = player_to_seasons.get(input_name, []) if input_name else []
        input_season = st.selectbox(
            "Reference Season:",
            available_player_seasons,
            index=0 if available_player_seasons else None,
            placeholder="Select player first",
            disabled=input_name is None or len(available_player_seasons) == 0
        )
    
    st.markdown("---")
    
    st.header("⚙️ Search Filters")
    
    col1, col2, col3 = st.columns([2,0.5,2])
    with col1:
        num_results = st.slider("Number of results (Top N):", min_value=1, max_value=15, value=5)
    with col3:
        val_max_age = st.slider("Max Age:", min_value=16, max_value=40, value=25)
    
    col1, col2, col3 = st.columns([2, 0.5, 2])
    with col1:
        season_options = ["All"] + list(available_seasons)
        val_season_filter = st.selectbox("Search in Season:", season_options)
    with col3:
        val_preferred_foot = st.selectbox("Preferred Foot:", ["All", "Left", "Right"])
    
    col1, col2, col3 = st.columns([2, 0.5, 2])
    with col1:
        val_same_pos = st.checkbox("Same Position Only")
    with col3:
        val_same_league = st.checkbox("Same League Only")
    
    st.markdown("---")
    
    st.markdown("**⚖️ Ensemble Weights** | *Tactical DNA vs Playing Intensity:*")
    
    col1, col2, col3 = st.columns([2, 0.5, 2])
    
    with col1:
        val_weight_tanh = st.slider("Tanh %", min_value=0.0, max_value=1.0, value=0.70, step=0.05, label_visibility="collapsed")
        val_weight_pca = 1.0 - val_weight_tanh
        st.caption(f"🎯 Tanh: {val_weight_tanh:.0%} | 📊 PCA: {val_weight_pca:.0%}")
    
    with col3:
        btn_search = st.button(
            "🔍 Find Similar Players",
            type="primary",
            use_container_width=True,
            disabled=input_name is None or input_season is None
        )
    
    st.markdown("---")
    
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