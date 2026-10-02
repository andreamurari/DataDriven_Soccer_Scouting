import streamlit as st

from data_loader import load_similarity_data

st.title("⚽ Data-Driven Football Scouting")
st.markdown("Unsupervised ML for tactical scouting. Mapping the *Tactical DNA* of elite players across Europe.")

df_latente_tanh, df_latente_pca = load_similarity_data()

# List of available seasons for the dropdown menus (using the Tanh DF as reference)
available_seasons = sorted(df_latente_tanh['season'].unique(), reverse=True)

st.header("🎯 The Hidden Gem Engine")

st.markdown("""
**Uncovering "Hidden Gems"**
Modern football is overwhelmed by data. Our objective is to cut through the noise by compressing over 100 physical and technical metrics into a pure, mathematical **"tactical DNA"** for every player. 

By standardizing this data, we can look past market values and league reputations to perform truly objective scouting. The result? We identify undervalued talent—affordable players in developing leagues who perfectly replicate the playing style and statistical output of world-class superstars.

**The Technical Goal:** Compress multidimensional player data into a low-dimensional *"latent space"* to mathematically match tactical profiles and find data-backed, cost-effective replacements for elite players.
""")

st.divider()

# 1. DATASET OVERVIEW
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

# 2. PCA (IL PUNTO DI PARTENZA)
with st.expander("📊 Baseline Model: Principal Component Analysis"):
    st.markdown("""
*Our starting point: summarizing a player's game without losing the big picture.*

**How It Works:**
- Compresses the 109 dimensions down to just **27 principal components**.
- These 27 components are configured to retain exactly **95%** of the cumulative explained variance.

**Why It Matters (The Baseline):**
- **Captures Volume & Intensity:** It is excellent at identifying how much a player actually plays and their overall game intensity.
- **The Limitation:** While it captures global impact perfectly, it is heavily biased toward expensive superstars and sometimes ignores strict positional discipline (e.g., suggesting a ball-playing center-back to replace a deep-lying playmaker).
    """)

# 3. AUTOENCODER (L'EVOLUZIONE)
with st.expander("🧠 Advanced Model: Deep Autoencoder"):
    st.markdown("""
*The evolution: a neural compression machine designed to isolate pure tactical DNA and fix the PCA's positional blind spots.*""")
    st.markdown("""            
**How It Works (The Architecture):**
- Built as a symmetric feedforward neural network, the process involves:
    1. **Compression Phase:** The 109 statistical dimensions are compressed down through hidden layers.
    2. **Latent Space (Bottleneck):** A **16-node linear layer** extracts a highly compressed, dense tactical signature.
    3. **Learning:** By rebuilding the original stats from the bottleneck, the model learns to preserve only important non-linear tactical patterns and discard noise.
    """)
    st.markdown("""        
- **The Winning Configuration:** After conducting empirical tests on 4 different architectures (including ReLU and Dropout variations), we selected a **Pure Tanh without Dropout**. 

- **Loss function**: Huber Loss to prevents extreme outliers (common in football stats) from distorting the network.
    """)

# 4. COSINE SIMILARITY
with st.expander("📐 Cosine Similarity in Latent Space"):
    st.markdown("""
    *Once we compress a player's data into a "latent space", we need a mathematical way to compare them.*

    Instead of measuring the straight-line distance between two players (which is heavily skewed by playing time), we calculate the **Cosine Similarity**, which measures the **angle** between their data vectors. 

    **Why Angles Matter in Scouting:**
    * **Direction = Playing Style:** If two players have the same tactical behavior, their vectors point in the same direction, even if their overall stat accumulation is different.
    * **Finding Gems:** A prospect playing 1,500 minutes can have the exact same "angle" (tactical DNA) as a superstar playing 3,500 minutes.
    """)

# 5. THE ENSEMBLE (LA FUSIONE)
with st.expander("🏆 The Solution: Z-Score Weighted Ensemble"):
    st.markdown("""
*Through rigorous empirical testing, we discovered that no single model was perfect on its own. We needed to fuse them.*

**The Synergy (70/30 Split):**
* **70% Tanh Autoencoder (The Scout):** Prioritizes tactical discipline and positional fidelity.
* **30% PCA (The Filter):** Acts as quality assurance to guarantee comparable statistical volume.

**The Math (Z-Score Standardization):**
Simply averaging the scores (50% + 50%) is mathematically invalid because Autoencoder and PCA similarities operate on different scales. To fix this:
1. We convert each model's scores to **standard deviations from the mean (Z-Scores)**.
2. This puts both models on the exact same statistical scale.
3. Result: A fair, "apples-to-apples" weighted ranking.
    """)

# 6. PIPELINE
with st.expander("🔄 Pipeline: How It Works"):
    st.markdown("""
1. **Input:** Select a target player and reference season.
2. **Extraction:** Calculate tactical DNA using both models.
3. **Similarity:** Compute cosine similarities across both latent spaces.
4. **Ensemble:** Standardize scores (Z-Score) and apply the 70/30 weighting.
5. **Results:** Apply user filters and display the top affordable alternatives!
    """)
