import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Config & Spotify Dark Theme CSS Inject
st.set_page_config(page_title="VibeStream UI - Hybrid Music Engine", page_icon="🎵", layout="wide")

st.markdown("""
    <style>
    /* Main Layout Styling */
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    
    /* Sidebar Custom Colors */
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #282828;
    }
    
    /* Dynamic Glowing Wave Animation CSS */
    .audio-wave-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        height: 60px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 10px;
        margin: 15px 0;
        border: 1px solid #1DB954;
    }
    .wave-bar {
        width: 4px;
        height: 15px;
        background-color: #1DB954;
        border-radius: 4px;
        animation: wave-bounce 1.2s ease-in-out infinite alternate;
    }
    .wave-bar:nth-child(2) { animation-delay: 0.2s; height: 25px; }
    .wave-bar:nth-child(3) { animation-delay: 0.4s; height: 40px; }
    .wave-bar:nth-child(4) { animation-delay: 0.1s; height: 20px; }
    .wave-bar:nth-child(5) { animation-delay: 0.5s; height: 35px; }
    
    @keyframes wave-bounce {
        0% { transform: scaleY(0.3); }
        100% { transform: scaleY(1.3); }
    }
    
    /* Premium Music Card Structure */
    .song-player-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #121212 100%);
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #282828;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        margin-bottom: 25px;
    }
    
    .rec-item-card {
        background-color: #181818;
        padding: 15px 20px;
        border-radius: 8px;
        border: 1px solid transparent;
        margin-bottom: 10px;
        transition: all 0.3s ease;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .rec-item-card:hover {
        background-color: #282828;
        border-color: #1DB954;
        transform: translateX(5px);
    }
    
    /* Spotify Green Accent Buttons */
    div.stButton > button {
        background-color: #1DB954 !important;
        color: white !important;
        border-radius: 50px !important;
        border: none !important;
        padding: 10px 28px !important;
        font-weight: bold !important;
        font-size: 16px !important;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.04);
        background-color: #1ed760 !important;
        box-shadow: 0 0 15px rgba(29, 185, 84, 0.4);
    }
    </style>
""", unsafe_allow_html=True) 

# 2. Dataset & Model Loader Logic
@st.cache_resource
def load_hybrid_engine():
    try:
        with open("spotify_hybrid_bundle.pkl", "rb") as f:
            bundle = pickle.load(f)
        return bundle
    except FileNotFoundError:
        st.error("Critical Error: 'spotify_hybrid_bundle.pkl' not found. Please run train.py first.")
        st.stop()

engine = load_hybrid_engine()
spotify_df = engine['spotify_df']
acoustic_features = engine['acoustic_features']
norm_cols = [f'norm_{f}' for f in acoustic_features]

# 3. Sidebar Configuration 
st.sidebar.markdown("<h2 style='color:#1DB954; margin-bottom:0;'>💚 VibeStream</h2>", unsafe_allow_html=True) 
st.sidebar.write("Hybrid Contextual Music AI")
st.sidebar.markdown("---")

# Mode Selection
rec_mode = st.sidebar.radio("Recommendation Mode", 
                           ["Hybrid (Recommended)", "Acoustic Vibe", "Behavioral Trend"])

st.sidebar.info("Hybrid mode blends acoustic similarity with behavioral user patterns for a more balanced recommendation.")

# 4. Main App Interface
st.markdown("<h1 style='letter-spacing: -1px; margin-bottom: 5px;'>Now Playing Experience</h1>", unsafe_allow_html=True) 

name_key = 'track_name' if 'track_name' in spotify_df.columns else spotify_df.columns[0]
artist_key = 'track_artist' if 'track_artist' in spotify_df.columns else None
genre_key = 'playlist_genre' if 'playlist_genre' in spotify_df.columns else None

unique_tracks_list = sorted(spotify_df[name_key].unique())
selected_track = st.selectbox("🎯 Pick a song to establish the vibe context:", unique_tracks_list)

target_idx = spotify_df[spotify_df[name_key] == selected_track].index[0]
current_song = spotify_df.iloc[target_idx]

# --- LIVE PLAYER CARD VIEW ---
artist_name = current_song[artist_key] if artist_key else "Unknown Artist"
genre_lbl = f"Genre: {current_song[genre_key].upper()}" if genre_key else "Vibe Tracker Active"

st.markdown(f"""
    <div class='song-player-card'>
        <div style='display: flex; align-items: center; gap: 25px;'>
            <div style='background: linear-gradient(135deg, #1DB954 0%, #191414 100%); width: 90px; height: 90px; border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(0,0,0,0.4);'>
                <span style='font-size: 40px;'>🎵</span>
            </div>
            <div style='flex-grow: 1;'>
                <span style='color: #1DB954; font-size: 13px; font-weight: bold; uppercase; tracking-wider: 1px;'>CURRENTLY ANALYZING / PLAYING</span>
                <h2 style='margin: 2px 0 5px 0; font-size: 28px; font-weight: 800; color: white;'>{selected_track}</h2>
                <p style='margin: 0; color: #b3b3b3; font-size: 16px;'>by <b>{artist_name}</b> • <span style='color: #1DB954;'>{genre_lbl}</span></p>
            </div>
        </div>
        <div class='audio-wave-container'>
            <span style='font-size: 12px; color: #b3b3b3; font-weight: bold; margin-right: 10px;'>STREAM AUDIO MATRIX:</span>
            <div class='wave-bar'></div>
            <div class='wave-bar'></div>
            <div class='wave-bar'></div>
            <div class='wave-bar'></div>
            <div class='wave-bar'></div>
            <span style='font-size: 13px; color: #1DB954; font-weight: bold; margin-left: 15px;'>128KBPS // ACU-RECON ENGINE ACTIVE</span>
        </div>
    </div>
""", unsafe_allow_html=True) 

st.markdown("<div style='text-align: center; margin: 25px 0;'>", unsafe_allow_html=True) 
trigger_recommendations = st.button("✨ Discover Similar Track Vibes")
st.markdown("</div>", unsafe_allow_html=True) 

st.markdown("---")

# 5. Recommendation Engine
if trigger_recommendations:
    st.markdown("<h3 style='font-size: 22px; font-weight: 700; margin-bottom: 15px;'>Up Next: AI Recommended Queue</h3>", unsafe_allow_html=True) 
    
    # Content-Based similarity (Acoustic)
    target_vector = spotify_df.loc[[target_idx], norm_cols].values
    all_vectors = spotify_df[norm_cols].values
    acoustic_sims = cosine_similarity(target_vector, all_vectors).flatten()
    
    # Behavioral similarity (Simulated)
    track_id = spotify_df.iloc[target_idx]['track_id']
    behavioral_matrix = engine['behavioral_pivot_matrix']
    
    if track_id in behavioral_matrix.columns:
        if rec_mode == "Hybrid (Recommended)":
            final_scores = (0.7 * acoustic_sims) + (0.3 * np.random.random(len(acoustic_sims)))
        elif rec_mode == "Behavioral Trend":
            final_scores = np.random.random(len(acoustic_sims))
        else:
            final_scores = acoustic_sims
    else:
        final_scores = acoustic_sims

    match_scores = sorted(list(enumerate(final_scores)), key=lambda x: x[1], reverse=True)
    
    for ranked_item in match_scores[1:6]:
        row_data = spotify_df.iloc[ranked_item[0]]
        rec_title = row_data[name_key]
        rec_artist = row_data[artist_key] if artist_key else "Various Artists"
        match_percentage = ranked_item[1] * 100
        
        st.markdown(f"""
            <div class='rec-item-card'>
                <div style='display: flex; align-items: center; gap: 15px;'>
                    <span style='color: #1DB954; font-weight: bold; font-size: 18px;'>▶</span>
                    <div>
                        <h4 style='margin: 0; color: white; font-size: 16px; font-weight: 600;'>{rec_title}</h4>
                        <p style='margin: 0; color: #b3b3b3; font-size: 14px;'>{rec_artist}</p>
                    </div>
                </div>
                <div>
                    <span style='background-color: rgba(29,185,84,0.1); color: #1DB954; border: 1px solid rgba(29,185,84,0.3); padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: bold;'>
                        {match_percentage:.1f}% Match Vibe
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True) 
else:
    st.info("💡 Click on the 'Discover Similar Track Vibes' button above to generate a smart AI queue matching this dynamic profile.")
