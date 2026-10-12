# VibeStream: Hybrid Music Recommendation Engine

VibeStream is a professional-grade AI music discovery system. Unlike simple genre-based filters, VibeStream analyzes the mathematical "DNA" of a song (acoustic features) and blends it with simulated user behavior to suggest tracks that truly match the vibe of your current selection.

---

## 1. How It Works (The Technical Logic)

The project operates on a **Hybrid Filtering Approach**, combining two main AI strategies:

### A. Content-Based Filtering (Acoustic Analysis)
Every song in the dataset is represented as a vector in an 11-dimensional space. We use the following features:
`Danceability, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo`.

1. **Normalization**: Since "Tempo" (e.g., 120 BPM) and "Danceability" (e.g., 0.8) have different scales, we use `MinMaxScaler` to squash all values between 0 and 1.
2. **Cosine Similarity**: When you pick a song, the AI calculates the cosine of the angle between your song's vector and every other song's vector. A smaller angle means a higher "Vibe Match."

### B. Collaborative Filtering Simulation (Behavioral Trends)
To make the AI feel more "human," we simulate user interaction matrices in `train.py`. This creates a behavioral map that identifies which songs are often liked together, regardless of their acoustic features.

### C. The Hybrid Blend
In **Hybrid Mode**, the system calculates:
`Final Score = (0.7 × Acoustic Similarity) + (0.3 × Behavioral Trend)`
This ensures the recommendation is musically similar but also includes "discovery" elements based on trends.

---

## ⚙️ 2. Complete Setup Guide

### Prerequisites
- Python 3.8 or higher installed on your system.
- A code editor (VS Code, PyCharm, etc.).

### Step-by-Step Installation
1. **Clone or Download the project folder** to your local machine.
2. **Open your Terminal / Command Prompt** and navigate to the project folder:
   ```bash
   cd path/to/Spotify_Song_project
   ```
3. **Install all required libraries**:
   ```bash
   pip install -r requirements.txt
   ```
   *(This installs Streamlit, Pandas, NumPy, and Scikit-Learn)*.

---

## 🚀 3. How to Run the Project

The project has a two-step execution process. **You cannot run the app without training the model first.**

### Step 1: Train the AI (The Backend)
Run the training script to clean the CSV data and generate the compressed model bundle (`.pkl` file).
```bash
python train.py
```
- **What happens?** This reads `spotify dataset.csv`, normalizes the data, creates the behavioral matrix, and saves everything into `spotify_hybrid_bundle.pkl`.

### Step 2: Launch the UI (The Frontend)
Run the Streamlit application to start the interactive web interface.
```bash
streamlit run app.py
```
- **What happens?** A local web server starts. Open the provided URL (usually `http://localhost:8501`) in your browser to start discovering music.

---

## 📂 4. Working with Different CSV Files

VibeStream is designed to be flexible. You can use your own music datasets as long as they follow specific rules.

### CSV Requirements
To use a different CSV file, ensure your file has the following columns:
1. **Identification**: `track_id`, `track_name`, `track_artist`.
2. **Acoustic Features**: All of these must be present for the AI to work:
   - `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`.

### How to swap the dataset:
1. Rename your new CSV file to `spotify dataset.csv` (or change the filename inside `train.py` line 10).
2. **CRITICAL**: You must run `python train.py` again. 
   - The AI doesn't read the CSV directly during the app run; it reads the `.pkl` bundle. If you change the CSV but don't re-run `train.py`, the app will still show the old songs.
3. Restart the app using `streamlit run app.py`.

---

## 🛠️ Summary Checklist
| Goal | Action | Command |
|---|---|---|
| Setup | Install Deps | `pip install -r requirements.txt` |
| New Data | Prepare CSV | Ensure acoustic columns exist |
| Process Data | Train Model | `python train.py` |
| Start App | Run UI | `streamlit run app.py` |
