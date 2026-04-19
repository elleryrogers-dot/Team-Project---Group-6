#import libraries
import kagglehub
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------
# Download dataset
path = kagglehub.dataset_download("alitaqishah/spotify-wrapped-2025-top-songs-and-artists")
print("Path to dataset files:", path)

file_path = os.path.join(path, "spotify_wrapped_2025_top50_songs.csv")
df = pd.read_csv(file_path, encoding='latin1')

# Clean/rename columns
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df['primary_genre'] = df['primary_genre'].astype(str).str.strip().str.lower()
df = df.rename(columns={
    'streams_2025_billions': 'streams',
    'duration_seconds': 'duration_sec'
})

# Convert data to numeric values
numeric_cols = ['streams', 'bpm', 'duration_sec', 'danceability', 'energy']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['streams', 'bpm', 'duration_sec', 'primary_genre'])

# Create log-transformed streams for better correlation analysis
df['log_streams'] = np.log1p(df['streams'])

#-----------------------------------------------------------------------------------------------
# EXPERIMENT 1: Danceability vs Streams
if 'danceability' in df.columns:
    df_dance = df.dropna(subset=['danceability', 'streams'])

    x = df_dance['danceability']
    y = np.log1p(df_dance['streams'])

    plt.figure()
    plt.scatter(x, y)

    plt.xlabel("Danceability")
    plt.ylabel("Log(Streams)")
    plt.title("Danceability vs Streams")

    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b)

    corr_dance = np.corrcoef(x, y)[0, 1]
    print("Correlation (Danceability vs Log Streams):", corr_dance)

    plt.show()
else:
    print("Column 'danceability' not found in dataset.")
#Included in event that danceability was not considered for the top 50 songs
#-----------------------------------------------------------------------------------------------
# EXPERIMENT 2: BPM vs Streams
x_bpm = df['bpm']
y_streams = df['log_streams']

plt.figure()
plt.scatter(x_bpm, y_streams)

plt.xlabel("BPM")
plt.ylabel("Log(Streams)")
plt.title("BPM vs Streams (Spotify Top 50 Songs 2025)")

# Add linear trendline
m, b = np.polyfit(x_bpm, y_streams, 1)
plt.plot(x_bpm, m * x_bpm + b)

# Calculate correlation coefficient
correlation_bpm = np.corrcoef(x_bpm, y_streams)[0, 1]
print("Correlation (BPM vs Log Streams):", correlation_bpm)

plt.show()

#-----------------------------------------------------------------------------------------------
# EXPERIMENT 3: Genre vs Average Streams

# Categorize genres
def simplify_genre(g):
    if 'k-pop' in g:
        return 'k-pop'
    elif 'pop' in g:
        return 'pop'
    elif 'hip-hop' in g or 'rap' in g:
        return 'hip-hop/rap'
    elif 'rock' in g or 'punk' in g:
        return 'rock/punk'
    elif 'r&b' in g or 'soul' in g:
        return 'r&b/soul'
    elif 'country' in g:
        return 'country'
    elif 'reggaeton' in g:
        return 'reggaeton'
    elif 'afrobeats' in g:
        return 'afrobeats'
    elif 'indie' in g:
        return 'indie'
    else:
        return g

df['genre_clean'] = df['primary_genre'].apply(simplify_genre)

# Average streams per genre
genre_avg = (
    df.groupby('genre_clean')['streams']
    .mean()
    .sort_values()
)

# Plot bar chart
plt.figure(figsize=(12, 6))
genre_avg.plot(kind='bar')

plt.xlabel("Genre")
plt.ylabel("Average Streams (billions)")
plt.title("Average Streams by Genre (Spotify Top 50 Songs 2025)")

plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------------------------



