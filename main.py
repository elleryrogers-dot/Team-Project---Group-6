#import libraries
import kagglehub
import pandas as pd
import os

#-----------------------------------------------------------------------------------------------

# Download dataset
path = kagglehub.dataset_download("alitaqishah/spotify-wrapped-2025-top-songs-and-artists")
print("Path to dataset files:", path)

# File selection
file_path = os.path.join(path, "spotify_wrapped_2025_top50_songs.csv")
df = pd.read_csv(file_path, encoding='latin1')

# Clean columns names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Rename columns
df = df.rename(columns={
    'streams_2025_billions': 'streams',
    'duration_seconds': 'duration_sec'
})

# Convert data to numeric values
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
df['bpm'] = pd.to_numeric(df['bpm'], errors='coerce')
df['duration_sec'] = pd.to_numeric(df['duration_sec'], errors='coerce')

df = df.dropna(subset=['streams', 'bpm', 'duration_sec', 'primary_genre'])
print(df.head())

#import analysis libraries
import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------------------------------------------------------------------
# Experiment 1: Duration vs Streams
x = df['duration_sec']
y = df['streams']

# Scatter plot
plt.figure()
plt.scatter(x, y)

# Axes Labels
plt.xlabel("Duration (seconds)")
plt.ylabel("Streams (billions)")
plt.title("Duration vs Streams (Spotify Top 50 Songs 2025)")

#Add linear trendline
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)

# Calculate correlation coefficient
correlation = np.corrcoef(x, y)[0, 1]
print("Correlation (Duration vs Streams):", correlation)

# Graph Output
plt.show()

#-------------------------------------------------------------------------------------------------
# EXPERIMENT 2: BPM vs Streams
x_bpm = df['bpm']
y_streams = df['streams']

#Scatter plot
plt.figure()
plt.scatter(x_bpm, y_streams)

#Axes Labels
plt.xlabel("BPM")
plt.ylabel("Streams (billions)")
plt.title("BPM vs Streams (Spotify Top 50 Songs 2025)")

# Add linear trendline
m, b = np.polyfit(x_bpm, y_streams, 1)
plt.plot(x_bpm, m*x_bpm + b)

# Calculate correlation coefficient
correlation_bpm = np.corrcoef(x_bpm, y_streams)[0, 1]
print("Correlation (BPM vs Streams):", correlation_bpm)

#Graph Output
plt.show()

# ----------------------------------------------------------------------------------------------------
# EXPERIMENT 3: Genre vs Avg Streams

# Average streams per genre
genre_avg = (
    df.groupby('primary_genre')['streams']
    .mean()
    .sort_values()  # sorts data categories ascending
)

# Plot bar chart
plt.figure()
genre_avg.plot(kind='bar')

plt.xlabel("Genre")
plt.ylabel("Average Streams (billions)")
plt.title("Average Streams by Genre (Spotify Top 50 Songs 2025)")

plt.xticks(rotation=45)  # rotate labels so they don’t overlap

plt.show()

#----------------------------------------------------------------------------------------------------
# Danceability vs Streams
import numpy as np
import matplotlib.pyplot as plt

# Ensure numeric
df['danceability'] = pd.to_numeric(df['danceability'], errors='coerce')
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')

# Drop missing values
df_clean = df.dropna(subset=['danceability', 'streams'])

# Log transform streams (IMPORTANT for better correlation)
x = df_clean['danceability']
y = np.log1p(df_clean['streams'])

# Scatter plot
plt.figure()
plt.scatter(x, y)

plt.xlabel("Danceability")
plt.ylabel("Log(Streams)")
plt.title("Danceability vs Streams")

# Trendline
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)

# Correlation
corr = np.corrcoef(x, y)[0, 1]
print("Correlation (Danceability vs Log Streams):", corr)

plt.show()
