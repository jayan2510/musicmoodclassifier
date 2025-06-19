import pandas as pd 
import time 
import os 
from dotenv import load_dotenv

from fetch_lyrics import get_song_lyrics

## load env variables 
load_dotenv()

## file paths 
input_path = "../data/seed_data.csv"
output_path = "../data/mood_lyrics_dataset.csv"

## loading seed data 
df = pd.read_csv(input_path)

## add a lyrics col 
df["lyrics"] = ""

for idx, row in df.iterrows():
    title = row["title"]
    artist = row["artist"]
    print(f"Fetching lyrics: {title} by {artist} ...")


    try:
        lyrics = get_song_lyrics(title,artist)
        df.at[idx, "lyrics"] = lyrics if lyrics else ""
    except Exception as e:
        print(f"failed to fetch for {title} by {artist}: {e}")
        df.at[idx, "lyrics"] = ""

    time.sleep(2) # prevent hittling genius rate limits 

# Drop any rows where lyrics couldn’t be fetched
df = df[df["lyrics"].str.strip() != ""]

# Save the final dataset
df.to_csv(output_path, index=False)
print(f"\n Dataset created: {output_path}")