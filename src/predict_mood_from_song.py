import torch 
from transformers import BertTokenizer, BertForSequenceClassification
from .fetch_lyrics import get_song_lyrics
import pandas as pd 
import csv
from datetime import datetime
import os 

## load model and tokenizer 
model_path = os.path.join(os.path.dirname(__file__), "../models/mood_classifier_bert")
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

## label mappinig ensure matching with training 
label_mapping= {
    0: "calm",
    1: "energetic",
    2: "happy",
    3: "romantic",
    4: "sad"
}

def predict_mood(lyrics):
    inputs = tokenizer(lyrics, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predict_mood = torch.argmax(outputs.logits,dim=1).item()
    return label_mapping[predict_mood]

def suggest_songs_by_mood(mood,csv_path="../data/seed_data.csv",top_n=3):
    try:
        df = pd.read_csv(csv_path)
        suggestions = df[df["mood"]==mood].sample(n = top_n, replace = True)
        return suggestions[["songs","artist"]].values.tolist()
    except Exception as e:
        return[]

def log_prediction(song_title, artist_name, predicted_mood, log_file="../data/prediction_log.csv"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_exists = os.path.exists(log_file)
    
    with open(log_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["song_title", "artist_name", "predicted_mood"])
        writer.writerow([song_title, artist_name, predicted_mood])

if __name__ == "__main__":
    song = input("Enter song title: ")
    artist = input("Enter artist name: ")

    lyrics = get_song_lyrics(song, artist)

    if lyrics:
        print("\nLyrics fetched successfully!\n")
        print(lyrics[:300], "...\n")  # preview first 300 characters
        mood = predict_mood(lyrics)
        print(f"Predicted Mood: {mood}")

        log_prediction(song,artist,mood)

        suggestions = suggest_songs_by_mood(mood)
        if suggestions:
            print("\n you might also like")
            for s, a in suggestions:
                print(f"-{s} by {a}")
    else:
        print("Could not find lyrics for that song.")
    
    