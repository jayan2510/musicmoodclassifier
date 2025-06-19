import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import os

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Paths
input_path = "../data/mood_lyrics_dataset.csv"
output_path = "../data/mood_lyrics_cleaned.csv"

# Load dataset
df = pd.read_csv(input_path)

# Initialize tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_lyrics(text):
    if pd.isna(text):
        return ""

    # Lowercase
    text = text.lower()
    # Remove digits and punctuation
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    # Tokenize and remove stopwords + lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply cleaning
df["cleaned_lyrics"] = df["lyrics"].apply(clean_lyrics)

# Drop original lyrics if you want (optional)
# df = df.drop(columns=["lyrics"])

# Save cleaned dataset
df.to_csv(output_path, index=False)
print(f"✅ Cleaned data saved to: {output_path}")
