import torch 
from transformers import BertTokenizer, BertForSequenceClassification
import sys 
import torch.nn.functional as F
import pandas as pd 

## load model and tokenizer 
model_path = "../models/mood_classifier_bert"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval() ## set to inference mode 

## label mapping (should match your training )
label_mapping = {
    0: "calm",
    1: "energetic",
    2: "happy",
    3: "romantic",
    4: "sad"
}

def predict_mood(lyrics):
    #tokenize lyrics
    inputs = tokenizer(lyrics,return_tensors="pt",truncation=True,padding=True,max_length=512)

    # get prediction 
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    print("\n🎯 Prediction Probabilities:")
    for idx, score in enumerate(probs):
        print(f"{label_mapping[idx]:<10}: {score.item():.4f}")

    return label_mapping[predicted_class]
    
    
# example 
if __name__ == "__main__":
    lyrics = input("Paste the song lyrics:\n")
    mood = predict_mood(lyrics)
    print(f"\n🎵 Predicted Mood: {mood}")