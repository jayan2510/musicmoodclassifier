# Mood-Based Music Classifier

A machine learning-powered web app that classifies the **mood of a song** based on its lyrics using a fine-tuned BERT model. Built with Python, Streamlit, and HuggingFace Transformers.

---

## Features

- Input any song title and artist name
- Automatically fetch lyrics using Genius API
- Classify mood into: `happy`, `sad`, `romantic`, `calm`, or `energetic`
- Fine-tuned BERT model for mood classification
- Logs all predictions into a CSV file
- Simple Streamlit UI for testing and interaction

---

##  Tech Stack
- Python 
- HuggingFace Transformers 
- BERT (Fine-tuned)
- Streamlit 
- Pandas, Scikit-learn, Datasets
- Genius API (for lyrics)

---

## 🛠 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/jayan2510/musicmoodclassifier.git
cd musicmoodclassifier
