# app.py

import streamlit as st
from src.fetch_lyrics import get_song_lyrics
from src import predict_mood_from_song as pms

st.set_page_config(page_title="Mood Classifier", page_icon="🎵")
st.title("🎧 Mood-Based Song Classifier")

st.markdown("Enter a song title and artist to analyze its mood using lyrics.")

song = st.text_input("🎼 Song Title", placeholder="e.g. Tum Hi Ho")
artist = st.text_input("🎤 Artist Name", placeholder="e.g. Arijit Singh")

if st.button("🔍 Analyze"):
    if song.strip() == "" or artist.strip() == "":
        st.warning("Please enter both song and artist.")
    else:
        with st.spinner("Fetching lyrics and analyzing..."):
            lyrics = pms.get_song_lyrics(song, artist)

            if lyrics:
                st.subheader("📜 Lyrics Preview")
                st.text(lyrics[:500] + ("..." if len(lyrics) > 500 else ""))

                mood = pms.predict_mood(lyrics)
                st.success(f"🧠 **Predicted Mood: {mood}**")

                pms.log_prediction(song, artist, mood)

                suggestions = pms.suggest_songs_by_mood(mood)
                if suggestions:
                    st.subheader("🎶 You might also like:")
                    for s, a in suggestions:
                        st.markdown(f"- **{s}** by *{a}*")
            else:
                st.error("❌ Could not find lyrics for this song.")
