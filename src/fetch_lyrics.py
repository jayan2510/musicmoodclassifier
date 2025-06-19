import requests
from bs4 import BeautifulSoup
import os 
from dotenv import load_dotenv

load_dotenv()

GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN")

def search_song_on_genius(song_title, artist_name):
    base_url = "https://api.genius.com"
    headers = {"Authorization":f"Bearer {GENIUS_API_TOKEN}"}
    search_url = f"{base_url}/search?q={song_title} {artist_name}" 

    response = requests.get(search_url, headers = headers)
    json_data = response.json()

    hits = json_data["response"]["hits"]
    if hits:
        for hit in hits:
            if artist_name.lower() in hit["result"]["primary_artist"]["name"].lower():
                song_url = hit["result"]["url"]
                return song_url
    return None

def scrape_lyrics_from_url(song_url):
    page = requests.get(song_url)
    soup = BeautifulSoup(page.text, "html.parser")

    # The lyrics are stored inside <div> with data-lyrics-container
    lyrics_divs = soup.find_all("div", attrs={"data-lyrics-container": "true"})
    lyrics = "\n".join([div.get_text(separator="\n") for div in lyrics_divs])

    return lyrics.strip()


def get_song_lyrics(song_title, artist_name):
    song_url = search_song_on_genius(song_title,artist_name)
    if song_url:
        return scrape_lyrics_from_url(song_url)
    else:
        return None

## example usage 
if __name__ == "__main__":
    song = "O Saathi"
    artist = "Atif Aslam"

    lyrics = get_song_lyrics(song, artist)
    if lyrics:
        print(f"Lyrics for '{song}' by {artist}:\n")
        print(lyrics)
    else:
        print("Lyrics not found.")
