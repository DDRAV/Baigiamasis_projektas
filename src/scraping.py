import requests
import re
from bs4 import BeautifulSoup
from db_engine import DBEngine

# Initialize the database connection
db = DBEngine()

# URL of the main page with songs list
url = "https://www.songsterr.com"

# Headers to mimic a real browser
headers = {"User-Agent": "Mozilla/5.0"}

# Fetch the main page content
response = requests.get(url, headers=headers)
if response.status_code != 200:
    print("Error fetching the page:", response.status_code)
    exit()

# Parse the HTML
soup = BeautifulSoup(response.text, "html.parser")

# Find all song links
song_links = soup.find_all("a", class_="B0cew B0c14s")

# Limit the number of songs to 10
song_links = song_links[:1000]


# Function to clean lyrics (remove unwanted punctuation and sequences)
def clean_lyrics(lyrics):
    # Remove all punctuation except for dot (.) and comma (,) and underscores (_)
    lyrics = re.sub(r"[^\w\s,.]", "", lyrics)  # Remove everything except word characters, spaces, commas, and periods

    # Remove underscores (_)
    lyrics = lyrics.replace('_', '')

    # Check if the lyrics contain any sequence of the same letters or unknown words
    if re.search(r"(.)\1{2,}", lyrics):  # Repeated characters (e.g. "aaa")
        return ""
    if any(word in lyrics.lower() for word in ["unknown", "nonsense", "na", "na-na"]):  # Example of unknown words
        return ""

    return lyrics


# Initialize counter for the number of successful songs
successful_songs = 0

# Loop through each song link to scrape its details
for link in song_links:
    song_url = "https://www.songsterr.com" + link['href']

    # Replace '-tab-' with '-chords-' in the URL
    song_url = song_url.replace("-tab-", "-chords-")

    song_name = link.find("div", class_="B0c2e8").get_text(strip=True)

    # Fetch the song's page content
    song_response = requests.get(song_url, headers=headers)
    if song_response.status_code != 200:
        print(f"Error fetching the song page: {song_name}")
        continue

    # Parse the song page HTML
    song_soup = BeautifulSoup(song_response.text, "html.parser")

    # Find all elements containing both chords and lyrics
    lines = song_soup.find_all("p", class_="C5zdi")

    # Initialize line counter for each song
    line_count = 0

    # Loop through each line and extract lyrics + chords
    for line in lines:
        words = line.find_all("span", class_="C5zy8")  # Lyrics
        chords = line.find_all("span", class_="Bsiek")  # Chords

        # If no lyrics found, skip this line
        if not words:
            continue

        # Extract lyrics with proper spacing
        line_lyrics = "".join([word.get_text() for word in words]).strip()
        line_lyrics = " ".join(line_lyrics.split())

        # Skip line if there are less than 2 words in the lyrics
        if len(line_lyrics.split()) < 2:
            continue

        # Clean the lyrics by removing unnecessary characters
        line_lyrics = clean_lyrics(line_lyrics)
        if not line_lyrics:
            continue

        # Check if chords exist and ensure that they have a corresponding lyric line
        line_chords = ""
        if chords and words:
            line_chords = ">".join([chord.get_text(strip=True) for chord in chords])

        # If chords are found but no corresponding lyrics, skip this line
        if not words and chords:
            continue

        # If chords count is more than double the words count, skip this line
        if len(line_chords.split(">")) > 2 * len(line_lyrics.split()):
            continue

        # Insert into database only if there's valid lyrics and chords
        if line_lyrics and line_chords:
            query = "INSERT INTO test (lyrics, chords) VALUES (%s, %s);"
            db.execute_sql(query, (line_lyrics, line_chords))
            line_count += 1

    # Print out the count of lines inserted for the current song
    if line_count > 0:
        print(f"✅ {line_count} lines for song '{song_name}' successfully inserted into PostgreSQL!")

    # If at least one line was inserted, increment successful songs counter
    if line_count > 0:
        successful_songs += 1

# Print the final count of songs with at least one line inserted
print(f"✅ {successful_songs} songs had at least one line successfully inserted into PostgreSQL.")
