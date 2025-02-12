import requests
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from db_engine import DBEngine
import time

# Initialize the database connection
db = DBEngine()

# URL of the main page with songs list
url = "https://www.songsterr.com"

# Headers to mimic a real browser
headers = {"User-Agent": "Mozilla/5.0"}

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

# Selenium setup for browser automation
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run headless if you don't need the browser GUI
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
driver = webdriver.Chrome(options=chrome_options)

# Open the main URL using Selenium
driver.get(url)
time.sleep(2)  # Wait for the page to load

# Infinite scroll (or you can find all the song links after scrolling)
while True:
    # Find all song links
    song_links = driver.find_elements(By.CLASS_NAME, "B0cew")  # Adjust class name if needed
    if len(song_links) == 0:
        break  # Stop if no more songs are found

    # Get all song URLs
    for link in song_links:
        song_url = link.get_attribute("href")
        # Replace '-tab-' with '-chords-' in the URL to get the correct page
        song_url = song_url.replace("-tab-", "-chords-")
        print(f"Found song URL: {song_url}")

        # Fetch the song's page content using requests
        song_response = requests.get(song_url, headers=headers)
        if song_response.status_code != 200:
            print(f"Error fetching the song page: {song_url}")
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
                try:
                    query = "INSERT INTO test (lyrics, chords) VALUES (%s, %s);"
                    db.execute_sql(query, (line_lyrics, line_chords))
                    line_count += 1
                except Exception as e:
                    print(f"Error while executing query for song '{song_url}': {e}")

        # Print out the count of lines inserted for the current song
        if line_count > 0:
            print(f"✅ {line_count} lines for song '{song_url}' successfully inserted into PostgreSQL!")

# Final report on the total number of songs with at least one line inserted
successful_songs = len([song_url for song_url in song_links if line_count > 0])
print(f"✅ {successful_songs} songs had at least one line successfully inserted into PostgreSQL.")

# Close the WebDriver
driver.quit()


