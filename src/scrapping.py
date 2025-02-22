"""
Songsterr Scraper

This script scrapes song lyrics and chords from Songsterr and stores them in a PostgreSQL database.
- Uses Selenium for automated browsing.
- Uses BeautifulSoup for HTML parsing.
- Cleans lyrics by removing invalid characters and duplicates.
- Ensures only valid 4-chord sequences are stored.
- Avoids inserting duplicate songs into the database.

Requirements:
- Install dependencies: `pip install selenium beautifulsoup4 psycopg2 requests`
- Ensure PostgreSQL database is running and `db_engine.py` is configured.
"""

import requests
import re
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from db_engine import DBEngine

# Initialize the database connection
db = DBEngine()

# Base URL for pagination
BASE_URL = "https://www.songsterr.com/tags/guitar?page={}"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Selenium setup for headless browser automation
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
driver = webdriver.Chrome(options=chrome_options)


def clean_lyrics(lyrics):
    """Cleans and filters lyrics by removing unwanted characters, numbers, and invalid formats."""
    lyrics = re.sub(r"[^\w\s,.]", "", lyrics)  # Remove special characters except spaces, commas, and periods
    lyrics = re.sub(r"\s+", " ", lyrics).strip()  # Normalize spacing
    lyrics = lyrics.replace('_', '')  # Remove underscores

    if re.search(r"\d", lyrics):  # Remove lyrics containing numbers
        return ""
    if re.search(r"(.)\1{2,}", lyrics):  # Remove repeated letters (e.g., "aaa")
        return ""
    words = lyrics.split()
    if len(set(words)) == 1:  # Remove lyrics with repeated single words
        return ""
    if any(word in lyrics.lower() for word in ["unknown", "nonsense", "na", "na-na"]):
        return ""

    return lyrics


def is_duplicate(title, lyrics, chords):
    """Checks if a song entry already exists in the database."""
    query = """
        SELECT COUNT(*) FROM test 
        WHERE title = %s AND lyrics = %s AND chord_1 = %s AND chord_2 = %s AND chord_3 = %s AND chord_4 = %s;
    """
    result = db.execute_sql(query, (title, lyrics, *chords))
    return result and result[0][0] > 0  # Returns True if the entry already exists


def scrape_page(page):
    """Scrapes a single page for song URLs and processes them."""
    url = BASE_URL.format(page)
    print(f"🔄 Scraping page {page}: {url}")

    driver.get(url)
    time.sleep(2)  # Wait for page load

    # Extract all song links from the page
    song_links = driver.find_elements(By.CSS_SELECTOR, ".Bw9243 a")
    chord_links = [link.get_attribute("href") for link in song_links if "-chords-" in link.get_attribute("href")]

    if not chord_links:
        print("✅ No more chord links found. Stopping.")
        return False  # Stop when no new links are found

    print(f"✅ Found {len(chord_links)} chord links on page {page}")

    # Process each song URL
    for song_url in chord_links:
        process_song(song_url)

    return True  # Continue to the next page


def process_song(song_url):
    """Scrapes lyrics, chords, title, and artist from a song page and saves them to the database."""
    print(f"🎸 Scraping song: {song_url}")

    response = requests.get(song_url, headers=HEADERS)
    if response.status_code != 200:
        print(f"❌ Error fetching song page: {song_url}")
        return

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title and artist
    title = soup.select_one("span.C612ph").get_text(strip=True) if soup.select_one("span.C612ph") else "Unknown"
    artist = soup.select_one("a.C61rs").get_text(strip=True) if soup.select_one("a.C61rs") else "Unknown"
    title = re.sub(r"Chords$", "", title).strip()  # Clean title
    print(f"🎵 Title: {title} | 🎤 Artist: {artist}")

    lines = soup.find_all("p", class_="C5zdi")
    line_count = 0

    for line in lines:
        words = line.find_all("span", class_="C5zy8")
        chords = line.find_all("span", class_="Bsiek")
        if not words:
            continue

        line_lyrics = " ".join([word.get_text() for word in words]).strip()
        if len(line_lyrics.split()) < 2:
            continue  # Ignore very short lines

        line_lyrics = clean_lyrics(line_lyrics)
        if not line_lyrics:
            continue

        chord_list = [chord.get_text(strip=True) for chord in chords]

        # Filtering out lines with too many chords or bad structure
        if len(chord_list) >= 2 * len(line_lyrics.split()):
            print(f"⚠️ Skipping line with too many chords: '{line_lyrics}'")
            continue
        if len(chord_list) > 4:
            print(f"⚠️ Skipping line with more than 4 chords: '{chord_list}'")
            continue

        chord_list += ["0"] * (4 - len(chord_list))  # Ensure 4 chord slots

        # Check for duplicates before inserting
        if is_duplicate(title, line_lyrics, chord_list):
            print(f"⚠️ Duplicate entry found, skipping: {title} - {artist}")
            continue

        try:
            query = """
                INSERT INTO test (title, artist, lyrics, chord_1, chord_2, chord_3, chord_4)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            db.execute_sql(query, (title, artist, line_lyrics, *chord_list))
            line_count += 1
        except Exception as e:
            print(f"⚠️ Error inserting data for {song_url}: {e}")

    if line_count > 0:
        print(f"✅ {line_count} lines saved for song '{title}' - '{artist}'")


# Main execution: Scrape multiple pages sequentially
for page in range(1, 53):  # Scrape from page 1 to 52
    if not scrape_page(page):
        break  # Stop when no more data is found

# Cleanup
driver.quit()
db.disconnect()
print("🎶 Scraping completed!")

"""
The scraped data will later be used to train machine learning models (Random Forest, LSTM, Transformer).
- Lyrics are converted to numerical embeddings.
- Chords are assigned to each lyrics line.
"""