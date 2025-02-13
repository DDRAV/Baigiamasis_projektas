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

# Base URL for pagination
BASE_URL = "https://www.songsterr.com/tags/guitar?page={}"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Selenium setup for browser automation
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run headless if you don't need the browser GUI
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
driver = webdriver.Chrome(options=chrome_options)

def clean_lyrics(lyrics):
    """Cleans the lyrics by removing unwanted characters and sequences."""
    lyrics = re.sub(r"[^\w\s,.]", "", lyrics)  # Remove special characters
    lyrics = lyrics.replace('_', '')  # Remove underscores
    if re.search(r"(.)\1{2,}", lyrics):  # Remove repeated letters (e.g., "aaa")
        return ""
    if any(word in lyrics.lower() for word in ["unknown", "nonsense", "na", "na-na"]):
        return ""
    return lyrics

def is_duplicate(title, lyrics, chords):
    """Checks if the song entry already exists in the database."""
    query = "SELECT COUNT(*) FROM test WHERE title = %s AND lyrics = %s AND chords = %s;"
    result = db.execute_sql(query, (title, lyrics, chords))
    return result and result[0][0] > 0  # If count > 0, it means entry exists

def scrape_page(page):
    """Scrapes a single page for song URLs and processes them."""
    url = BASE_URL.format(page)
    print(f"🔄 Scraping page {page}: {url}")

    # Fetch the page
    driver.get(url)
    time.sleep(2)  # Wait for page load

    # Find all song links inside the class Bw9243
    song_links = driver.find_elements(By.CSS_SELECTOR, ".Bw9243 a")

    # Extract URLs and filter only those with "-chords-"
    chord_links = [link.get_attribute("href")
        for link in song_links if "-chords-" in link.get_attribute("href")
    ]

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

    # Fetch the song page content
    response = requests.get(song_url, headers=HEADERS)
    if response.status_code != 200:
        print(f"❌ Error fetching song page: {song_url}")
        return

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title & artist
    title_tag = soup.select_one("span.C612ph")
    artist_tag = soup.select_one("a.C61rs")

    title = title_tag.get_text(strip=True) if title_tag else "Unknown"
    artist = artist_tag.get_text(strip=True) if artist_tag else "Unknown"

    print(f"🎵 Title: {title} | 🎤 Artist: {artist}")

    lines = soup.find_all("p", class_="C5zdi")  # Extract lyrics & chords

    line_count = 0
    for line in lines:
        words = line.find_all("span", class_="C5zy8")  # Lyrics
        chords = line.find_all("span", class_="Bsiek")  # Chords

        if not words:
            continue

        # Extract lyrics
        line_lyrics = "".join([word.get_text() for word in words]).strip()
        line_lyrics = " ".join(line_lyrics.split())
        if len(line_lyrics.split()) < 2:
            continue  # Ignore very short lines

        line_lyrics = clean_lyrics(line_lyrics)
        if not line_lyrics:
            continue

        # Extract chords
        line_chords = ">".join([chord.get_text(strip=True) for chord in chords]) if chords else ""

        # Check if this entry already exists
        if is_duplicate(title, line_lyrics, line_chords):
            print(f"⚠️ Duplicate entry found, skipping: {title} - {artist}")
            continue

        # Store in DB if valid
        if line_lyrics and line_chords:
            try:
                db.execute_sql(
                    "INSERT INTO test (title, artist, lyrics, chords) VALUES (%s, %s, %s, %s);",
                    (title, artist, line_lyrics, line_chords)
                )
                line_count += 1
            except Exception as e:
                print(f"⚠️ Error inserting data for {song_url}: {e}")

    if line_count > 0:
        print(f"✅ {line_count} lines saved for song '{title}' - '{artist}'")

# Main execution: Scrape all pages sequentially
page = 1
while scrape_page(page):
    page += 1  # Move to the next page

# Cleanup
driver.quit()
print("🎶 Scraping completed!")
