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
BASE_URL = "https://www.songsterr.com/tags/guitar"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Selenium setup for browser automation
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run headless if you don't need the browser GUI
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
driver = webdriver.Chrome(options=chrome_options)

def clean_lyrics(lyrics):
    """Cleans the lyrics by removing unwanted characters, fixing spacing, and filtering numbers."""
    lyrics = re.sub(r"[^\w\s,.]", "", lyrics)  # Remove special characters except spaces, commas, and periods
    lyrics = re.sub(r"\s+", " ", lyrics).strip()  # Normalize spacing
    lyrics = lyrics.replace('_', '')  # Remove underscores

    # Remove lines that contain numbers
    if re.search(r"\d", lyrics):
        return ""

    # Remove repeated letters (e.g., "aaa")
    if re.search(r"(.)\1{2,}", lyrics):
        return ""

    # Remove lyrics with only repeated words (e.g., "la la la la la")
    words = lyrics.split()
    unique_words = set(words)
    if len(unique_words) == 1:  # All words are the same
        return ""

    # Remove unwanted words
    if any(word in lyrics.lower() for word in ["unknown", "nonsense", "na", "na-na"]):
        return ""

    return lyrics

def is_duplicate(title, lyrics, chords):
    """Checks if the song entry already exists in the database."""
    query = "SELECT COUNT(*) FROM test WHERE title = %s AND lyrics = %s AND chord_1 = %s AND chord_2 = %s AND chord_3 = %s AND chord_4 = %s;"
    result = db.execute_sql(query, (title, lyrics, chords[0] if chords else "0"))
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

    # Remove "Chords" from the end of the title if it exists
    title = re.sub(r"Chords$", "", title).strip()

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
        chord_list = [chord.get_text(strip=True) for chord in chords]

        # **NEW FILTER: Skip if chords are at least 2x more than lyrics**
        if len(chord_list) >= 2 * len(line_lyrics.split()):
            print(f"⚠️ Skipping line with too many chords: '{line_lyrics}'")
            continue

        # **NEW FILTER: Skip if there are more than 6 chords**
        if len(chord_list) > 4:
            print(f"⚠️ Skipping line with more than 4 chords: '{chord_list}'")
            continue

        # Pad chords to ensure we have exactly 4 columns
        chord_list += ["0"] * (4 - len(chord_list))

        # Check if this entry already exists
        if is_duplicate(title, line_lyrics, chord_list):
            print(f"⚠️ Duplicate entry found, skipping: {title} - {artist}")
            continue

        # Store in DB if valid
        if line_lyrics:
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

# Main execution: Scrape all pages sequentially
#for page in range(1, 53):  # Loop through pages 1 to 50
scrape_page(BASE_URL)

# Cleanup
driver.quit()
print("🎶 Scraping completed!")