import pandas as pd
import re
import nltk
import enchant
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from db_engine import DBEngine
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Ensure required NLTK packages are downloaded
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Load English dictionary for word filtering
english_dict = enchant.Dict("en_US")

# Extended stopword list
custom_stopwords = {
    "yeah", "oh", "ooh", "aah", "mmm", "uh", "huh", "aha", "hey", "ho", "ha",
    "whoa", "woah", "doo", "doo-wop", "sha", "shoo", "bam", "dam", "mmm", "hmm",
    "na", "nanana", "la", "lalala", "dumdum", "babababa", "wop", "ahh", "wah", "yi",
    "ow", "ouch", "yipee", "whoops", "yay", "whew", "whaa", "whee", "tsk", "grr", "huhh",
    "uhhuh", "yea", "ok", "woop", "yo", "damn", "dang", "boo", "al", "ala", "ah", "ai",
    "av", "aw", "ca", "col", "dado", "dc", "db", "dy", "dun", "er", "erst", "ebb",
    "et", "est", "fa", "gi", "ii", "ja", "jo", "lo", "ma", "mi", "mo", "mi", "min", "mu",
    "moi", "mot", "mon", "monde", "ob", "och", "om", "oo", "op", "oft", "ne", "pa", "pe", "sou",
    "thy", "ti", "va", "wo", "ya", "ye", "yah", "dm", "em", "el"

}

# Connect to database
db = DBEngine()

# Load lyrics from SQL
query = "SELECT id, lyrics FROM test;"
df = pd.DataFrame(db.execute_sql(query), columns=["id", "lyrics"])

# 🔹 Preprocessing Function
def clean_lyrics(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = " ".join([
        lemmatizer.lemmatize(word) for word in text.split()
        if word not in stop_words and word not in custom_stopwords and english_dict.check(word)
    ])  # Remove stopwords, custom words, and non-English words

    return text.strip()

# 🔹 Apply preprocessing
df["clean_lyrics"] = df["lyrics"].apply(clean_lyrics)

# 🔹 Word Frequency Filtering (Keep words appearing at least in 5 songs)
all_words = " ".join(df["clean_lyrics"]).split()
word_counts = Counter(all_words)
filtered_words = {word for word, count in word_counts.items() if count >= 2}

# Apply frequency filtering
df["filtered_lyrics"] = df["clean_lyrics"].apply(lambda text: " ".join([word for word in text.split() if word in filtered_words]))

# 🔹 Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=100000, stop_words="english")

# 🔹 Fit and transform the filtered lyrics
tfidf_matrix = vectorizer.fit_transform(df["filtered_lyrics"])

# 🔹 Convert to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# 🔹 Save TF-IDF features to a CSV file
tfidf_df.to_csv("lyrics_tfidf_cleaned.csv", index=False)

print("✅ Lyrics preprocessing & vectorization complete! Cleaned TF-IDF saved.")
