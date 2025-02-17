import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
import numpy as np
from db_engine import DBEngine

# 🔹 Ensure required NLTK packages are downloaded
nltk.download("punkt_tab")
nltk.download("punkt")

# 🔹 Load Pretrained FastText (300D English)
print("📥 Loading FastText Model (this may take a while)...")
fasttext_model = KeyedVectors.load_word2vec_format("cc.en.300.vec", binary=False)
print("✅ FastText Model Loaded!")

# 🔹 Connect to database & load lyrics
db = DBEngine()
query = "SELECT id, lyrics FROM test;"
df = pd.DataFrame(db.execute_sql(query), columns=["id", "lyrics"])


# 🔹 Text Cleaning & Tokenization
def clean_and_tokenize(text):
    if not isinstance(text, str):
        return []

    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    return word_tokenize(text)  # Tokenize into words


df["tokens"] = df["lyrics"].apply(clean_and_tokenize)


# 🔹 Convert Each Lyric Line into an Average Vector
def lyrics_to_vector(tokens, model):
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)  # 300D vector


df["vector"] = df["tokens"].apply(lambda tokens: lyrics_to_vector(tokens, fasttext_model))

# 🔹 Convert to DataFrame & Save
embedding_df = pd.DataFrame(df["vector"].tolist())
embedding_df.to_csv("lyrics_fasttext.csv", index=False)

print("✅ Lyrics Vectorization with FastText Complete! Vectors saved to lyrics_fasttext.csv.")

print("\n✅ Sample Vectorized Lyrics:")
for i in range(5):  # Show first 5 rows
    print(f"📝 Lyrics ID {df.iloc[i]['id']}: {df.iloc[i]['lyrics']}")
    print(f"🎸 Vector: {df.iloc[i]['vector'][:10]} ... (showing first 10 values)\n")