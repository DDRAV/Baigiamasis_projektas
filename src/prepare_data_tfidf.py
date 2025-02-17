import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from db_engine import DBEngine  # Import your DB connection class

# 🔹 Load FastText Vectorized Lyrics
fasttext_df = pd.read_csv("lyrics_fasttext.csv")

# 🔹 Connect to Database & Load Chord Labels
db = DBEngine()
query = "SELECT id, chord_1, chord_2, chord_3, chord_4 FROM test;"
chords_df = pd.DataFrame(db.execute_sql(query), columns=["id", "chord_1", "chord_2", "chord_3", "chord_4"])

# 🔹 Ensure Data Alignment (FastText features should match chords)
assert len(fasttext_df) == len(chords_df), "Mismatch between FastText and chord dataset sizes!"

# 🔹 Encode Chords Using Ordinal Encoding
encoder = OrdinalEncoder()
Y = encoder.fit_transform(chords_df[["chord_1", "chord_2", "chord_3", "chord_4"]])

# 🔹 Convert Features (X) and Labels (Y) into Numpy Arrays
X = fasttext_df.values  # FastText Features

# 🔹 Normalize FastText Embeddings to Standardize Input Data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🔹 Split into Training & Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 🔹 Display Sample Data for Verification
print("✅ **Training Data Sample (First 5 Rows)**")
print(pd.DataFrame(X_train[:5]))

print("\n🎵 **Chord Labels Sample (First 5 Rows - Encoded)**")
print(pd.DataFrame(Y_train[:5], columns=["chord_1", "chord_2", "chord_3", "chord_4"]))

# 🔹 Display Unique Encoded Chords for Reference
for i, chord in enumerate(encoder.categories_):
    print(f"\n🎸 **Chord Mapping for chord_{i+1}:**")
    print({j: chord[j] for j in range(len(chord))})

print("\n✅ Data preparation complete! Check the output above to verify.")
