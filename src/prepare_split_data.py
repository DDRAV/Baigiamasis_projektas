import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from db_engine import DBEngine  # Import your DB connection class

# 🔹 Load TF-IDF Vectorized Lyrics
tfidf_df = pd.read_csv("lyrics_tfidf_cleaned.csv")

# 🔹 Connect to Database & Load Chord Labels
db = DBEngine()
query = "SELECT id, chord_1, chord_2, chord_3, chord_4 FROM test;"
chords_df = pd.DataFrame(db.execute_sql(query), columns=["id", "chord_1", "chord_2", "chord_3", "chord_4"])

# 🔹 Ensure Data Alignment (TF-IDF features should match chords)
assert len(tfidf_df) == len(chords_df), "Mismatch between TF-IDF and chord dataset sizes!"

# 🔹 Encode Chords Using Label Encoding
label_encoders = {}
for col in ["chord_1", "chord_2", "chord_3", "chord_4"]:
    label_encoders[col] = LabelEncoder()
    chords_df[col] = label_encoders[col].fit_transform(chords_df[col].astype(str))  # Convert chords to numerical labels

# 🔹 Combine Features (X) and Labels (Y)
X = tfidf_df.values  # TF-IDF Features
Y = chords_df[["chord_1", "chord_2", "chord_3", "chord_4"]].values  # Chord labels

# 🔹 Split into Training & Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 🔹 Display Sample Data for Verification
print("✅ **Training Data Sample (First 5 Rows)**")
print(pd.DataFrame(X_train[:5], columns=tfidf_df.columns))

print("\n🎵 **Chord Labels Sample (First 5 Rows - Encoded)**")
print(pd.DataFrame(Y_train[:5], columns=["chord_1", "chord_2", "chord_3", "chord_4"]))

# 🔹 Display Unique Encoded Chords for Reference
for col in ["chord_1", "chord_2", "chord_3", "chord_4"]:
    unique_chords = list(label_encoders[col].classes_)
    print(f"\n🎸 **Chord Mapping for {col}:**")
    print({i: chord for i, chord in enumerate(unique_chords)})

print("\n✅ Data preparation complete! Check the output above to verify.")
