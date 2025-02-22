"""
Chord Preprocessing & Normalization

This script retrieves chord data from the SQL database, normalizes the chord notations, and updates the cleaned data back into the database.

Main functionalities:
1. **Chord Normalization** - Standardizes chord names by removing unnecessary extensions, correcting typos, and handling slash chords.
2. **Data Cleaning** - Removes invalid chords and replaces missing values.
3. **Database Update** - Saves cleaned chords back to the SQL database.
4. **Chord Analysis** - Analyzes chord distributions, root notes, and common sequences.
5. **CSV Export** - Saves a summary of chord statistics for further analysis.

"""


import pandas as pd
from db_engine import DBEngine
import re
from collections import Counter

# Connect to the database
db = DBEngine()

# Fetch chord data from the database
query = "SELECT id, chord_1, chord_2, chord_3, chord_4 FROM test;"
df = pd.DataFrame(db.execute_sql(query), columns=["id", "chord_1", "chord_2", "chord_3", "chord_4"])

# Store original data for comparison
df_original = df.copy()

# 🔹 Display unique chords before normalization
print("\n🎵 Unique Chords Before Normalization:")
for col in df.columns[1:]:  # Skip 'id' column
    unique_chords = df[col].dropna().unique()
    print(f"🎸 Column: {col} - Unique Count: {len(unique_chords)}")
    print(f"🎵 Unique Chords: {unique_chords}\n")


# 🔹 Chord normalization function
def normalize_chord(chord):
    """Standardizes chord notation by removing unnecessary characters and fixing naming inconsistencies."""
    if chord is None or chord == "0":
        return None  # Keep missing values as None

    # Remove unnecessary extensions
    chord = re.sub(r'\d+', '', chord)
    chord = chord.replace("maj", "").replace("sus", "").replace("add", "").replace("M", "m").replace("min", "m")
    chord = chord.replace("o", "dim").replace("aug", "i").replace("b", "").replace("♯", "#")
    chord = chord.replace("(", "").replace(")", "")

    # Fix specific chord typos
    chord_corrections = {
        "Bd": "B", "Ems": "Em", "Fa": "F", "Amm": "Am", "Dd": "D", "Fis": "E",
        "Dsu": "D", "Bi": "B", "Bim": "B", "Dim": "D", "Fmj": "F", "Famj": "F"
    }
    for key, value in chord_corrections.items():
        chord = chord.replace(key, value)

    # Convert slash chords (e.g., "C/E") to main chord (C)
    if "/" in chord:
        chord = chord.split("/")[0]

    return chord.strip()


# 🔹 Apply normalization
for col in ["chord_1", "chord_2", "chord_3", "chord_4"]:
    df[col] = df[col].apply(normalize_chord)

# Count changed chords
changed_chords = (df != df_original).sum().sum()

# 🔹 Display unique chords after normalization
print("\n🎵 Unique Chords After Normalization:")
for col in df.columns[1:]:  # Skip 'id' column
    unique_chords = sorted(df[col].dropna().unique())
    print(f"🎸 Column: {col} - Unique Count: {len(unique_chords)}")
    print(f"🎵 Unique Chords: {unique_chords}\n")

# 🔹 Replace "0" with NaN for easier counting
df.replace("0", None, inplace=True)

# 🔹 Count valid chords per row
df["num_chords"] = df.notna().sum(axis=1)

# 🔹 Count occurrences of different chord sequences
chord_counts = df["num_chords"].value_counts().sort_index()

# 🔹 Print chord sequence distribution
print("\n📊 Chord Sequence Distribution:")
for num_chords, count in chord_counts.items():
    print(f"🎵 {int(num_chords) - 1} chords: {count} rows")

# 🔹 Batch update normalized chords back to SQL
update_query = """
    UPDATE test 
    SET chord_1 = %s, chord_2 = %s, chord_3 = %s, chord_4 = %s
    WHERE id = %s
"""
update_data = [(row["chord_1"], row["chord_2"], row["chord_3"], row["chord_4"], row["id"]) for _, row in df.iterrows()]

# Execute batch update
try:
    db.execute_batch(update_query, update_data)
    print(f"\n✅ Chord normalization complete! SQL database updated.")
    print(f"🔄 Total chords changed: {changed_chords}")
except Exception as e:
    print(f"❌ Error during batch update: {e}")

# 🔹 Analyze chord frequency across all columns
all_chords = df[["chord_1", "chord_2", "chord_3", "chord_4"]].values.flatten()
chord_counts = Counter(all_chords)

# 🔹 Define main chord roots (A-G)
root_notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
root_chord_counts = {root: 0 for root in root_notes}

# Count chords grouped by root note
for chord, count in chord_counts.items():
    if chord:
        root_match = re.match(r"^[A-G]#?", chord)
        if root_match:
            root_chord_counts[root_match.group()] += count

# 🔹 Print occurrences per root note
print("\n🎵 Chord Frequency by Root Note:")
for root, count in root_chord_counts.items():
    print(f"🎸 {root}: {count} occurrences")

# 🔹 Count chords per column
column_chord_counts = {col: Counter(df[col].dropna()) for col in ["chord_1", "chord_2", "chord_3", "chord_4"]}

# 🔹 Save final statistics to CSV
summary_df = pd.DataFrame(column_chord_counts).fillna(0).astype(int)
summary_df.to_csv("chord_statistics.csv", index=True)
print("\n📊 Chord statistics saved to 'chord_statistics.csv'.")

# 🔹 Print chord frequency per column
print("\n📊 Chord Frequency Per Column:")
for col, counts in column_chord_counts.items():
    print(f"\n🎸 {col}:")
    for chord, count in counts.most_common():
        print(f"  {chord}: {count} times")

