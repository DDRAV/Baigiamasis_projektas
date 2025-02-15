import pandas as pd
from db_engine import DBEngine
import re
from collections import Counter

# Prisijungiame prie duomenų bazės
db = DBEngine()

# SQL užklausa duomenims nuskaityti
query = "SELECT id, chord_1, chord_2, chord_3, chord_4 FROM test;"
df = pd.DataFrame(db.execute_sql(query), columns=["id", "chord_1", "chord_2", "chord_3", "chord_4"])

# Kopijuojame originalius duomenis, kad galėtume palyginti pokyčius
df_original = df.copy()

# 🔹 Print unique chords before normalization
print("🎵 **UNIQUE CHORDS BEFORE NORMALIZATION:**")
for col in df.columns[1:]:  # Skip 'id' column
    unique_chords = df[col].dropna().unique()
    print(f"🎸 Stulpelis: {col}")
    print(f"🔹 Unikalių akordų skaičius: {len(unique_chords)}")
    print(f"🎵 Unikalūs akordai: {unique_chords}\n")

# 🔹 Chord normalization function
def normalize_chord(chord):
    if chord is None or chord == "0":
        return None  # Missing values remain None

    # Remove unnecessary extensions
    chord = re.sub(r'\d+', '', chord)
    chord = chord.replace("maj", "").replace("sus", "").replace("add", "").replace("M", "m").replace("min", "m")
    chord = chord.replace("o", "dim").replace("suus", "").replace("ii", "i").replace("aug", "i").replace("b", "")
    chord = chord.replace("(", "").replace(")", "").replace("♯", "#").replace("Bd", "B").replace("Ems", "Em")
    chord = chord.replace("Fa", "F").replace("Amm", "Am").replace("Dd", "D").replace("Fmj", "F").replace("Famj", "F")
    chord = chord.replace("Fis", "E").replace("Dsu", "D").replace("Bi", "B").replace("Bim", "B").replace("Dim", "D")
    # Convert slash chords to main chord (e.g., "C/E" -> "C")
    if "/" in chord:
        chord = chord.split("/")[0]

    return chord.strip()


# 🔹 Apply normalization
for col in ["chord_1", "chord_2", "chord_3", "chord_4"]:
    df[col] = df[col].apply(normalize_chord)

# 🔹 Count changed chords
changed_chords = (df != df_original).sum().sum()

# 🔹 Print unique chords after normalization
print("🎵 **UNIQUE CHORDS AFTER NORMALIZATION:**")
for col in df.columns[1:]:  # Skip 'id' column
    unique_chords = df[col].dropna().unique()
    unique_chords = sorted(unique_chords)
    print(f"🎸 Stulpelis: {col}")
    print(f"🔹 Unikalių akordų skaičius: {len(unique_chords)}")
    print(f"🎵 Unikalūs akordai: {unique_chords}\n")

# 🔹 Paverčiame "0" į NaN, kad galėtume lengvai skaičiuoti akordus
df.replace("0", None, inplace=True)

# 🔹 Apskaičiuojame, kiek kiekvienoje eilutėje yra realių akordų
df["num_chords"] = df.notna().sum(axis=1)

# 🔹 Suskaičiuojame eilučių skaičių pagal akordų kiekį
chord_counts = df["num_chords"].value_counts().sort_index()

# 🔹 Atspausdiname rezultatus
print("📊 **Eilučių skaičius pagal akordų kiekį:**")
for num_chords, count in chord_counts.items():
    print(f"🎵 {int(num_chords)-1} akordai: {count} eilutės")

# 🔹 Update normalized chords back to SQL using batch processing
update_query = """
    UPDATE test 
    SET chord_1 = %s, chord_2 = %s, chord_3 = %s, chord_4 = %s
    WHERE id = %s
"""

# Gather all the updates into a list of tuples
update_data = [(row["chord_1"], row["chord_2"], row["chord_3"], row["chord_4"], row["id"]) for _, row in df.iterrows()]

# Execute batch updates
try:
    db.execute_batch(update_query, update_data)
    print(f"✅ **Chord normalization complete! SQL database updated.**")
    print(f"🔄 **Total chords changed: {changed_chords}**")
except Exception as e:
    print(f"❌ Error during batch update: {e}")

# 🔹 Flatten all chords into a single list
all_chords = df[["chord_1", "chord_2", "chord_3", "chord_4"]].values.flatten()

# 🔹 Count occurrences of each chord
chord_counts = Counter(all_chords)

# 🔹 Define main chord roots
root_notes = ["A", "A#", "B", "B#", "C", "C#", "D", "D#", "E", "E#", "F", "F#", "G", "G#"]

# 🔹 Count chords grouped by root note
root_chord_counts = {root: 0 for root in root_notes}
for chord, count in chord_counts.items():
    if chord:  # Ignore None values
        root = re.match(r"^[A-G]#?", chord)  # Extract root note
        if root:
            root_chord_counts[root.group()] += count

# 🔹 Print occurrences per root note
print("🎵 **Chord occurrences grouped by root note:**")
for root, count in root_chord_counts.items():
    print(f"🎸 {root}: {count} times")

# 🔹 Count chords in each column separately
column_chord_counts = {col: Counter(df[col].dropna()) for col in ["chord_1", "chord_2", "chord_3", "chord_4"]}

# 🔹 Print occurrences per column
print("\n📊 **Chord occurrences per column:**")
for col, counts in column_chord_counts.items():
    print(f"\n🎸 {col}:")
    for chord, count in counts.most_common():
        print(f"  {chord}: {count} times")