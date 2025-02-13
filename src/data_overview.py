import pandas as pd
from db_engine import DBEngine

# Prisijungiame prie duomenų bazės
db = DBEngine()

# SQL užklausa duomenims nuskaityti
query = "SELECT chord_1, chord_2, chord_3, chord_4, chord_5, chord_6 FROM test;"
df = pd.DataFrame(db.execute_sql(query), columns=["chord_1", "chord_2", "chord_3", "chord_4", "chord_5", "chord_6"])

# Unikalių akordų kiekviename stulpelyje skaičius
for col in df.columns:
    unique_chords = df[col].unique()
    print(f"🎸 Stulpelis: {col}")
    print(f"🔹 Unikalių akordų skaičius: {len(unique_chords)}")
    print(f"🎵 Unikalūs akordai: {unique_chords}\n")

# Paverčiame "0" į NaN, kad galėtume lengvai skaičiuoti akordus
df.replace("0", None, inplace=True)

# Apskaičiuojame, kiek kiekvienoje eilutėje yra realių akordų (ne '0')
df["num_chords"] = df.notna().sum(axis=1)

# Suskaičiuojame eilučių skaičių pagal akordų kiekį
chord_counts = df["num_chords"].value_counts().sort_index()

# Atspausdiname rezultatus
print("📊 Eilučių skaičius pagal akordų kiekį:")
for num_chords, count in chord_counts.items():
    print(f"🎵 {int(num_chords)} akordai: {count} eilutės")