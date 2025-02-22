"""
Chord Prediction System Using Multiple Models

This script loads trained models (Random Forest, LSTM, Transformer) and predicts chord sequences from song lyrics.
It allows the user to input a lyrics line and returns chord predictions from all three models.

Features:
- Loads pre-trained models for chord prediction.
- Uses FastText embeddings for lyrics representation.
- Predicts chord sequences sequentially.
- Searches for songs with matching chord progressions in the database.
- Provides an interactive loop for multiple predictions.
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from db_engine import DBEngine
from nltk.tokenize import word_tokenize

# 🔹 Load Data & Encoder
fasttext_df = pd.read_csv("lyrics_fasttext.csv")
scaler = StandardScaler()
X_base = scaler.fit_transform(fasttext_df.values)  # Normalize stored vectors

db = DBEngine()
query = "SELECT id, chord_1, chord_2, chord_3, chord_4 FROM test;"
chords_df = pd.DataFrame(db.execute_sql(query), columns=["id", "chord_1", "chord_2", "chord_3", "chord_4"])

encoder = OrdinalEncoder()
encoder.fit(chords_df[["chord_1", "chord_2", "chord_3", "chord_4"]])


# 🔹 Load Trained Models
def load_rf_models():
    return [joblib.load(f"saved_models/rf_model_chord_{i + 1}.pkl") for i in range(4)]


def load_lstm_models():
    return [load_model(f"saved_models/lstm_model_chord_{i + 1}.keras") for i in range(4)]


def load_transformer_models():
    return [load_model(f"saved_models/transformer_model_chord_{i + 1}.keras") for i in range(4)]


# Load all models into a dictionary
models = {
    "rf": load_rf_models(),
    "lstm": load_lstm_models(),
    "transformer": load_transformer_models()
}


# 🔹 Generate FastText Embedding from Lyrics
def get_embedding_from_lyrics(text, fasttext_df):
    words = word_tokenize(text.lower())
    vectors = [fasttext_df.loc[:, word].values for word in words if word in fasttext_df.columns]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)  # Return mean vector or zero


# 🔹 Predict Chords for a Single Model
def predict_chords(user_embedding, model_list, model_type):
    Y_pred_user = np.zeros((1, 4), dtype=int)
    X_user_seq = user_embedding

    for i, model in enumerate(model_list):
        if model_type == "rf":
            Y_pred_user[:, i] = model.predict(X_user_seq)
        else:  # LSTM & Transformer require reshaping
            X_user_seq_reshaped = X_user_seq.reshape((X_user_seq.shape[0], X_user_seq.shape[1], 1))
            Y_pred_user[:, i] = np.clip(np.round(model.predict(X_user_seq_reshaped)).flatten(), 0,
                                        len(encoder.categories_[i]) - 1)

        if i < 3:
            X_user_seq = np.hstack((X_user_seq, Y_pred_user[:, :i + 1]))

    return [encoder.categories_[i][int(ch)] for i, ch in enumerate(Y_pred_user[0])]


# 🔹 Predict Chords from User Input Lyrics Using All Models
def predict_chords_from_lyrics():
    while True:
        user_input = input("\n🎤 Enter a lyrics line (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("👋 Exiting prediction system. Have a great day!")
            break

        user_embedding = get_embedding_from_lyrics(user_input, fasttext_df).reshape(1, -1)
        user_embedding = scaler.transform(user_embedding)

        predictions = {model_type: predict_chords(user_embedding, model_list, model_type) for model_type, model_list in
                       models.items()}

        print("\n🎶 Predicted Chords:")
        for model_type, chords in predictions.items():
            print(f"🔹 {model_type.upper()} Model: {chords}")

        # 🔹 Search for matching chord sequences in DB
        for model_type, predicted_chords in predictions.items():
            print(f"\n🔍 Searching songs for {model_type.upper()} model chords...")
            chord_query = f"""
                SELECT artist, title, lyrics FROM test
                WHERE chord_1 = '{predicted_chords[0]}' 
                AND chord_2 = '{predicted_chords[1]}' 
                AND chord_3 = '{predicted_chords[2]}' 
                AND chord_4 = '{predicted_chords[3]}'
                LIMIT 3;
            """
            matching_lyrics = db.execute_sql(chord_query)

            if matching_lyrics:
                print(f"\n🎵 Songs matching {model_type.upper()} chord progression:")
                for i, row in enumerate(matching_lyrics, 1):
                    artist, title, lyrics = row
                    print(f"{i}. {artist} - {title} - {lyrics}")
            else:
                print(f"❌ No matching songs found for {model_type.upper()} model.")

        while True:
            rerun_choice = input("\n🔁 Would you like to predict another lyrics line? (Yes/No): ").strip().lower()
            if rerun_choice == "yes":
                break  # Restart loop
            elif rerun_choice == "no":
                print("👋 Exiting prediction system. Have a great day!")
                return  # Exit function
            else:
                print("❌ Invalid input! Please type 'Yes' or 'No'.")


# Run Prediction Loop
if __name__ == "__main__":
    predict_chords_from_lyrics()
