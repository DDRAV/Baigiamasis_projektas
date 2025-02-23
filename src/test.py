"""
LSTM Chord Prediction System

This script loads pre-trained LSTM models and predicts chord sequences based on user-input lyrics.

Features:
- Loads trained LSTM models from disk.
- Uses FastText embeddings for lyrics representation.
- Predicts chords sequentially, ensuring correct feature alignment.
- Searches for songs with matching chord progressions in the database.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from db_engine import DBEngine
from nltk.tokenize import word_tokenize

# ✅ Load Data & Encoder
fasttext_df = pd.read_csv("lyrics_fasttext.csv")
scaler = StandardScaler()
X_base = scaler.fit_transform(fasttext_df.values)  # Normalize stored vectors

db = DBEngine()
query = "SELECT id, chord_1, chord_2, chord_3, chord_4 FROM test;"
chords_df = pd.DataFrame(db.execute_sql(query), columns=["id", "chord_1", "chord_2", "chord_3", "chord_4"])

encoder = OrdinalEncoder()
encoder.fit(chords_df[["chord_1", "chord_2", "chord_3", "chord_4"]])

# ✅ Load Pre-trained LSTM Models
def load_lstm_models():
    return [load_model(f"saved_models/lstm_model_chord_{i + 1}.keras") for i in range(4)]

lstm_models = load_lstm_models()

# ✅ Generate FastText Embedding from Lyrics
def get_embedding_from_lyrics(text):
    words = word_tokenize(text.lower())
    vectors = [fasttext_df.loc[:, word].values for word in words if word in fasttext_df.columns]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)  # Return mean vector or zero

# ✅ Predict Chords Sequentially Using LSTM
def predict_chords(user_embedding, model_list):
    """
    Predicts chords sequentially using the LSTM models.

    Args:
        user_embedding (numpy array): The input lyrics embedding (300 features).
        model_list (list): List of trained LSTM models.

    Returns:
        list: Predicted chords as labels.
    """
    Y_pred_user = np.zeros((1, 4), dtype=int)  # Placeholder for predicted chords
    X_user_seq = user_embedding  # Start with only lyrics (300 features)

    for i, model in enumerate(model_list):
        if i > 0:
            X_user_seq = np.hstack((X_user_seq, Y_pred_user[:, :i]))  # Append previous chords

        expected_features = 300 + i  # Expected input features
        actual_features = X_user_seq.shape[1]  # What we're giving to the model

        print(f"🔍 Debug: LSTM Model {i+1} expects {expected_features} features, got {actual_features}.")  # Debug print

        assert actual_features == expected_features, (
            f"❌ LSTM Model {i+1} expects {expected_features} features, but got {actual_features}."
        )

        # Reshape for LSTM input
        X_user_seq_reshaped = X_user_seq.reshape((X_user_seq.shape[0], X_user_seq.shape[1], 1))
        Y_pred_user[:, i] = np.clip(np.round(model.predict(X_user_seq_reshaped)).flatten(), 0, len(encoder.categories_[i]) - 1)

        X_user_seq = np.hstack((X_user_seq, Y_pred_user[:, i].reshape(-1, 1)))

        print(f"✅ After Chord {i + 1} Prediction: X_user_seq now has {X_user_seq.shape[1]} features.")  # Debugging

    return [encoder.categories_[i][int(ch)] for i, ch in enumerate(Y_pred_user[0])]

# ✅ Predict Chords from User Input Lyrics
def predict_chords_from_lyrics():
    while True:
        user_input = input("\n🎤 Enter a lyrics line (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("👋 Exiting prediction system. Have a great day!")
            break

        user_embedding = get_embedding_from_lyrics(user_input).reshape(1, -1)
        user_embedding = scaler.transform(user_embedding)

        predictions = predict_chords(user_embedding, lstm_models)

        print("\n🎶 Predicted Chords:", predictions)

        # ✅ Search for matching chord sequences in DB
        print("\n🔍 Searching songs for matching chords...")
        chord_query = f"""
            SELECT artist, title, lyrics FROM test
            WHERE chord_1 = '{predictions[0]}' 
            AND chord_2 = '{predictions[1]}' 
            AND chord_3 = '{predictions[2]}' 
            AND chord_4 = '{predictions[3]}'
            LIMIT 3;
        """
        matching_lyrics = db.execute_sql(chord_query)

        if matching_lyrics:
            print("\n🎵 Songs matching predicted chord progression:")
            for i, row in enumerate(matching_lyrics, 1):
                artist, title, lyrics = row
                print(f"{i}. {artist} - {title} - {lyrics}")
        else:
            print("❌ No matching songs found.")

        while True:
            rerun_choice = input("\n🔁 Would you like to predict another lyrics line? (Yes/No): ").strip().lower()
            if rerun_choice == "yes":
                break  # Restart loop
            elif rerun_choice == "no":
                print("👋 Exiting prediction system. Have a great day!")
                return  # Exit function
            else:
                print("❌ Invalid input! Please type 'Yes' or 'No'.")

# ✅ Run Prediction Loop
if __name__ == "__main__":
    predict_chords_from_lyrics()
