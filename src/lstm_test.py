import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping  # ✅ Import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from db_engine import DBEngine
import seaborn as sns
import matplotlib.pyplot as plt

# Use all CPU cores
os.environ["OMP_NUM_THREADS"] = "8"  # Set to number of CPU cores
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
tf.config.threading.set_inter_op_parallelism_threads(8)  # Adjust as per your CPU
tf.config.threading.set_intra_op_parallelism_threads(8)

# 1️⃣ Load Data
fasttext_df = pd.read_csv("lyrics_fasttext.csv")

db = DBEngine()
query = "SELECT id, chord_1, chord_2, chord_3, chord_4 FROM test;"
chords_df = pd.DataFrame(db.execute_sql(query), columns=["id", "chord_1", "chord_2", "chord_3", "chord_4"])
assert len(fasttext_df) == len(chords_df), "Mismatch between FastText and chord dataset sizes!"

# 2️⃣ Encode Chords
encoder = OrdinalEncoder()
Y = encoder.fit_transform(chords_df[["chord_1", "chord_2", "chord_3", "chord_4"]])
X_base = fasttext_df.values
scaler = StandardScaler()
X_base = scaler.fit_transform(X_base)

# 📌 Stratified Split
train_idx, test_idx = train_test_split(
    np.arange(len(Y)), test_size=0.1, stratify=np.count_nonzero(~np.isnan(Y), axis=1), random_state=42
)
X_train, X_test = X_base[train_idx], X_base[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]


# 3️⃣ Define LSTM Model
def build_lstm_model(input_shape, units=64, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        Input(shape=input_shape),
        Reshape((input_shape[0], 1)),
        LSTM(units),  # Only one LSTM layer
        Dropout(dropout_rate),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


# 4️⃣ Hyperparameter Tuning
param_grid = [
    {"units": 64, "dropout_rate": 0.2, "learning_rate": 0.001, "epochs": 50, "batch_size": 64}  # Increased epochs for early stopping
]

best_models = []
X_train_aug = X_train
X_test_aug = X_test

# 5️⃣ Sequential Chord Prediction (Similar to RF Logic)
for i in range(Y_train.shape[1]):
    if i > 0:
        X_train_aug = np.hstack((X_train_aug, Y_train[:, :i]))
        X_test_aug = np.hstack((X_test_aug, Y_test[:, :i]))

    input_shape = (X_train_aug.shape[1], 1)  # Add time dimension

    best_model = None
    best_loss = float("inf")

    for params in param_grid:
        print(f"🔍 Training model for Chord {i + 1} with params: {params}")
        model = build_lstm_model(input_shape, units=params["units"], dropout_rate=params["dropout_rate"],
                                 learning_rate=params["learning_rate"])

        X_train_reshaped = X_train_aug.reshape((X_train_aug.shape[0], X_train_aug.shape[1], 1))  # Reshape to 3D
        X_test_reshaped = X_test_aug.reshape((X_test_aug.shape[0], X_test_aug.shape[1], 1))  # Reshape to 3D

        # ✅ Add EarlyStopping Callback
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=2)

        history = model.fit(
            X_train_reshaped, Y_train[:, i],
            epochs=params["epochs"], batch_size=params["batch_size"],
            validation_data=(X_test_reshaped, Y_test[:, i]),
            callbacks=[early_stopping],  # ✅ Added EarlyStopping
            verbose=1
        )

        val_loss = min(history.history["val_loss"])
        print(f"📉 Validation loss for Chord {i + 1}: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model  # Save best model

    best_models.append(best_model)

# 6️⃣ Sequential Prediction on Test Set
Y_pred_test = np.zeros(Y_test.shape)

X_test_seq = X_test
for i, model in enumerate(best_models):
    Y_pred_test[:, i] = np.clip(np.round(model.predict(X_test_seq)).flatten(), 0, len(encoder.categories_[i]) - 1)
    if i < Y_test.shape[1] - 1:
        X_test_seq = np.hstack((X_test_seq, Y_pred_test[:, :i + 1]))


# Save LSTM models
for i, model in enumerate(best_models):
    model.save(f"saved_models/lstm_model_chord_{i+1}")

print("✅ LSTM models saved successfully!")

# 7️⃣ Decode Predictions
def decode_chords(encoded_chords):
    return [encoder.categories_[i][int(ch)] if not np.isnan(ch) else None for i, ch in enumerate(encoded_chords)]


real_chords = [decode_chords(y) for y in Y_test]
predicted_chords = [decode_chords(y) for y in Y_pred_test]

# 8️⃣ Evaluate Performance on Test Data
accuracies = [accuracy_score(Y_test[:, i], Y_pred_test[:, i]) for i in range(Y_test.shape[1])]
precisions = [precision_score(Y_test[:, i], Y_pred_test[:, i], average='weighted', zero_division=0) for i in
              range(Y_test.shape[1])]
recalls = [recall_score(Y_test[:, i], Y_pred_test[:, i], average='weighted', zero_division=0) for i in
           range(Y_test.shape[1])]
f1_scores = [f1_score(Y_test[:, i], Y_pred_test[:, i], average='weighted') for i in range(Y_test.shape[1])]

print(f"✅ Test Avg Accuracy: {np.mean(accuracies):.4f}")
print(f"✅ Test Avg Precision: {np.mean(precisions):.4f}")
print(f"✅ Test Avg Recall: {np.mean(recalls):.4f}")
print(f"✅ Test Avg F1 Score: {np.mean(f1_scores):.4f}")

# 🔥 Show Sample Predictions
print("\n🔎 Sample Predictions:")
for i in range(5):
    print(f"Real: {real_chords[i]} -> Predicted: {predicted_chords[i]}")


# 🎵 Predict Chords from User Input
def predict_chords_from_lyrics():
    user_input = input("\n🎤 Enter a lyrics line: ")
    user_embedding = scaler.transform(np.random.rand(1, 300))  # Placeholder for real embedding function

    Y_pred_user = np.zeros((1, Y.shape[1]), dtype=int)
    X_user_seq = user_embedding

    for i, model in enumerate(best_models):
        Y_pred_user[:, i] = np.clip(np.round(model.predict(X_user_seq)), 0, len(encoder.categories_[i]) - 1)
        if i < Y.shape[1] - 1:
            X_user_seq = np.hstack((X_user_seq, Y_pred_user[:, :i + 1]))

    predicted_chords_user = decode_chords(Y_pred_user[0])
    print(f"🎶 Predicted Chords: {predicted_chords_user}")


predict_chords_from_lyrics()
