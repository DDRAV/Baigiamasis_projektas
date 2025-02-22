"""
LSTM Model Training for Chord Prediction

This script trains **LSTM models** to predict chord sequences from song lyrics embeddings.
It performs the following steps:

1. **Load Data** - Retrieves precomputed FastText embeddings and chord labels from the database.
2. **Preprocessing** - Encodes chords, normalizes input features, and splits data for training/testing.
3. **Train LSTM Models** - Trains one LSTM model per chord position using hyperparameter tuning.
4. **Evaluate Performance** - Computes accuracy, precision, recall, and F1-score.
5. **Save Models** - Saves trained models for later inference.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from db_engine import DBEngine

# ✅ Configure TensorFlow for Multi-Core Processing
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# ✅ Step 1: Load Data
print("\n📥 Loading Data...")
fasttext_df = pd.read_csv("lyrics_fasttext.csv")  # Precomputed embeddings

# Fetch chord labels from the database
db = DBEngine()
query = "SELECT id, chord_1, chord_2, chord_3, chord_4 FROM test;"
chords_df = pd.DataFrame(db.execute_sql(query), columns=["id", "chord_1", "chord_2", "chord_3", "chord_4"])

# Ensure dataset consistency
assert len(fasttext_df) == len(chords_df), "❌ Mismatch between lyrics embeddings and chord dataset sizes!"

# ✅ Step 2: Encode Chords
encoder = OrdinalEncoder()
Y = encoder.fit_transform(chords_df[["chord_1", "chord_2", "chord_3", "chord_4"]])  # Convert chords to numerical labels

# Standardize input features
X_base = fasttext_df.values
scaler = StandardScaler()
X_base = scaler.fit_transform(X_base)

# ✅ Step 3: Train-Test Split
train_idx, test_idx = train_test_split(
    np.arange(len(Y)), test_size=0.1, stratify=np.count_nonzero(~np.isnan(Y), axis=1), random_state=42
)
X_train, X_test = X_base[train_idx], X_base[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

# ✅ Step 4: Define LSTM Model
def build_lstm_model(input_shape, units=64, dropout_rate=0.2, learning_rate=0.001):
    """
    Builds and compiles an LSTM model for chord sequence prediction.

    Args:
        input_shape (tuple): Shape of the input data.
        units (int): Number of LSTM units.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for Adam optimizer.

    Returns:
        model: Compiled LSTM model.
    """
    model = Sequential([
        Input(shape=input_shape),
        Reshape((input_shape[0], 1)),  # Reshape input to 3D for LSTM
        LSTM(units),  # LSTM layer
        Dropout(dropout_rate),
        Dense(1, activation="linear")  # Output a single value for regression
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model

# ✅ Step 5: Hyperparameter Tuning
param_grid = [
    {"units": 64, "dropout_rate": 0.3, "learning_rate": 0.001, "epochs": 50, "batch_size": 64}
]

best_models = []
X_train_aug = X_train
X_test_aug = X_test

# ✅ Step 6: Train LSTM Models for Each Chord Position
for i in range(Y_train.shape[1]):
    if i > 0:
        X_train_aug = np.hstack((X_train_aug, Y_train[:, :i]))  # Append previous chord predictions
        X_test_aug = np.hstack((X_test_aug, Y_test[:, :i]))

    input_shape = (X_train_aug.shape[1], 1)  # Shape for LSTM

    best_model = None
    best_loss = float("inf")

    for params in param_grid:
        print(f"\n🔍 Training LSTM for Chord {i + 1} with params: {params}")

        model = build_lstm_model(input_shape, units=params["units"], dropout_rate=params["dropout_rate"],
                                 learning_rate=params["learning_rate"])

        X_train_reshaped = X_train_aug.reshape((X_train_aug.shape[0], X_train_aug.shape[1], 1))
        X_test_reshaped = X_test_aug.reshape((X_test_aug.shape[0], X_test_aug.shape[1], 1))

        # ✅ Early Stopping
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=2)

        history = model.fit(
            X_train_reshaped, Y_train[:, i],
            epochs=params["epochs"], batch_size=params["batch_size"],
            validation_data=(X_test_reshaped, Y_test[:, i]),
            callbacks=[early_stopping],
            verbose=1
        )

        val_loss = min(history.history["val_loss"])
        print(f"📉 Validation loss for Chord {i + 1}: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

    best_models.append(best_model)

# ✅ Step 7: Save LSTM Models
os.makedirs("saved_models", exist_ok=True)
for i, model in enumerate(best_models):
    model.save(f"saved_models/lstm_model_chord_{i+1}.keras")

print("\n✅ LSTM models saved successfully!")
# ✅ Step 8: Evaluate Model Performance
def decode_chords(encoded_chords):
    """ Decodes numerical chord predictions back to their original labels. """
    return [encoder.categories_[i][int(ch)] if not np.isnan(ch) else None for i, ch in enumerate(encoded_chords)]

real_chords = [decode_chords(y) for y in Y_test]
predicted_chords = [decode_chords(y) for y in np.round(Y_test)]
Y_pred_test = np.zeros(Y_test.shape)

X_test_seq = X_test
for i, model in enumerate(best_models):
    Y_pred_test[:, i] = np.clip(np.round(model.predict(X_test_seq)).flatten(), 0, len(encoder.categories_[i]) - 1)
    if i < Y_test.shape[1] - 1:
        X_test_seq = np.hstack((X_test_seq, Y_pred_test[:, :i + 1]))

# Convert predictions to proper integer format
Y_pred_test_rounded = np.round(Y_pred_test).astype(int)

# Ensure predictions stay within valid range
for i in range(Y_pred_test_rounded.shape[1]):
    Y_pred_test_rounded[:, i] = np.clip(Y_pred_test_rounded[:, i], 0, len(encoder.categories_[i]) - 1)

# Compute evaluation metrics properly
accuracies = [accuracy_score(Y_test[:, i], Y_pred_test_rounded[:, i]) for i in range(Y_test.shape[1])]
precisions = [precision_score(Y_test[:, i], Y_pred_test_rounded[:, i], average='weighted', zero_division=0) for i in range(Y_test.shape[1])]
recalls = [recall_score(Y_test[:, i], Y_pred_test_rounded[:, i], average='weighted', zero_division=0) for i in range(Y_test.shape[1])]
f1_scores = [f1_score(Y_test[:, i], Y_pred_test_rounded[:, i], average='weighted') for i in range(Y_test.shape[1])]

print(f"\n✅ Test Avg Accuracy: {np.mean(accuracies):.4f}")
print(f"✅ Test Avg Precision: {np.mean(precisions):.4f}")
print(f"✅ Test Avg Recall: {np.mean(recalls):.4f}")
print(f"✅ Test Avg F1 Score: {np.mean(f1_scores):.4f}")




# ✅ Step 9: Predict Chords from User Input
def predict_chords_from_lyrics():
    """
    Predicts chord sequences based on user-input lyrics.
    """
    user_input = input("\n🎤 Enter a lyrics line: ")
    user_embedding = scaler.transform(np.random.rand(1, 300))  # Placeholder embedding

    Y_pred_user = np.zeros((1, Y.shape[1]), dtype=int)
    X_user_seq = user_embedding

    for i, model in enumerate(best_models):
        Y_pred_user[:, i] = np.clip(np.round(model.predict(X_user_seq)), 0, len(encoder.categories_[i]) - 1)
        if i < Y.shape[1] - 1:
            X_user_seq = np.hstack((X_user_seq, Y_pred_user[:, :i + 1]))

    predicted_chords_user = decode_chords(Y_pred_user[0])
    print(f"🎶 Predicted Chords: {predicted_chords_user}")

predict_chords_from_lyrics()
