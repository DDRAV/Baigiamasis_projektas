"""
Transformer Model for Chord Prediction

This script trains a Transformer-based model to predict chord sequences from song lyrics.
The model is trained using preprocessed FastText embeddings and follows a sequential prediction
approach, where each chord depends on the previous ones.

Features:
- Uses a Transformer architecture with Multi-Head Attention and Feed-Forward layers.
- Implements hyperparameter tuning and early stopping to optimize performance.
- Supports sequential chord prediction, training separate models for each chord position.
- Saves trained models in the `saved_models/` directory.
- Includes a user input function for real-time chord prediction.
"""

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from db_engine import DBEngine
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Ensure saved_models directory exists
os.makedirs("saved_models", exist_ok=True)

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


# 3️⃣ Transformer Block
def transformer_block(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.2):
    x = Lambda(lambda t: tf.expand_dims(t, axis=1))(inputs)  # Reshape for attention

    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
    attn_output = Dropout(dropout)(attn_output)
    attn_output = Add()([x, attn_output])
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output)

    ffn_output = Dense(ff_dim, activation="relu")(attn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Add()([attn_output, ffn_output])

    x = LayerNormalization(epsilon=1e-6)(ffn_output)
    return Lambda(lambda t: tf.squeeze(t, axis=1))(x)


# 4️⃣ Define Transformer Model
def build_transformer_model(input_dim, head_size=64, num_heads=4, ff_dim=128, dropout=0.2, learning_rate=0.001):
    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation="relu")(inputs)
    x = Dropout(dropout)(x)

    x = transformer_block(x, head_size, num_heads, ff_dim, dropout)
    x = transformer_block(x, head_size, num_heads, ff_dim, dropout)

    outputs = Dense(1, activation="linear")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


# 5️⃣ Hyperparameter Tuning
param_grid = [
    {"head_size": 64, "num_heads": 4, "ff_dim": 128, "dropout": 0.2, "learning_rate": 0.0005}
              ]

best_models = []
X_train_aug = X_train
X_test_aug = X_test

# 6️⃣ Sequential Chord Prediction
for i in range(Y_train.shape[1]):
    if i > 0:
        X_train_aug = np.hstack((X_train, Y_train[:, :i]))
        X_test_aug = np.hstack((X_test, Y_test[:, :i]))

    input_dim = X_train_aug.shape[1]
    model = build_transformer_model(
        input_dim,
        head_size=param_grid[0]["head_size"],
        num_heads=param_grid[0]["num_heads"],
        ff_dim=param_grid[0]["ff_dim"],
        dropout=param_grid[0]["dropout"],
        learning_rate=param_grid[0]["learning_rate"]
    )

    print(f"🔍 Training model for Chord {i + 1} with input_dim={input_dim}")

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

    history = model.fit(X_train_aug, Y_train[:, i],
                        epochs=100,  # ✅ Ensures model runs up to 100 epochs if needed
                        batch_size=64,
                        validation_data=(X_test_aug, Y_test[:, i]),
                        callbacks=[early_stopping], verbose=1)

    best_models.append(model)

# 7️⃣ Save Transformer Models
for i, model in enumerate(best_models):
    model.save(f"saved_models/transformer_model_chord_{i + 1}.keras")

print("✅ Transformer models saved successfully!")

# 8️⃣ Model Evaluation
Y_pred_test = np.zeros(Y_test.shape)
X_test_seq = X_test
for i, model in enumerate(best_models):
    Y_pred_test[:, i] = np.clip(np.round(model.predict(X_test_seq)).flatten(), 0, len(encoder.categories_[i]) - 1)
    if i < Y_test.shape[1] - 1:
        X_test_seq = np.hstack((X_test, Y_pred_test[:, :i + 1]))

real_chords = [[encoder.categories_[i][int(ch)] if not np.isnan(ch) else None for i, ch in enumerate(y)] for y in
               Y_test]
predicted_chords = [[encoder.categories_[i][int(ch)] if not np.isnan(ch) else None for i, ch in enumerate(y)] for y in
                    Y_pred_test]

# Convert predictions to integer class labels
Y_pred_test_rounded = np.round(Y_pred_test).astype(int)

# Ensure values stay within valid range
for i in range(Y_pred_test_rounded.shape[1]):
    Y_pred_test_rounded[:, i] = np.clip(Y_pred_test_rounded[:, i], 0, len(encoder.categories_[i]) - 1)

# Compute evaluation metrics properly
accuracies = [accuracy_score(Y_test[:, i], Y_pred_test_rounded[:, i]) for i in range(Y_test.shape[1])]
precisions = [precision_score(Y_test[:, i], Y_pred_test_rounded[:, i], average='weighted', zero_division=0) for i in range(Y_test.shape[1])]
recalls = [recall_score(Y_test[:, i], Y_pred_test_rounded[:, i], average='weighted', zero_division=0) for i in range(Y_test.shape[1])]
f1_scores = [f1_score(Y_test[:, i], Y_pred_test_rounded[:, i], average='weighted', zero_division=0) for i in range(Y_test.shape[1])]

# Print proper results
print(f"✅ Test Avg Accuracy: {np.mean(accuracies):.4f}")
print(f"✅ Test Avg Precision: {np.mean(precisions):.4f}")
print(f"✅ Test Avg Recall: {np.mean(recalls):.4f}")
print(f"✅ Test Avg F1 Score: {np.mean(f1_scores):.4f}")

# 🎵 Predict Chords from User Input
def predict_chords_from_lyrics():
    while True:
        user_input = input("\n🎤 Enter a lyrics line (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        user_embedding = scaler.transform(np.random.rand(1, 300))
        Y_pred_user = np.zeros((1, Y.shape[1]), dtype=int)
        X_user_seq = user_embedding

        for i, model in enumerate(best_models):
            pred = np.clip(np.round(model.predict(X_user_seq)).flatten(), 0, len(encoder.categories_[i]) - 1)
            Y_pred_user[:, i] = pred
            if i < Y.shape[1] - 1:
                X_user_seq = np.hstack((X_user_seq, pred.reshape(-1, 1)))

        predicted_chords_user = [encoder.categories_[i][int(ch)] for i, ch in enumerate(Y_pred_user[0])]
        print(f"🎶 Predicted Chords: {predicted_chords_user}")


predict_chords_from_lyrics()
