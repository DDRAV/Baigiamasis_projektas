import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Reshape, Dense, Dropout, Bidirectional, Input, Flatten, MultiHeadAttention, \
    LayerNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from nltk.translate.bleu_score import sentence_bleu
from db_engine import DBEngine  # Import your DB connection class

# 🔹 Load FastText Vectorized Lyrics
fasttext_df = pd.read_csv("lyrics_fasttext.csv")

# 🔹 Connect to Database & Load Chord Labels
db = DBEngine()
query = "SELECT id, chord_1, chord_2, chord_3, chord_4 FROM test;"
chords_df = pd.DataFrame(db.execute_sql(query), columns=["id", "chord_1", "chord_2", "chord_3", "chord_4"])

# 🔹 Ensure Data Alignment
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

print("✅ Data preparation complete!")


# 🔹 Define LSTM Model
def build_lstm_model():
    model = Sequential([
        Input(shape=(300,)),
        Reshape((1, 300)),
        Dense(256, activation="relu"),
        Dropout(0.2),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(4, activation="linear")  # Predict 4 chords (instead of 200)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# 🔹 Define Transformer Model
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs):
        # Reshape inputs to have shape (batch_size, 1, embed_dim)
        inputs = tf.expand_dims(inputs, axis=1)  # Adding time dimension (timesteps=1)

        # Self-attention with inputs as both query and value
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(inputs + attn_output)  # Add residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.norm2(out1 + ffn_output)  # Add residual connection


def build_transformer_model():
    input_layer = Input(shape=(300,))  # Input shape is (300,)
    x = Dense(256, activation="relu")(input_layer)  # Initial dense layer
    x = TransformerBlock(embed_dim=256, num_heads=8, ff_dim=512)(x)  # Apply transformer block
    x = Flatten()(x)  # Flatten the output after transformer block
    output_layer = Dense(4, activation="linear")(x)  # Output layer: Predict 4 chords
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# 🔹 Train Models
lstm_model = build_lstm_model()
transformer_model = build_transformer_model()

# Train LSTM Model
lstm_model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test))
print("LSTM model training complete!")

# Train Transformer Model
transformer_model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test))
print("Transformer model training complete!")


# 🔹 BLEU Score Evaluation
def calculate_bleu_score(real_chords, predicted_chords):
    return sentence_bleu([real_chords], predicted_chords)


# Example Evaluation (Needs real and predicted chord conversion back to names)
real_chords = ["C", "G", "Am", "F"]
predicted_chords = ["C", "G", "Em", "F"]
print("BLEU Score:", calculate_bleu_score(real_chords, predicted_chords))

print("✅ Model training & evaluation complete!")
