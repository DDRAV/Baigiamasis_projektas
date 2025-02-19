import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Reshape, Dense, Dropout, Bidirectional, Input, Flatten, MultiHeadAttention, LayerNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

# 🔹 Reverse the Ordinal Encoding (for BLEU Score Evaluation)
def decode_chords(encoded_chords):
    # Ensure the values are within the valid range of the encoder's categories
    try:
        return encoder.inverse_transform(encoded_chords)
    except IndexError as e:
        print(f"Error in decoding chords: {e}")
        print("Encoded chords:", encoded_chords)
        return None

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
        Dense(4, activation="linear")  # Predict 4 chords * 50D embeddings
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
        inputs = tf.expand_dims(inputs, axis=1)  # Adding time dimension (timesteps=1)
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

# 🔹 Train Random Forest Model
print("Training Random Forest Model...")
#rf_model = RandomForestRegressor(n_estimators=100, random_state=42, verbose=0)  # Set n_estimators to a small value for quick testing
#rf_model.fit(X_train, Y_train)  # Train the model on the entire training data
print("Random Forest Model training complete!")

# 🔹 Train Neural Models
lstm_model = build_lstm_model()
transformer_model = build_transformer_model()

try:
    print("Training LSTM Model...")
    lstm_model.fit(X_train, Y_train, epochs=3, batch_size=32, validation_data=(X_test, Y_test), verbose=1)
    print("LSTM Model training complete!")
except Exception as e:
    print(f"Error during LSTM model training: {e}")

try:
    print("Training Transformer Model...")
    transformer_model.fit(X_train, Y_train, epochs=3, batch_size=32, validation_data=(X_test, Y_test), verbose=1)
    print("Transformer Model training complete!")
except Exception as e:
    print(f"Error during Transformer model training: {e}")

# 🔹 Evaluate Random Forest Model
# try:
#     print("Evaluating Random Forest Model...")
#     rf_predictions = rf_model.predict(X_test)
#     rf_mae = mean_absolute_error(Y_test, rf_predictions)
#     rf_mse = mean_squared_error(Y_test, rf_predictions)
#
#     print("Random Forest Model Evaluation:")
#     print(f"MAE: {rf_mae}")
#     print(f"MSE: {rf_mse}")
# except Exception as e:
#     print(f"Error during Random Forest evaluation: {e}")

# 🔹 BLEU Score Calculation
def calculate_bleu_score(real_chords, predicted_chords):
    return sentence_bleu([real_chords], predicted_chords)

# 🔹 Generate Predictions from Models
try:
    print("Generating predictions from LSTM model...")
    lstm_predictions = lstm_model.predict(X_test)
    print("LSTM Predictions:", lstm_predictions)
except Exception as e:
    print(f"Error during LSTM prediction: {e}")

try:
    print("Generating predictions from Transformer model...")
    transformer_predictions = transformer_model.predict(X_test)
    print("Transformer Predictions:", transformer_predictions)
except Exception as e:
    print(f"Error during Transformer prediction: {e}")

# Užtikriname, kad reikšmės būtų teisingame intervale
lstm_predictions = np.round(lstm_predictions).astype(int)
transformer_predictions = np.round(transformer_predictions).astype(int)

lstm_predictions = np.clip(lstm_predictions, 0, len(encoder.categories_[0]) - 1)
transformer_predictions = np.clip(transformer_predictions, 0, len(encoder.categories_[0]) - 1)

# 🔹 Decode Predictions to Chords (with additional checks)
lstm_predictions_chords = decode_chords(lstm_predictions)
transformer_predictions_chords = decode_chords(transformer_predictions)

# Check if decoding was successful before continuing
if lstm_predictions_chords is not None and transformer_predictions_chords is not None:
    #rf_predictions_chords = decode_chords(rf_predictions)

    # 🔹 Convert Y_test to Actual Chord Names
    Y_test_chords = decode_chords(Y_test)

    # 🔹 Calculate BLEU Score for Each Model
    lstm_bleu = calculate_bleu_score(Y_test_chords[0], lstm_predictions_chords[0])
    transformer_bleu = calculate_bleu_score(Y_test_chords[0], transformer_predictions_chords[0])
    #rf_bleu = calculate_bleu_score(Y_test_chords[0], rf_predictions_chords[0])

    # 🔹 Print BLEU Scores for Each Model
    print("BLEU Scores for Each Model:")
    print("LSTM sugeneruoti akordai:", lstm_predictions_chords[0])
    print("Transformer sugeneruoti akordai:", transformer_predictions_chords[0])

    print(f"LSTM BLEU Score: {lstm_bleu}")
    print(f"Transformer BLEU Score: {transformer_bleu}")
    #print(f"Random Forest BLEU Score: {rf_bleu}")
else:
    print("Prediction decoding failed. Skipping BLEU score calculation.")

# 🔹 Output final metrics
print("✅ Model training & evaluation complete!")
