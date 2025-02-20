import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from db_engine import DBEngine  # Jūsų SQL duomenų bazės klasė
import seaborn as sns
from sklearn.metrics import confusion_matrix


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

# 📌 Filter Rare Chord Sequences
def filter_rare_sequences(Y, min_count=2):
    unique, counts = np.unique(Y, axis=0, return_counts=True)
    valid_combinations = unique[counts >= min_count]
    mask = np.array([any((y == valid).all() for valid in valid_combinations) for y in Y])
    return mask

mask = filter_rare_sequences(Y)
Y, X_base = Y[mask], X_base[mask]

# 📌 Stratified Split ensuring distribution for all chords
def stratified_split(Y, test_size=0.1, random_state=42):
    unique_combinations, indices = np.unique(Y, axis=0, return_inverse=True)
    if np.min(np.bincount(indices)) < 2:
        print("⚠️ Warning: Some chord sequences are too rare for stratified splitting. Using random split instead.")
        return train_test_split(np.arange(len(Y)), test_size=test_size, random_state=random_state)
    else:
        return train_test_split(np.arange(len(Y)), test_size=test_size, stratify=indices, random_state=random_state)

train_idx, test_idx = stratified_split(Y)
X_train, X_test = X_base[train_idx], X_base[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

# 3️⃣ Define Hyperparameters
param_grid = {
    'n_estimators': [10, 20],
    'max_depth': [10, 20],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}
cv_folds = 3


best_models = []
rf_models = []
cross_val_results = {}

# 4️⃣ Training: Each chord prediction depends on previous chords
X_train_aug = X_train
X_test_aug = X_test

for i in range(Y_train.shape[1]):
    if i > 0:
        X_train_aug = np.hstack((X_train_aug, Y_train[:, :i]))
        X_test_aug = np.hstack((X_test_aug, Y_test[:, :i]))

    rf = RandomForestClassifier(random_state=42, class_weight='balanced', verbose=1)
    grid_search = GridSearchCV(rf, param_grid, cv=cv_folds, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train_aug, Y_train[:, i])
    best_model = grid_search.best_estimator_

    best_models.append(best_model)
    rf_models.append(best_model)
    cross_val_results[f"chord_{i + 1}"] = cross_val_score(best_model, X_train_aug, Y_train[:, i], cv=cv_folds, scoring='accuracy').mean()

# 5️⃣ Sequential Prediction on Test Set
Y_pred_test = np.zeros(Y_test.shape)

X_test_seq = X_test
for i, model in enumerate(rf_models):
    Y_pred_test[:, i] = model.predict(X_test_seq)
    if i < Y_test.shape[1] - 1:
        X_test_seq = np.hstack((X_test_seq, Y_pred_test[:, :i + 1]))

# 6️⃣ Decode Predictions
def decode_chords(encoded_chords):
    return [encoder.categories_[i][int(ch)] if not np.isnan(ch) else None for i, ch in enumerate(encoded_chords)]

real_chords = [decode_chords(y) for y in Y_test]
predicted_chords = [decode_chords(y) for y in Y_pred_test]

for i, model in enumerate(best_models):
    print(f"🎯 Best parameters for Chord {i+1}: {model.get_params()}")

# 7️⃣ Evaluate Performance on Test Data
accuracies = [accuracy_score(Y_test[:, i], Y_pred_test[:, i]) for i in range(Y_test.shape[1])]
precisions = [precision_score(Y_test[:, i], Y_pred_test[:, i], average='weighted', zero_division=0) for i in range(Y_test.shape[1])]
recalls = [recall_score(Y_test[:, i], Y_pred_test[:, i], average='weighted', zero_division=0) for i in range(Y_test.shape[1])]
f1_scores = [f1_score(Y_test[:, i], Y_pred_test[:, i], average='weighted') for i in range(Y_test.shape[1])]

print(f"✅ Test Avg Accuracy: {np.mean(accuracies):.4f}")
print(f"✅ Test Avg Precision: {np.mean(precisions):.4f}")
print(f"✅ Test Avg Recall: {np.mean(recalls):.4f}")
print(f"✅ Test Avg F1 Score: {np.mean(f1_scores):.4f}")

# 🔥 Show 5 sample test results
print("\n🔎 Sample Predictions:")
for i in range(5):
    print(f"Real: {real_chords[i]} -> Predicted: {predicted_chords[i]}")

# 🎵 Predict chords from user input lyrics
def predict_chords_from_lyrics():
    user_input = input("\n🎤 Enter a lyrics line: ")

    # 🔹 Load precomputed FastText embeddings
    fasttext_df = pd.read_csv("lyrics_fasttext.csv")
    scaler = StandardScaler()
    X_base = scaler.fit_transform(fasttext_df.values)  # Normalize stored vectors

    # 🔹 Generate embedding by averaging known vectors (if needed)
    def get_embedding_from_lyrics(text, fasttext_df):
        from nltk.tokenize import word_tokenize
        words = word_tokenize(text.lower())
        vectors = []

        for word in words:
            try:
                word_vec = fasttext_df.loc[:, word].values  # Fetch word's embedding (if exists)
                vectors.append(word_vec)
            except KeyError:
                continue  # Ignore words without embeddings

        return np.mean(vectors, axis=0) if vectors else np.zeros(300)  # Return mean vector or zero

    user_embedding = get_embedding_from_lyrics(user_input, fasttext_df).reshape(1, -1)
    user_embedding = scaler.transform(user_embedding)  # Normalize input

    # 🔹 Predict chords sequentially
    Y_pred_user = np.zeros((1, Y.shape[1]), dtype=int)
    X_user_seq = user_embedding

    for i, model in enumerate(rf_models):
        Y_pred_user[:, i] = model.predict(X_user_seq)
        if i < Y.shape[1] - 1:
            X_user_seq = np.hstack((X_user_seq, Y_pred_user[:, :i + 1]))

    # 🔹 Decode and print predictions
    predicted_chords_user = decode_chords(Y_pred_user[0])
    print(f"🎶 Predicted Chords: {predicted_chords_user}")

# Run prediction
predict_chords_from_lyrics()



# 1️⃣ Tikslumo analizė (accuracy per chord)
fig, ax = plt.subplots()
ax.bar(range(1, 5), accuracies, color='skyblue')
ax.set_xticks(range(1, 5))
ax.set_xticklabels([f'Chord {i}' for i in range(1, 5)])
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy per Chord")
plt.show()

# 2️⃣ Sanklodos matrica (Confusion Matrix kiekvienam akordui)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, ax in enumerate(axes.flat):
    cm = confusion_matrix(Y_test[:, i], Y_pred_test[:, i])
    cm_labels = encoder.categories_[i]  # Get decoded chord labels
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=cm_labels, yticklabels=cm_labels)
    ax.set_title(f'Confusion Matrix - Chord {i+1}')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()

# 3️⃣ 4-gramo tikslumo vizualizacija
fig, ax = plt.subplots()
ax.hist([1 if real == pred else 0 for real, pred in zip(real_chords, predicted_chords)], bins=2, color='green', alpha=0.7)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Incorrect', 'Correct'])
ax.set_ylabel("Count")
ax.set_title("4-Gram Accuracy Distribution")
plt.show()

# 4️⃣ Klaidų analizė
fig, ax = plt.subplots()
errors = [sum(1 for real, pred in zip(r, p) if real != pred) for r, p in zip(real_chords, predicted_chords)]
ax.hist(errors, bins=range(5), color='red', alpha=0.7)
ax.set_xticks(range(5))
ax.set_xlabel("Number of Errors per 4-Chord Sequence")
ax.set_ylabel("Count")
ax.set_title("Error Distribution Across 4-Chord Sequences")
plt.show()

# for i in range(Y_test.shape[1]):
#     cm = confusion_matrix(Y_test[:, i], Y_pred_test[:, i])
#     print(f"Confusion Matrix for Chord {i+1}:")
#     print(cm)
#     print("-" * 40)
