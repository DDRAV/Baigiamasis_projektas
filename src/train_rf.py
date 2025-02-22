"""
Random Forest Model Training for Chord Prediction

This script trains a **Random Forest model** to predict **chord sequences** based on **lyrics embeddings**.
It performs the following steps:
1. **Load Data** - Fetches preprocessed lyrics embeddings and chord labels from the database.
2. **Preprocessing** - Encodes chords, normalizes input features, and filters rare sequences.
3. **Train Random Forest Models** - Trains one model per chord position, using Grid Search for hyperparameter tuning.
4. **Evaluate Performance** - Measures accuracy, precision, recall, and F1-score.
5. **Save Models** - Stores trained models for later use in inference.

"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from db_engine import DBEngine  # SQL Database connection

# ✅ Step 1: Load Data
print("\n📥 Loading Data...")
fasttext_df = pd.read_csv("lyrics_fasttext.csv")  # Precomputed FastText embeddings

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

# ✅ Step 3: Filter Rare Chord Sequences
def filter_rare_sequences(Y, min_count=2):
    """
    Filters out rare chord sequences that appear less than 'min_count' times.
    Helps the model generalize better.
    """
    unique, counts = np.unique(Y, axis=0, return_counts=True)
    valid_combinations = unique[counts >= min_count]
    mask = np.array([any((y == valid).all() for valid in valid_combinations) for y in Y])
    return mask

mask = filter_rare_sequences(Y)
Y, X_base = Y[mask], X_base[mask]

# ✅ Step 4: Stratified Split (Ensuring Chord Distribution)
def stratified_split_by_chord_count(Y, test_size=0.1, random_state=42):
    """
    Performs stratified train-test split based on the number of chords per row.
    Ensures even distribution of simple vs. complex chord sequences.
    """
    num_chords_per_row = np.count_nonzero(~np.isnan(Y), axis=1)  # Count non-NaN values per row
    train_idx, test_idx = train_test_split(
        np.arange(len(Y)), test_size=test_size, stratify=num_chords_per_row, random_state=random_state
    )
    return train_idx, test_idx

train_idx, test_idx = stratified_split_by_chord_count(Y)
X_train, X_test = X_base[train_idx], X_base[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

# ✅ Step 5: Define Hyperparameters
param_grid = {
    'n_estimators': [200],  # Number of trees
    'max_depth': [20, 50],  # Tree depth
    'min_samples_split': [5],  # Minimum samples to split a node
    'min_samples_leaf': [2],  # Minimum samples per leaf node
    'criterion': ["entropy"]  # Splitting criterion
}
cv_folds = 2

best_models = []
rf_models = []
cross_val_results = {}

# ✅ Step 6: Train Random Forest Models for Each Chord
X_train_aug = X_train
X_test_aug = X_test

for i in range(Y_train.shape[1]):  # Train a model per chord position
    if i > 0:
        X_train_aug = np.hstack((X_train_aug, Y_train[:, :i]))  # Add previous chords
        X_test_aug = np.hstack((X_test_aug, Y_test[:, :i]))  # Ensure the test set matches

    print(f"\n🎸 Training Random Forest for Chord {i+1}...")

    rf = RandomForestClassifier(random_state=42, class_weight='balanced', verbose=1, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=cv_folds, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_aug, Y_train[:, i])
    best_model = grid_search.best_estimator_

    best_models.append(best_model)
    rf_models.append(best_model)
    cross_val_results[f"chord_{i + 1}"] = cross_val_score(best_model, X_train_aug, Y_train[:, i], cv=cv_folds, scoring='accuracy').mean()

# ✅ Step 7: Predict on Test Set
Y_pred_test = np.zeros(Y_test.shape)

X_test_seq = X_test
for i, model in enumerate(rf_models):
    Y_pred_test[:, i] = model.predict(X_test_seq)
    if i < Y_test.shape[1] - 1:
        X_test_seq = np.hstack((X_test_seq, Y_pred_test[:, :i+1]))

# ✅ Step 8: Save Models
os.makedirs("saved_models", exist_ok=True)
for i, model in enumerate(rf_models):
    joblib.dump(model, f"saved_models/rf_model_chord_{i+1}.pkl")
print("\n✅ Random Forest models saved successfully!")

# ✅ Step 9: Evaluate Model Performance
def decode_chords(encoded_chords):
    """ Decodes numerical chord predictions back to their original labels. """
    return [encoder.categories_[i][int(ch)] if not np.isnan(ch) else None for i, ch in enumerate(encoded_chords)]

real_chords = [decode_chords(y) for y in Y_test]
predicted_chords = [decode_chords(y) for y in Y_pred_test]

# Calculate accuracy metrics
accuracies = [accuracy_score(Y_test[:, i], Y_pred_test[:, i]) for i in range(Y_test.shape[1])]
precisions = [precision_score(Y_test[:, i], Y_pred_test[:, i], average='weighted', zero_division=0) for i in range(Y_test.shape[1])]
recalls = [recall_score(Y_test[:, i], Y_pred_test[:, i], average='weighted', zero_division=0) for i in range(Y_test.shape[1])]
f1_scores = [f1_score(Y_test[:, i], Y_pred_test[:, i], average='weighted') for i in range(Y_test.shape[1])]

print(f"\n✅ Test Avg Accuracy: {np.mean(accuracies):.4f}")
print(f"✅ Test Avg Precision: {np.mean(precisions):.4f}")
print(f"✅ Test Avg Recall: {np.mean(recalls):.4f}")
print(f"✅ Test Avg F1 Score: {np.mean(f1_scores):.4f}")

# ✅ Step 10: Visualizations
fig, ax = plt.subplots()
ax.bar(range(1, 5), accuracies, color='skyblue')
ax.set_xticks(range(1, 5))
ax.set_xticklabels([f'Chord {i}' for i in range(1, 5)])
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy per Chord")
plt.show()

# Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, ax in enumerate(axes.flat):
    cm = confusion_matrix(Y_test[:, i], Y_pred_test[:, i])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=encoder.categories_[i], yticklabels=encoder.categories_[i])
    ax.set_title(f'Confusion Matrix - Chord {i+1}')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()
