# 🎵 Chord Prediction from Lyrics

This project implements a machine learning-based chord prediction system that suggests chord progressions for song lyrics. It leverages multiple models, including Random Forest, LSTM, and Transformer architectures, trained on a dataset of lyrics and corresponding chord sequences.

## 📌 Features
- **Data Scraping & Storage**: Automatically scrapes song lyrics and chords from Songsterr and stores them in a PostgreSQL database.
- **Preprocessing & Feature Engineering**: Cleans and normalizes chords, tokenizes lyrics, and converts lyrics into FastText embeddings.
- **Multiple Model Training**:
  - **Random Forest**: Traditional ML model for chord prediction.
  - **LSTM**: Sequential deep learning model capturing dependencies in chord sequences.
  - **Transformer**: Attention-based deep learning model for enhanced prediction.
- **Model Comparison & Evaluation**: Analyzes performance metrics such as accuracy, precision, recall, and F1-score.
- **Real-time Chord Prediction**: Allows users to input lyrics and receive predicted chord progressions using trained models.
- **Song Search**: Finds songs with similar chord progressions from the database.

## 📂 Project Structure
```
├── data/                      # Raw and processed datasets
├── saved_models/              # Trained models
├── src/
│   ├── db_engine.py           # PostgreSQL database connection
│   ├── scrapping.py           # Web scraping script for lyrics & chords
│   ├── preprocess_chords.py   # Chord cleaning and normalization
│   ├── preprocess_lyrics.py   # Lyrics preprocessing and FastText embedding
│   ├── train_rf.py            # Random Forest training & evaluation
│   ├── train_lstm.py          # LSTM model training & evaluation
│   ├── train_transformer.py   # Transformer model training & evaluation
│   ├── predict.py             # Runs predictions using all models
└── README.md
└── requirements.txt       # Dependencies
```

## 📊 Model Training & Evaluation
Each model is trained using a dataset of lyrics and their corresponding chord progressions. The training process involves:
1. **Data Preprocessing**:
   - Chord normalization and encoding.
   - Lyrics tokenization and conversion into vector embeddings using FastText.
2. **Training**:
   - Hyperparameter tuning using GridSearchCV (for Random Forest).
   - Sequential chord prediction for deep learning models.
3. **Evaluation**:
   - Accuracy, precision, recall, and F1-score calculations.
   - Confusion matrices and error analysis.

### 🔍 Model Performance
| Model         | AVG Accuracy | AVG Precision | AVG Recall | AVG F1 Score |
|---------------|--------------|---------------|------------|--------------|
| Random Forest | 66.04%       | 71.81%        | 66.04%     | 64.07%       |
| LSTM          | 1.92%        | 0.63%         | 1.92%      | 0.43%        |
| Transformer   | 17.39%       | 35.82%        | 17.39%     | 19.39%       |


## 🚀 Running the Project
### 1️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 2️⃣ Set Up PostgreSQL Database
Ensure your `.env` file contains database credentials:
```
DATABASE_NAME=your_database
DB_USERNAME=your_username
PASSWORD=your_password
HOST=your_host
PORT=your_port
```

### 3️⃣ Run Data Scraping
```sh
python src/scrapping.py
```

### 4️⃣ Preprocess Data
```sh
python src/preprocess_chords.py
python src/preprocess_lyrics.py
```

### 5️⃣ Train Models
```sh
python src/train_rf.py
python src/train_lstm.py
python src/train_transformer.py
```

### 6️⃣ Run Chord Prediction
```sh
python src/predict_chords.py
```

## 📌 Future Improvements
- Expand dataset with additional songs and genres.
- Enhance a separate model to predict how many chords should be generated.
- Implement a web interface for real-time chord prediction.
- Integrate more sophisticated NLP techniques for better lyric embeddings.
- Expand model so he could predict chords for paragraph not only for line of text.


