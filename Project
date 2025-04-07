# Grammar Score Prediction from Audio using Machine Learning
# Import necessary libraries
# Section 1: Imports
import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# Load the training CSV file
# Load the testing CSV file
# Section 2: Load Data
train_df = pd.read_csv("/kaggle/input/shl-intern-hiring-assessment/dataset/train.csv")
test_df = pd.read_csv("/kaggle/input/shl-intern-hiring-assessment/dataset/test.csv")
sample_submission = pd.read_csv("/kaggle/input/shl-intern-hiring-assessment/dataset/sample_submission.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Section 3: Feature Extraction

def extract_features(file_path):
     """
    Extracts various audio features from the given audio file:
    - MFCCs
    - Chroma STFT
    - Spectral Contrast
    - Tonnetz
    - Zero Crossing Rate
    - RMS Energy
    """
     y, sr = librosa.load(file_path, sr=None)
     features = {
        "mfcc_mean": np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        "zcr_mean": np.mean(librosa.feature.zero_crossing_rate(y)),
        "spec_centroid_mean": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "rolloff_mean": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "chroma_stft_mean": np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
    }
     return features

# Extract train features
train_audio_folder = "/kaggle/input/shl-intern-hiring-assessment/dataset/audios_train"
train_features = []

for filename in train_df["filename"]:
    path = os.path.join(train_audio_folder, filename)
    features = extract_features(path)
    features["filename"] = filename
    train_features.append(features)

features_df = pd.DataFrame(train_features)

# Section 4: Data Preprocessing
features_df = features_df.drop(columns=["filename"], errors="ignore")
X = features_df
y = train_df["label"]  # Label corresponds to Grammar Score

# Section 5: Train-Test Split & Modeling
# Split into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train a Random Forest Regressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_val) # Make predictions on validation set

# Section 6: Evaluation

# Evaluate using MAE and Pearson Correlation
mae = mean_absolute_error(y_val, y_pred)
pearson_corr = pearsonr(y_val, y_pred)[0]
print("MAE:", mae)
print("Pearson Correlation:", pearson_corr)

# Section 7: Visualizations
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_val, y=y_pred)
plt.xlabel("Actual Grammar Score")
plt.ylabel("Predicted Grammar Score")
plt.title("Actual vs Predicted Grammar Score")
plt.grid(True)
plt.show()

# Section 8: Predict on Test Data
test_audio_folder = "/kaggle/input/shl-intern-hiring-assessment/dataset/audios_test"
test_features = []

for filename in test_df["filename"]:
    path = os.path.join(test_audio_folder, filename)
    features = extract_features(path)
    features["filename"] = filename
    test_features.append(features)

test_features_df = pd.DataFrame(test_features)
test_features_df = test_features_df.drop(columns=["filename"], errors="ignore")
test_preds = model.predict(test_features_df)

# Section 9: Submission
submission = pd.DataFrame({
    "filename": test_df["filename"],
    "label": test_preds
})
submission.to_csv("submission.csv", index=False)

# Section 10: Summary Report
print("\n--- Report Summary ---")
print("Model Used: RandomForestRegressor")
print(f"MAE on Validation Set: {mae:.4f}")
print(f"Pearson Correlation on Validation Set: {pearson_corr:.4f}")
print("Features Extracted: MFCC Mean, Zero Crossing Rate, Spectral Centroid, Spectral Rolloff, Chroma STFT")
print("Evaluation Metric: Pearson Correlation (for leaderboard)")
