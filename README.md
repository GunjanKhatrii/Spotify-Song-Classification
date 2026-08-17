# 🎵 Spotify Genre Classifier — SVM

> Automatically classifying songs as **Acoustic/Chill** or **High-Energy Party** using a Support Vector Machine, based purely on audio features.

**Final MCC Score: 0.709** on a hidden test set of 5,000 tracks.

---

## 💡 Motivation

Spotify receives over **60,000 new tracks every day**. Manually sorting them into playlists is impossible at that scale. This project explores whether a machine learning model can learn the "vibe" of a song — using only mathematical audio properties like energy, danceability, and tempo — and automatically assign it to the right mood category.

No lyrics. No genre tags. Just numbers.

---

## 🗂️ Project Structure

```
spotify-genre-classifier/
    ├── vorlage.py            # Training and prediction script
    ├── train.csv             # Labeled training data (2,000 tracks)
    ├── test.csv              # Unlabeled test data (5,000 tracks)
    ├── result.csv            # Model predictions on test set
    └── README.md             # This file
```

> ⚠️ **Note:** Some variable names and comments in the code are in German. You can translate them easily using [DeepL](https://www.deepl.com) or [Google Translate](https://translate.google.com).

---

## 📊 Dataset

A cleaned subset of the [Spotify Tracks dataset](https://www.kaggle.com/datasets/darrylljk/spotify-tracks), containing 2,000 labeled training tracks and 5,000 unlabeled test tracks.

| Feature | Description |
|---------|-------------|
| `danceability` | How suitable the track is for dancing |
| `energy` | Perceived intensity and activity level |
| `speechiness` | Proportion of spoken words |
| `liveness` | Probability the track was recorded live |
| `valence` | Musical positivity — happy vs. sad/angry |
| `tempo` | Speed in Beats Per Minute (BPM) |

**Target variable:**
- `-1` → Acoustic / Chill
- `+1` → High-Energy Party

> The dataset is imbalanced — 75% Acoustic and 25% Party. This is why MCC is used as the evaluation metric rather than plain accuracy.

---

## 🧠 Model

A **Support Vector Machine** with an RBF kernel. SVMs work by finding the decision boundary that **maximizes the margin** between the two classes — making predictions more robust on unseen data.

The RBF (Radial Basis Function) kernel was chosen because audio features have non-linear relationships with genre. A straight boundary simply doesn't capture the complexity of what makes a track feel "high energy."

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = SVC(
    kernel='rbf',
    C=2.2,
    gamma=0.09,
    class_weight={-1: 1, 1: 2.8}
)
model.fit(X_train_scaled, y_train)
```

### Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `kernel` | `'rbf'` | Handles non-linear boundaries between audio feature clusters |
| `C` | `2.2` | Balanced penalty — avoids both overfitting and underfitting |
| `gamma` | `0.09` | Smooth, medium-reach influence per training point |
| `class_weight` | `{-1: 1, 1: 2.8}` | Corrects for class imbalance — Party misses penalized 2.8× more |

---

## 📈 Results

Evaluated using **Matthews Correlation Coefficient (MCC)** — the gold standard for imbalanced binary classification. Unlike accuracy, MCC accounts for all four outcomes (TP, TN, FP, FN) simultaneously.

| Metric | Value |
|--------|-------|
| MCC (test set) | **0.709** |
| Acoustic recall | ~82% |
| Party recall | ~89% |

```
MCC = (TP × TN − FP × FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

A score of `0.709` means the model is genuinely learning audio signatures — not exploiting the class imbalance.

---

## 🔄 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/spotify-genre-classifier
cd spotify-genre-classifier
```

### 2. Install dependencies
```bash
pip install scikit-learn pandas numpy
```

### 3. Run
```bash
python vorlage.py
```

This generates `result.csv` with predictions for every track in `test.csv`.

---

## 🔍 Key Findings

- **Energy is the single most predictive feature** — alone it achieves MCC ≈ 0.57
- All 6 features together are needed for maximum performance — removing any one hurts
- The dataset has genuine overlap — some tracks are ambiguous even to humans
- The SVM hard ceiling on this feature set is around MCC ≈ 0.71, regardless of further tuning
- A balanced test set assumption (50/50 classes) was key to optimizing `class_weight` correctly

---

## 📚 Concepts

- [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine)
- [RBF Kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)
- [Matthews Correlation Coefficient](https://en.wikipedia.org/wiki/Phi_coefficient)
- [Soft Margin SVM](https://en.wikipedia.org/wiki/Support_vector_machine#Soft-margin)
- [Class Imbalance](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)
