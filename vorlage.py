# ==========================================
# Aufgabe 2: Support Vector Machines
# ==========================================

# ==========================================
# BLOCK 1: Bibliotheken importieren
# (nicht bearbeiten)
# ==========================================
import pandas as pd
import warnings
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ==========================================
# BLOCK 2: Datensatz laden und vorbereiten
# (nicht bearbeiten)
# ==========================================
train_csv = 'train.csv'
test_csv = 'test.csv'

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

print("Trainingsdatensatz (Ausschnitt):")
print(train_df.head())
print("\nTestdatensatz (Ausschnitt):")
print(test_df.head())

# Features und Labels trennen
X_train_raw = train_df.drop(columns=['label'])
y_train = train_df['label'].values
X_test_raw = test_df.copy()

# Daten skalieren (Standardisierung)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

print("\nDaten erfolgreich geladen und vorbereitet!")

# ==========================================
# BLOCK 3: SVM & Hyperparameter
# (HIER BEARBEITEN)
# ==========================================
# Konfigurieren Sie hier Ihre Support Vector Machine!



# --- 1. HYPERPARAMETER ---
KERNEL = 'rbf'
C = 2.2
GAMMA = 0.09
CLASS_WEIGHT = {-1:1,1:2.8} 


# --- 2. MODELL ERSTELLEN ---
# NICHT bearbeiten - das baut das Modell für das Training zusammen.
music_SVM = SVC(
    kernel=KERNEL,
    C=C,
    gamma=GAMMA,
    class_weight=CLASS_WEIGHT,
)
 
print("\nModell erfolgreich erstellt! Training wird gestartet...")

# ==========================================
# BLOCK 4: Training und Vorhersage
# (nicht bearbeiten)
# ==========================================
# Nun lassen wir das Modell anhand der Trainingsdaten lernen.
# Sobald das Training abgeschlossen ist, wird das Modell die Musik-Genres
# für den Testsatz vorhersagen und zur Bewertung in einer CSV speichern.

music_SVM.fit(X_train_scaled, y_train)

print("\nTraining abgeschlossen! Vorhersagen für den Testsatz werden generiert...")
predictions = music_SVM.predict(X_test_scaled)

# Vorhersagen speichern
submission_df = pd.DataFrame({'label': predictions})
submission_filename = 'result.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\nVorhersage abgeschlossen! Die Datei wurde lokal als '{submission_filename}' in Ihrem Ausführungsverzeichnis gespeichert.")
print("Lade diese Datei nun auf den Bewertungsserver hoch.")