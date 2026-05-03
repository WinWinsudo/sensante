import pandas as pd
import numpy as np

# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")

# Vérifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")


from sklearn.preprocessing import LabelEncoder

le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print(f"\nFeatures X : {X.shape}")
print(f"Cible y    : {y.shape}")

# ============================================================
# ETAPE 3 : Séparer entraînement et test
# ============================================================
from sklearn.model_selection import train_test_split

print("\n" + "=" * 50)
print("ETAPE 3 : Separation entrainement / test")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Total patients     : {len(X)}")
print(f"Entrainement (80%) : {X_train.shape[0]} patients")
print(f"Test         (20%) : {X_test.shape[0]} patients")
print(f"\nDistribution dans le test :")
print(y_test.value_counts())

# ============================================================
# ETAPE 4 : Entraîner le modèle
# ============================================================
from sklearn.ensemble import RandomForestClassifier

print("\n" + "=" * 50)
print("ETAPE 4 : Entrainement du modele RandomForest")
print("=" * 50)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Modele entraine avec succes !")
print(f"Nombre d'arbres    : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes            : {list(model.classes_)}")

# ============================================================
# ETAPE 5.1 : Prédictions sur les données de test
# ============================================================
print("\n" + "=" * 50)
print("ETAPE 5.1 : Predictions sur les donnees de test")
print("=" * 50)

y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    'Vrai diagnostic' : y_test.values[:10],
    'Prediction'      : y_pred[:10],
    'Correct ?'       : ['OK' if v == p else 'ERREUR'
                         for v, p in zip(y_test.values[:10], y_pred[:10])]
})
print("10 premieres predictions vs realite :")
print(comparison.to_string(index=False))

# ============================================================
# ETAPE 5.2 : Accuracy
# ============================================================
from sklearn.metrics import accuracy_score

print("\n" + "=" * 50)
print("ETAPE 5.2 : Accuracy du modele")
print("=" * 50)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy              : {accuracy:.2%}")
print(f"Predictions correctes : {int(accuracy * 100)} / 100 patients")
print(f"Erreurs               : {100 - int(accuracy * 100)} / 100 patients")

# ============================================================
# ETAPE 5.3 : Matrice de confusion
# ============================================================
from sklearn.metrics import confusion_matrix, classification_report

print("\n" + "=" * 50)
print("ETAPE 5.3 : Matrice de confusion")
print("=" * 50)

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Matrice de confusion :")
print(cm)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# ============================================================
# ETAPE 5.4 : Visualisation matrice de confusion
# ============================================================
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("\n" + "=" * 50)
print("ETAPE 5.4 : Visualisation matrice de confusion")
print("=" * 50)

os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()
print("Figure sauvegardee : figures/confusion_matrix.png")

# ============================================================
# ETAPE 6.1 : Sérialiser le modèle
# ============================================================
import joblib

print("\n" + "=" * 50)
print("ETAPE 6.1 : Serialisation du modele")
print("=" * 50)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

size = os.path.getsize("models/model.pkl")
print(f"Modele sauvegarde : models/model.pkl")
print(f"Taille            : {size / 1024:.1f} Ko")

# ============================================================
# ETAPE 6.2 : Sauvegarder les encodeurs
# ============================================================
print("\n" + "=" * 50)
print("ETAPE 6.2 : Sauvegarde des encodeurs")
print("=" * 50)

joblib.dump(le_sexe,      "models/encoder_sexe.pkl")
joblib.dump(le_region,    "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Fichiers sauvegardes dans models/ :")
for f in os.listdir("models"):
    taille = os.path.getsize(f"models/{f}")
    print(f"  - {f} ({taille / 1024:.1f} Ko)")

# ============================================================
# ETAPE 7.1 : Recharger le modèle
# ============================================================
print("\n" + "=" * 50)
print("ETAPE 7.1 : Rechargement du modele depuis fichier")
print("=" * 50)

model_loaded     = joblib.load("models/model.pkl")
le_sexe_loaded   = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"Modele recharge : {type(model_loaded).__name__}")
print(f"Classes         : {list(model_loaded.classes_)}")

# ============================================================
# ETAPE 7.2 : Prédire pour un nouveau patient
# ============================================================
print("\n" + "=" * 50)
print("ETAPE 7.2 : Prediction pour un nouveau patient")
print("=" * 50)

nouveau_patient = {
    'age': 28, 'sexe': 'F', 'temperature': 39.5,
    'tension_sys': 110, 'toux': True, 'fatigue': True,
    'maux_tete': True, 'region': 'Dakar'
}

print("Donnees du patient :")
for k, v in nouveau_patient.items():
    print(f"  {k:12s} : {v}")

sexe_enc   = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

features = [
    nouveau_patient['age'], sexe_enc,
    nouveau_patient['temperature'], nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']), int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']), region_enc
]

diagnostic = model_loaded.predict([features])[0]
probas     = model_loaded.predict_proba([features])[0]

print(f"\n--- Resultat du pre-diagnostic ---")
print(f"Patient    : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilite: {probas.max():.1%}")
print(f"\nProbabilites par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"  {classe:8s} : {proba:.1%} {bar}")

print("\n" + "=" * 50)
print("LAB 2 TERMINE ! Fichiers dans models/")
print("=" * 50)

# ============================================================
# EXERCICE 1 : Importance des features
# ============================================================
print("\n" + "=" * 50)
print("EXERCICE 1 : Importance des features")
print("=" * 50)

importances = model.feature_importances_
print("Features classees par importance :")
for name, imp in sorted(zip(feature_cols, importances),
                        key=lambda x: x[1], reverse=True):
    bar = '#' * int(imp * 50)
    print(f"  {name:20s} : {imp:.3f} {bar}")

# ============================================================
# EXERCICE 2 : Tester avec 3 autres patients
# ============================================================
print("\n" + "=" * 50)
print("EXERCICE 2 : Test avec 3 patients fictifs")
print("=" * 50)

patients = [
    {
        'nom': 'Patient 1 - Jeune sans symptomes',
        'age': 18, 'sexe': 'M', 'temperature': 37.0,
        'tension_sys': 120, 'toux': False, 'fatigue': False,
        'maux_tete': False, 'region': 'Dakar'
    },
    {
        'nom': 'Patient 2 - Adulte forte fievre',
        'age': 45, 'sexe': 'F', 'temperature': 40.5,
        'tension_sys': 130, 'toux': True, 'fatigue': True,
        'maux_tete': True, 'region': 'Thies'
    },
    {
        'nom': 'Patient 3 - Age avec toux',
        'age': 67, 'sexe': 'M', 'temperature': 38.8,
        'tension_sys': 95, 'toux': True, 'fatigue': True,
        'maux_tete': False, 'region': 'Diourbel'
    },
]

for p in patients:
    sexe_enc   = le_sexe_loaded.transform([p['sexe']])[0]
    region_enc = le_region_loaded.transform([p['region']])[0]

    features = pd.DataFrame([[
        p['age'], sexe_enc, p['temperature'], p['tension_sys'],
        int(p['toux']), int(p['fatigue']), int(p['maux_tete']), region_enc
    ]], columns=feature_cols)

    diag  = model_loaded.predict(features)[0]
    proba = model_loaded.predict_proba(features)[0].max()

    print(f"\n{p['nom']}")
    print(f"  Age        : {p['age']} ans | Sexe : {p['sexe']}")
    print(f"  Temperature: {p['temperature']}°C")
    print(f"  Diagnostic : {diag} ({proba:.1%})")