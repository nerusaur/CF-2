import os
import pickle
import pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# --- Robust Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA = os.path.join(SCRIPT_DIR, "data", "processed", "train_clean.csv")
TEST_DATA = os.path.join(SCRIPT_DIR, "data", "processed", "test_clean.csv")
OUTPUTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "backend", "app", "models"))

def run_final_training():
    print("═══ Training Final Thesis Model (NB-Only) ═══")

    # 1. Load the fixed datasets
    if not os.path.exists(TRAIN_DATA) or not os.path.exists(TEST_DATA):
        print(f"Error: CSV files not found at:\n{TRAIN_DATA}\n{TEST_DATA}")
        return

    df_train = pd.read_csv(TRAIN_DATA)
    df_test = pd.read_csv(TEST_DATA)

    print(f"Loaded {len(df_train)} training samples.")
    print(f"Loaded {len(df_test)} test samples.")

    X_train = df_train['text'].fillna('')
    y_train = df_train['label']
    X_test = df_test['text'].fillna('')
    y_test = df_test['label']

    # 2. Encode Labels (For backend compatibility)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # 3. Vectorization (Max Features 5000)
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=1,
        ngram_range=(1, 1),
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 4. Training (Alpha=1.5)
    print("Training ComplementNB...")
    clf = ComplementNB(alpha=1.5)
    clf.fit(X_train_tfidf, y_train_encoded)

    # 5. Evaluation (DETAILED RESULTS FOR CHAPTER 5)
    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test_encoded, y_pred)
    
    print("\n" + "="*55)
    print(f"NAÏVE BAYES HOLDOUT ACCURACY: {acc:.2%}")
    print("="*55)
    print("\n[Standalone NB] Detailed Classification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_, digits=4))
    print("="*55)

    # 6. Save the outputs
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    model_path = os.path.join(OUTPUTS_DIR, "nb_model.pkl")
    model_data = {
        'model': clf,
        'vectorizer': vectorizer,
        'label_encoder': le,
        'label_names': list(le.classes_),
        'metrics': {
            'accuracy': acc,
            'alpha': 1.5,
            'train_size': len(df_train)
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n✅ Success! New 'nb_model.pkl' saved to {OUTPUTS_DIR}")

if __name__ == "__main__":
    run_final_training()