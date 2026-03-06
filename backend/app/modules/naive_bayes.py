"""
ChildFocus - Naïve Bayes Metadata Classifier
backend/app/modules/naive_bayes.py

What this does:
  - Loads the trained nb_model.pkl and vectorizer.pkl
  - Exposes score_metadata() → returns Score_NB between 0.0 and 1.0
  - Score_NB represents the probability that the video is Overstimulating
    based on its title, tags, and description
  - Used by hybrid_fusion.py to compute the final OIR score

Thesis reference:
  Score_NB = (1/Z) × [log P(C_over) + Σ log P(token | C_over)]
  Normalized using logistic transformation → Score_NB ∈ [0, 1]
"""

import os
import pickle
import re
import numpy as np

# ── Model paths ────────────────────────────────────────────────────────────────
_MODULE_DIR  = os.path.dirname(__file__)
_MODELS_DIR  = os.path.join(_MODULE_DIR, "..", "models")
_MODEL_PATH  = os.path.join(_MODELS_DIR, "nb_model.pkl")
_VEC_PATH    = os.path.join(_MODELS_DIR, "vectorizer.pkl")

# ── Lazy-loaded globals (loaded once on first call) ────────────────────────────
_model         = None
_vectorizer    = None
_label_encoder = None
_label_names   = None
_OVER_IDX      = None   # index of "Overstimulating" in label_encoder.classes_


def _load_models():
    """Load model and vectorizer from disk. Called once on first use."""
    global _model, _vectorizer, _label_encoder, _label_names, _OVER_IDX

    if _model is not None:
        return True   # already loaded

    if not os.path.exists(_MODEL_PATH):
        print(f"[NB] ✗ Model not found at {_MODEL_PATH}. Run train_nb.py first.")
        return False

    if not os.path.exists(_VEC_PATH):
        print(f"[NB] ✗ Vectorizer not found at {_VEC_PATH}. Run preprocess.py first.")
        return False

    try:
        with open(_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)

        # Support both plain model and bundled dict (from train_nb.py)
        if isinstance(bundle, dict):
            _model         = bundle["model"]
            _label_encoder = bundle["label_encoder"]
            _label_names   = bundle.get("label_names", list(_label_encoder.classes_))
        else:
            # Fallback: plain model object
            _model = bundle
            from sklearn.preprocessing import LabelEncoder
            _label_encoder = LabelEncoder()
            _label_encoder.fit(["Educational", "Neutral", "Overstimulating"])
            _label_names = list(_label_encoder.classes_)

        with open(_VEC_PATH, "rb") as f:
            _vectorizer = pickle.load(f)

        # Find index of "Overstimulating" class
        classes = list(_label_encoder.classes_)
        _OVER_IDX = classes.index("Overstimulating") if "Overstimulating" in classes else -1

        print(f"[NB] ✓ Model loaded. Classes: {classes}")
        return True

    except Exception as e:
        print(f"[NB] ✗ Failed to load model: {e}")
        return False


def _clean_text(text: str) -> str:
    """Same cleaning as preprocess.py — must match exactly."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _logistic(x: float) -> float:
    """Logistic/sigmoid normalization: maps any real number to (0, 1)."""
    return float(1.0 / (1.0 + np.exp(-x)))


def score_metadata(title: str = "", tags: list = None, description: str = "") -> dict:
    """
    Compute Score_NB for a video's metadata.

    Args:
        title:       Video title string
        tags:        List of tag strings (optional)
        description: Video description string (truncated to 500 chars)

    Returns:
        dict with:
            score_nb (float):  Overstimulation probability [0.0, 1.0]
            label (str):       Predicted class: Educational / Neutral / Overstimulating
            confidence (float): Max class probability
            probabilities (dict): Per-class probabilities
            status (str):      "success" or "error"
    """
    if not _load_models():
        return {
            "score_nb":      0.5,
            "label":         "Uncertain",
            "confidence":    0.0,
            "probabilities": {},
            "status":        "model_not_loaded",
        }

    # Build combined text — same as preprocess.py
    tags_str = " ".join(tags) if tags else ""
    combined = _clean_text(f"{title} {description} {tags_str}")

    if not combined.strip():
        return {
            "score_nb":      0.5,
            "label":         "Uncertain",
            "confidence":    0.0,
            "probabilities": {},
            "status":        "empty_text",
        }

    try:
        # Vectorize
        X = _vectorizer.transform([combined])

        # Get class probabilities
        proba = _model.predict_proba(X)[0]

        # Map to class names
        classes      = list(_label_encoder.classes_)
        proba_dict   = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

        # Score_NB = probability of "Overstimulating" class
        score_nb = float(proba[_OVER_IDX]) if _OVER_IDX >= 0 else 0.5

        # Predicted label
        pred_idx = int(np.argmax(proba))
        label    = classes[pred_idx]

        return {
            "score_nb":      round(score_nb, 4),
            "label":         label,
            "confidence":    round(float(np.max(proba)), 4),
            "probabilities": proba_dict,
            "status":        "success",
        }

    except Exception as e:
        print(f"[NB] ✗ Scoring error: {e}")
        return {
            "score_nb":      0.5,
            "label":         "Uncertain",
            "confidence":    0.0,
            "probabilities": {},
            "status":        f"error: {e}",
        }


def get_model_metrics() -> dict:
    """Returns the training metrics saved with the model (for API transparency)."""
    if not _load_models():
        return {}
    if isinstance(_model, dict):
        return {}
    # Try to get from bundle
    try:
        with open(_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict):
            return bundle.get("metrics", {})
    except Exception:
        pass
    return {}