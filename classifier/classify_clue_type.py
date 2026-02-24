"""
Clue Type Decision Tree Classifier

Trains a gradient boosting classifier on labeled cryptic crossword clues
to predict wordplay type from structural features + optional TF-IDF text features.

Usage:
    python classifier/classify_clue_type.py                              # Train + evaluate + save
    python classifier/classify_clue_type.py --model rf                   # Use RandomForest instead
    python classifier/classify_clue_type.py --tfidf                      # Add TF-IDF text features
    python classifier/classify_clue_type.py --predict "clue" ANSWER 7    # Predict single clue
"""

import argparse
import os
import re
import sqlite3
import sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from scipy.sparse import issparse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH = PROJECT_ROOT / "data" / "clues_master.db"
MODEL_PATH = SCRIPT_DIR / "clue_type_model.joblib"

# ---------------------------------------------------------------------------
# Target classes — types with >= 500 samples kept, rest grouped as "other"
# ---------------------------------------------------------------------------
MIN_CLASS_SAMPLES = 500

# Indicator type mapping: indicators table types → clue wordplay_type equivalents
INDICATOR_TYPE_MAP = {
    "anagram": "anagram",
    "container": "container",
    "insertion": "container",
    "reversal": "reversal",
    "deletion": "deletion",
    "homophone": "homophone",
    "hidden": "hidden",
    "acrostic": "acrostic",
    "parts": "charade",
    "selection": "deletion",
    "alternating": "deletion",
}

# The 7 indicator families we track as features
INDICATOR_FAMILIES = [
    "anagram", "container", "reversal", "deletion",
    "homophone", "hidden", "acrostic",
]


# ---------------------------------------------------------------------------
# Load indicators from DB into a lookup dict
# ---------------------------------------------------------------------------
def load_indicators(db_path):
    """Returns dict: family_name → set of lowercase indicator words."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT word, wordplay_type FROM indicators").fetchall()
    conn.close()

    indicators = {f: set() for f in INDICATOR_FAMILIES}
    for word, wtype in rows:
        family = INDICATOR_TYPE_MAP.get(wtype)
        if family and family in indicators:
            indicators[family].add(word.lower().strip())
    return indicators


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def parse_enumeration(enum_str):
    """Parse enumeration string like '5', '4,3', '2-4' into total char count."""
    if not enum_str:
        return 0
    return sum(int(n) for n in re.findall(r"\d+", enum_str))


def check_hidden_word(clue_text, answer):
    """Check if answer appears as a hidden substring in consecutive clue words."""
    if not answer or len(answer) < 3:
        return False
    # Strip punctuation, join words, search for answer
    words = re.findall(r"[a-zA-Z]+", clue_text)
    # Try all consecutive subsequences of 2..len(words) words
    for start in range(len(words)):
        concat = ""
        for end in range(start, min(start + 8, len(words))):
            concat += words[end].upper()
            if len(concat) >= len(answer) + 2:  # answer must be properly hidden
                # Check substring (not at very start/end of single word)
                if answer.upper() in concat and concat != answer.upper():
                    # Verify it spans at least 2 words
                    if end > start:
                        return True
    return False


def check_consecutive_anagram(clue_text, answer):
    """Check if any consecutive sequence of clue words is an anagram of the answer."""
    if not answer or len(answer) < 3:
        return False
    answer_sorted = sorted(answer.upper().replace(" ", "").replace("-", ""))
    answer_len = len(answer_sorted)
    words = re.findall(r"[a-zA-Z]+", clue_text)

    for start in range(len(words)):
        concat = ""
        for end in range(start, min(start + 6, len(words))):
            concat += words[end].upper()
            if len(concat) == answer_len:
                if sorted(concat) == answer_sorted:
                    return True
                break  # longer won't match
            elif len(concat) > answer_len:
                break
    return False


def check_acrostic_match(clue_text, answer):
    """Check if first letters of consecutive words spell the answer."""
    if not answer or len(answer) < 3:
        return False
    words = re.findall(r"[a-zA-Z]+", clue_text)
    answer_upper = answer.upper().replace(" ", "").replace("-", "")
    n = len(answer_upper)

    if len(words) < n:
        return False

    for start in range(len(words) - n + 1):
        initials = "".join(w[0].upper() for w in words[start:start + n])
        if initials == answer_upper:
            return True
    return False


def find_indicators_in_clue(clue_words_lower, indicators_by_family):
    """
    For each indicator family, find count and earliest position (normalised 0-1).
    Returns (counts_dict, positions_dict).
    """
    n_words = len(clue_words_lower)
    counts = {f: 0 for f in INDICATOR_FAMILIES}
    positions = {f: -1.0 for f in INDICATOR_FAMILIES}  # -1 = not found

    for family, ind_set in indicators_by_family.items():
        # Check single-word indicators
        for i, w in enumerate(clue_words_lower):
            if w in ind_set:
                counts[family] += 1
                if positions[family] < 0:
                    positions[family] = i / max(n_words - 1, 1)
        # Check two-word indicators
        for i in range(len(clue_words_lower) - 1):
            bigram = clue_words_lower[i] + " " + clue_words_lower[i + 1]
            if bigram in ind_set:
                counts[family] += 1
                if positions[family] < 0:
                    positions[family] = i / max(n_words - 1, 1)

    return counts, positions


def extract_features(clue_text, answer, enumeration, indicators_by_family):
    """
    Extract 24 features from a single clue.

    Returns a list of 24 numeric values:
      [0]  clue_word_count
      [1]  answer_char_length
      [2]  answer_clue_char_ratio
      [3]  trailing_question_mark
      [4]  comma_count
      [5]  capitalized_word_count
      [6]  multi_word_answer
      [7]  hidden_word_found
      [8]  consecutive_anagram_found
      [9]  acrostic_match_found
      [10-16] indicator_count per family (x7)
      [17-23] indicator_position per family (x7)
    """
    # Basic text features
    clue_clean = clue_text.strip()
    words = re.findall(r"[a-zA-Z]+", clue_clean)
    clue_words_lower = [w.lower() for w in words]
    n_words = len(words)

    answer_clean = answer.upper().replace(" ", "").replace("-", "")
    answer_len = len(answer_clean)

    clue_alpha_len = sum(1 for c in clue_clean if c.isalpha())
    ratio = answer_len / max(clue_alpha_len, 1)

    trailing_q = 1 if clue_clean.endswith("?") else 0
    comma_count = clue_clean.count(",")

    # Capitalised words (skip first word which is always capitalised)
    cap_count = 0
    for w in words[1:]:
        if w[0].isupper():
            cap_count += 1

    # Multi-word answer (from enumeration)
    enum_parts = re.findall(r"\d+", enumeration) if enumeration else []
    multi_word = 1 if len(enum_parts) > 1 else 0

    # Pattern match features
    hidden = 1 if check_hidden_word(clue_text, answer_clean) else 0
    anagram = 1 if check_consecutive_anagram(clue_text, answer_clean) else 0
    acrostic = 1 if check_acrostic_match(clue_text, answer_clean) else 0

    # Indicator features
    ind_counts, ind_positions = find_indicators_in_clue(clue_words_lower, indicators_by_family)

    features = [
        n_words,            # 0
        answer_len,         # 1
        ratio,              # 2
        trailing_q,         # 3
        comma_count,        # 4
        cap_count,          # 5
        multi_word,         # 6
        hidden,             # 7
        anagram,            # 8
        acrostic,           # 9
    ]
    # 10-16: indicator counts
    for f in INDICATOR_FAMILIES:
        features.append(ind_counts[f])
    # 17-23: indicator positions
    for f in INDICATOR_FAMILIES:
        features.append(ind_positions[f])

    return features


FEATURE_NAMES = [
    "clue_word_count",
    "answer_char_length",
    "answer_clue_char_ratio",
    "trailing_question_mark",
    "comma_count",
    "capitalized_word_count",
    "multi_word_answer",
    "hidden_word_found",
    "consecutive_anagram_found",
    "acrostic_match_found",
] + [f"indicator_count_{f}" for f in INDICATOR_FAMILIES] + \
    [f"indicator_position_{f}" for f in INDICATOR_FAMILIES]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_training_data(db_path, indicators_by_family):
    """Load labeled clues, extract features, return X, y, class_names, clue_texts."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT clue_text, answer, enumeration, wordplay_type "
        "FROM clues WHERE wordplay_type IS NOT NULL AND wordplay_type != ''"
    ).fetchall()
    conn.close()

    print(f"Loaded {len(rows):,} labeled clues from {db_path.name}")

    # Count class frequencies
    type_counts = {}
    for _, _, _, wtype in rows:
        type_counts[wtype] = type_counts.get(wtype, 0) + 1

    # Determine which classes to keep vs group as "other"
    keep_classes = {t for t, c in type_counts.items() if c >= MIN_CLASS_SAMPLES}
    print(f"Keeping {len(keep_classes)} classes with >= {MIN_CLASS_SAMPLES} samples:")
    for t in sorted(keep_classes):
        print(f"  {t}: {type_counts[t]:,}")

    grouped = sum(c for t, c in type_counts.items() if t not in keep_classes)
    if grouped:
        print(f"  other: {grouped:,} (from {len(type_counts) - len(keep_classes)} rare types)")

    # Extract features
    X = []
    y = []
    clue_texts = []
    skipped = 0
    for clue_text, answer, enum_str, wtype in rows:
        if not clue_text or not answer:
            skipped += 1
            continue
        label = wtype if wtype in keep_classes else "other"
        features = extract_features(clue_text, answer, enum_str or "", indicators_by_family)
        X.append(features)
        y.append(label)
        clue_texts.append(clue_text)

    if skipped:
        print(f"Skipped {skipped} rows with missing clue_text/answer")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    class_names = sorted(set(y))
    print(f"Feature matrix: {X.shape[0]:,} samples x {X.shape[1]} features")
    print(f"Classes: {class_names}")
    return X, y, class_names, clue_texts


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def compute_sample_weights(y):
    """Compute balanced sample weights inversely proportional to class frequency."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[label] for label in y])


def build_tfidf_features(clue_texts, vectorizer=None, max_features=300):
    """
    Build TF-IDF features from clue texts.
    If vectorizer is None, fit a new one; otherwise transform with existing.
    Returns (tfidf_matrix_dense, vectorizer, feature_names).
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8,
            sublinear_tf=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",  # words only, 2+ chars
        )
        tfidf_matrix = vectorizer.fit_transform(clue_texts)
    else:
        tfidf_matrix = vectorizer.transform(clue_texts)

    # Convert to dense for tree-based models
    tfidf_dense = tfidf_matrix.toarray().astype(np.float32)
    feature_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    return tfidf_dense, vectorizer, feature_names


def train_and_evaluate(X, y, class_names, model_type="hgb",
                       clue_texts=None, use_tfidf=False):
    """Train classifier, run CV, evaluate on test set, return model + extras."""

    tfidf_vectorizer = None
    tfidf_feature_names = []

    # Stratified train/test split (split indices first, then apply to all arrays)
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    # Add TF-IDF features if requested
    if use_tfidf and clue_texts is not None:
        texts_train = [clue_texts[i] for i in idx_train]
        texts_test = [clue_texts[i] for i in idx_test]

        print("Building TF-IDF features...")
        tfidf_train, tfidf_vectorizer, tfidf_feature_names = build_tfidf_features(texts_train)
        tfidf_test, _, _ = build_tfidf_features(texts_test, vectorizer=tfidf_vectorizer)

        print(f"  TF-IDF features: {tfidf_train.shape[1]}")
        X_train = np.hstack([X_train, tfidf_train])
        X_test = np.hstack([X_test, tfidf_test])
        print(f"  Combined features: {X_train.shape[1]}")

    all_feature_names = FEATURE_NAMES + tfidf_feature_names
    print(f"\nTrain: {len(X_train):,}  Test: {len(X_test):,}")

    # Sample weights for balanced training
    sample_weights = compute_sample_weights(y_train)

    # Build model
    if model_type == "rf":
        print("Training RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        print("Training HistGradientBoostingClassifier...")
        model = HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=8,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42,
        )

    # 5-fold stratified cross-validation
    print("\nRunning 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if model_type == "rf":
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Per-fold: {', '.join(f'{s:.4f}' for s in cv_scores)}")

    # Train on full training set
    print("\nTraining final model on full training set...")
    if model_type == "rf":
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # Classification report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=class_names, digits=3, zero_division=0))

    # Confusion matrix
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    # Header
    label_width = max(len(c) for c in class_names)
    header = " " * (label_width + 2) + "  ".join(f"{c[:6]:>6}" for c in class_names)
    print(header)
    for i, row_label in enumerate(class_names):
        row_str = f"{row_label:<{label_width}}  " + "  ".join(f"{v:>6}" for v in cm[i])
        print(row_str)

    # Feature importances
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCES (top 30)")
    print("=" * 70)

    if model_type == "rf":
        importances = model.feature_importances_
    else:
        print("Computing permutation importance (5000-sample subsample, 5 repeats)...")
        n_sub = min(5000, len(X_test))
        rng = np.random.RandomState(42)
        idx_sub = rng.choice(len(X_test), n_sub, replace=False)
        result = permutation_importance(
            model, X_test[idx_sub], y_test[idx_sub],
            n_repeats=5, random_state=42, n_jobs=-1, scoring="accuracy"
        )
        importances = result.importances_mean

    sorted_idx = np.argsort(importances)[::-1]
    max_imp = max(importances) if max(importances) > 0 else 1
    n_show = min(30, len(all_feature_names))
    for rank, idx in enumerate(sorted_idx[:n_show], 1):
        bar = "#" * int(importances[idx] / max_imp * 40)
        name = all_feature_names[idx] if idx < len(all_feature_names) else f"feature_{idx}"
        print(f"  {rank:>2}. {name:<40} {importances[idx]:.4f}  {bar}")

    return model, tfidf_vectorizer, all_feature_names


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_single(clue_text, answer, enum_val, model_bundle, indicators_by_family):
    """Predict wordplay type for a single clue."""
    model = model_bundle["model"]
    class_names = model_bundle["class_names"]
    tfidf_vectorizer = model_bundle.get("tfidf_vectorizer")

    enum_str = str(enum_val) if enum_val else ""
    features = extract_features(clue_text, answer, enum_str, indicators_by_family)
    X = np.array([features], dtype=np.float32)

    # Add TF-IDF features if model was trained with them
    if tfidf_vectorizer is not None:
        tfidf_feat, _, _ = build_tfidf_features([clue_text], vectorizer=tfidf_vectorizer)
        X = np.hstack([X, tfidf_feat])

    prediction = model.predict(X)[0]

    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        sorted_idx = np.argsort(proba)[::-1]
        print(f"\nClue: \"{clue_text}\"")
        print(f"Answer: {answer} ({enum_str})")
        print(f"\nPrediction: {prediction}")
        print(f"\nAll probabilities:")
        for idx in sorted_idx:
            pct = proba[idx] * 100
            bar = "#" * int(pct)
            print(f"  {class_names[idx]:<25} {pct:>5.1f}%  {bar}")
    else:
        print(f"\nClue: \"{clue_text}\"")
        print(f"Answer: {answer} ({enum_str})")
        print(f"Prediction: {prediction}")

    return prediction


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cryptic clue type classifier")
    parser.add_argument("--model", choices=["hgb", "rf"], default="hgb",
                        help="Model type: hgb (HistGradientBoosting) or rf (RandomForest)")
    parser.add_argument("--tfidf", action="store_true",
                        help="Add TF-IDF text features from clue words")
    parser.add_argument("--tfidf-features", type=int, default=300,
                        help="Number of TF-IDF features (default: 300)")
    parser.add_argument("--predict", nargs=3, metavar=("CLUE", "ANSWER", "ENUM"),
                        help="Predict wordplay type for a single clue")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help=f"Path to clues DB (default: {DB_PATH})")
    parser.add_argument("--model-path", type=str, default=str(MODEL_PATH),
                        help=f"Path to save/load model (default: {MODEL_PATH})")
    args = parser.parse_args()

    db_path = Path(args.db)
    model_path = Path(args.model_path)

    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        sys.exit(1)

    # Load indicators (needed for both training and prediction)
    print(f"Loading indicators from {db_path.name}...")
    indicators_by_family = load_indicators(db_path)
    for family, words in sorted(indicators_by_family.items()):
        print(f"  {family}: {len(words)} indicators")

    # Predict mode
    if args.predict:
        clue_text, answer, enum_val = args.predict
        if not model_path.exists():
            print(f"ERROR: No trained model found at {model_path}")
            print("Run without --predict first to train and save a model.")
            sys.exit(1)
        bundle = joblib.load(model_path)
        predict_single(clue_text, answer, enum_val, bundle, indicators_by_family)
        return

    # Training mode
    print("\n" + "=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)
    X, y, class_names, clue_texts = load_training_data(db_path, indicators_by_family)

    print("\n" + "=" * 70)
    print(f"TRAINING {'(with TF-IDF)' if args.tfidf else '(structural features only)'}")
    print("=" * 70)
    model, tfidf_vectorizer, all_feature_names = train_and_evaluate(
        X, y, class_names,
        model_type=args.model,
        clue_texts=clue_texts,
        use_tfidf=args.tfidf,
    )

    # Save model bundle
    bundle = {
        "model": model,
        "class_names": class_names,
        "feature_names": all_feature_names,
        "tfidf_vectorizer": tfidf_vectorizer,
    }
    joblib.dump(bundle, model_path)
    print(f"\nModel saved to {model_path}")
    n_feat = len(all_feature_names)
    tfidf_str = f", TF-IDF vectorizer ({tfidf_vectorizer is not None})" if args.tfidf else ""
    print(f"Bundle contains: model, {len(class_names)} class names, {n_feat} feature names{tfidf_str}")


if __name__ == "__main__":
    main()
