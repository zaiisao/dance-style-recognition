import os
import glob
import re
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Mapping from AIST++ filename codes to full Genre names [cite: 73]
GENRE_MAP = {
    'BR': 'Break', 'HO': 'House', 'JB': 'Jazz Ballet', 'JS': 'Jazz Street',
    'KR': 'Krump', 'LH': 'LA Hip Hop', 'LO': 'Lock', 'MH': 'Middle Hip Hop',
    'PO': 'Pop', 'WA': 'Waack'
}

def _dict_to_matrix(d):
    """
    Convert a dictionary of 1D arrays (same length) to a (T, F) matrix.
    If values are already arrays of shape (T,), they become columns.
    """
    keys = sorted(d.keys())
    cols = []
    # Filter keys that are arrays and share the same max length
    lengths = {k: (np.asarray(d[k]).shape[0] if hasattr(d[k], '__len__') else 0) for k in keys}
    max_len = max(lengths.values()) if lengths else 0
    keys = [k for k in keys if lengths[k] == max_len and max_len > 0]

    for k in keys:
        arr = np.asarray(d[k])
        if arr.ndim == 1:
            cols.append(arr)
        elif arr.ndim == 2 and arr.shape[1] == 1:
            cols.append(arr[:, 0])
        else:
            # flatten multi-column entries into separate columns
            for i in range(arr.shape[1]):
                cols.append(arr[:, i])
    if not cols:
        raise ValueError("No valid feature arrays found in dict.")
    return np.stack(cols, axis=1)

def load_dataset(input_dir):
    """
    Robust loading of features.
    - Handles .npy dictionaries or arrays.
    - Filters by AIST++ genre codes (gBR, gHO, etc.) in filenames.
    """
    X_list = []
    y_list = []
    groups_list = []

    search_path = os.path.join(input_dir, "*.npy")
    all_npy_files = sorted(glob.glob(search_path))

    if not all_npy_files:
        print(f"[!] No .npy files found in {input_dir}")
        return None, None, None

    print(f"Scanning {len(all_npy_files)} files...")

    # Regex to find 'gXX' where XX is a known genre code
    genre_codes = "|".join(GENRE_MAP.keys())
    genre_regex = re.compile(rf"g({genre_codes})", re.IGNORECASE)

    for f_path in all_npy_files:
        filename = os.path.basename(f_path)

        # strict filter for feature files only (case-insensitive)
        if not filename.lower().endswith("_features.npy"):
            continue

        # Detect Genre
        match = genre_regex.search(filename)
        if not match:
            print(f"  [?] Skipping (no genre token): {filename}")
            continue

        style_code = match.group(1).upper()
        if style_code not in GENRE_MAP:
            print(f"  [?] Unknown genre code in filename: {filename}")
            continue
        label = GENRE_MAP[style_code]

        try:
            raw = np.load(f_path, allow_pickle=True)

            # unwrap 0-d object arrays that contain a dict
            if isinstance(raw, np.ndarray) and raw.dtype == object and raw.ndim == 0:
                raw = raw.item()

            if isinstance(raw, dict):
                data = _dict_to_matrix(raw)
            else:
                data = np.asarray(raw)

            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if data.ndim != 2:
                print(f"  [!] Unsupported data shape for {filename}: {data.shape}. Skipping.")
                continue

            if data.shape[1] != 55:
                print(f"  [!] Warning: {filename} has {data.shape[1]} features (expected 55). Proceeding anyway.")

            video_id = filename.replace('_features.npy', '')

            X_list.append(data)
            y_list.append(np.array([label] * data.shape[0], dtype=object))
            groups_list.append(np.array([video_id] * data.shape[0], dtype=object))

        except Exception as e:
            print(f"  [!] Error reading {filename}: {e}")

    if not X_list:
        print("[!] No valid data loaded.")
        return None, None, None

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(groups_list)

    print(f"Loaded {X.shape[0]} frames from {len(np.unique(groups))} videos.")
    print(f"Feature dimension: {X.shape[1]}")
    return X, y, groups

def train_and_evaluate(X, y, groups, save_model_path=None):
    """
    Faithful Implementation of Training Protocol[cite: 129]:
      - Outer: 3-Fold GroupKFold (Held-out Test)
      - Inner: GridSearchCV with GroupKFold (Held-out Validation)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    gkf_outer = GroupKFold(n_splits=3)
    fold_accuracies = []

    print("\n" + "="*60)
    print("Starting Faithful Evaluation (3-Fold Group CV + GridSearch)")
    print("="*60)

    for fold, (train_idx, test_idx) in enumerate(gkf_outer.split(X, y, groups)):
        print(f"\nProcessing Fold {fold + 1}/3...")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        # Inner GridSearch with GroupKFold
        # This prevents video leakage during hyperparameter tuning
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [300, 500],
            'max_depth': [None, 20],
            'min_samples_split': [2, 5]
        }

        inner_cv = GroupKFold(n_splits=3)
        grid = GridSearchCV(rf, param_grid, cv=inner_cv, n_jobs=-1, verbose=1)
        # pass groups for inner group-aware splitting
        grid.fit(X_train, y_train, groups=groups_train)

        best_model = grid.best_estimator_
        print(f"  > Best Params: {grid.best_params_}")

        # Evaluate on outer test set
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)

        print(f"  > Fold Accuracy: {acc:.4f}")
        print("  > Classification Report:")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))

        # Optionally save the best model per fold (append fold id)
        if save_model_path:
            fname = os.path.join(save_model_path, f"best_model_fold{fold+1}.joblib")
            joblib.dump(best_model, fname)
            print(f"  > Saved model to: {fname}")

    print("\n" + "="*60)
    print(f"FINAL AVERAGE ACCURACY: {np.mean(fold_accuracies):.4f}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_models", type=str, default=None,
                        help="Optional folder to save best models per fold")
    args = parser.parse_args()

    X, y, groups = load_dataset(args.data_dir)
    if X is not None:
        if args.save_models:
            os.makedirs(args.save_models, exist_ok=True)
        train_and_evaluate(X, y, groups, save_model_path=args.save_models)