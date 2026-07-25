"""
data_prep.py
恒星データ(stars.csv, 240件)の読み込みと前処理。

特徴量: Temperature(K), log10(Luminosity), log10(Radius), Absolute magnitude(Mv)
目的変数: Star category (Brown Dwarf / Red Dwarf / White Dwarf / Main Sequence / Supergiant / Hypergiant)

前回のJava版(k-NN, StarClassifier.java)と同じ特徴量・同じ層化80/20分割(seed=42)を
再現することで、k-NNとNNを同一条件で公平に比較できるようにしている。
"""
import numpy as np
import csv

FEATURE_COLS = ["Temperature (K)", "Luminosity (L/Lo)", "Radius (R/Ro)", "Absolute magnitude (Mv)"]
TARGET_COL = "Star category"
CLASSES = ["Brown Dwarf", "Red Dwarf", "White Dwarf", "Main Sequence", "Supergiant", "Hypergiant"]
SEED = 42
TEST_RATIO = 0.2


def load_raw(path="data/stars.csv"):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    T = np.array([float(r["Temperature (K)"]) for r in rows])
    L = np.array([float(r["Luminosity (L/Lo)"]) for r in rows])
    R = np.array([float(r["Radius (R/Ro)"]) for r in rows])
    Mv = np.array([float(r["Absolute magnitude (Mv)"]) for r in rows])
    cat = np.array([r["Star category"] for r in rows])

    X = np.stack([T, np.log10(L), np.log10(R), Mv], axis=1)
    y = np.array([CLASSES.index(c) for c in cat])
    return X, y


def standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
    return (X - mean) / std, mean, std


def stratified_split(X, y, test_ratio=TEST_RATIO, seed=SEED):
    """クラスごとに層化して80/20に分割する(Java版と同じロジック)。"""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in range(len(CLASSES)):
        idx = np.where(y == c)[0]
        idx = idx.copy()
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx), np.array(test_idx)


def load_dataset(path="data/stars.csv"):
    X, y = load_raw(path)
    train_idx, test_idx = stratified_split(X, y)
    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]

    X_train, mean, std = standardize(X_train_raw)
    X_test, _, _ = standardize(X_test_raw, mean, std)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "X_train_raw": X_train_raw, "X_test_raw": X_test_raw,
        "mean": mean, "std": std,
    }


if __name__ == "__main__":
    d = load_dataset()
    print(f"訓練データ: {len(d['y_train'])}件, テストデータ: {len(d['y_test'])}件")
    print("クラス一覧:", CLASSES)
    print("訓練データのクラス内訳:", np.bincount(d["y_train"]))
    print("テストデータのクラス内訳:", np.bincount(d["y_test"]))
