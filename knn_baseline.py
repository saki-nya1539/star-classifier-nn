"""
knn_baseline.py
k近傍法(k-NN)による恒星分類。以前のJava実装(StarClassifier.java)と同じロジック
(標準化特徴量 + ユークリッド距離 + 多数決, k=5)をPythonで再実装したもの。

NN分類器(train_nn.py)と全く同じtrain/testスプリット(data_prep.py)を使うことで、
両手法をフェアな条件で比較できるようにしている。
"""
import numpy as np
from data_prep import load_dataset, CLASSES


def knn_predict(X_train, y_train, X_query, k=5):
    preds = np.zeros(len(X_query), dtype=int)
    for i, x in enumerate(X_query):
        dists = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        nearest = np.argsort(dists)[:k]
        votes = y_train[nearest]
        counts = np.bincount(votes, minlength=len(CLASSES))
        preds[i] = np.argmax(counts)
    return preds


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main():
    d = load_dataset()
    k = 5
    pred = knn_predict(d["X_train"], d["y_train"], d["X_test"], k=k)
    acc = np.mean(pred == d["y_test"])
    print(f"k-NN (k={k}) 正解率: {acc:.4f} ({np.sum(pred == d['y_test'])}/{len(d['y_test'])})")

    cm = confusion_matrix(d["y_test"], pred, len(CLASSES))
    print("\n混同行列 (行=正解, 列=予測):")
    print("\t" + "\t".join(c[:6] for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(c[:10] + "\t" + "\t".join(str(v) for v in cm[i]))

    np.save("results/knn_confusion.npy", cm)
    np.save("results/knn_pred.npy", pred)
    with open("results/knn_accuracy.txt", "w") as f:
        f.write(f"{acc:.4f}\n")


if __name__ == "__main__":
    main()
