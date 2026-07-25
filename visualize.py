"""
visualize.py
学習結果の可視化。

出力:
  results/learning_curve.png      学習曲線(損失, train/test accuracy)
  results/hr_diagram.png          HR図(温度 vs 光度)にNNの予測結果を重ねた散布図
  results/confusion_matrices.png  k-NNとNNの混同行列を並べて比較
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_JP_FONT_CANDIDATES = [
    "Yu Gothic", "Meiryo", "MS Gothic",
    "Hiragino Sans", "Hiragino Kaku Gothic ProN",
    "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "TakaoGothic",
]
_available = {f.name for f in fm.fontManager.ttflist}
for _name in _JP_FONT_CANDIDATES:
    if _name in _available:
        matplotlib.rcParams["font.family"] = _name
        break
matplotlib.rcParams["axes.unicode_minus"] = False

from data_prep import load_dataset, CLASSES

COLORS = plt.cm.tab10(np.linspace(0, 1, len(CLASSES)))


def plot_learning_curve():
    h = np.load("results/nn_history.npz")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(h["loss"])
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("交差エントロピー損失")
    axes[0].set_title("学習曲線: 損失")

    axes[1].plot(h["train_acc"], label="train")
    axes[1].plot(h["test_acc"], label="test")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("正解率")
    axes[1].set_title("学習曲線: 正解率")
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("results/learning_curve.png", dpi=130)
    plt.close()
    print("保存: results/learning_curve.png")


def plot_hr_diagram():
    d = load_dataset()
    X_raw = np.concatenate([d["X_train_raw"], d["X_test_raw"]], axis=0)
    y = np.concatenate([d["y_train"], d["y_test"]], axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(len(CLASSES)):
        mask = y == c
        ax.scatter(X_raw[mask, 0], X_raw[mask, 1], label=CLASSES[c], color=COLORS[c], s=25, alpha=0.8)
    ax.set_xlabel("温度 (K)")
    ax.set_ylabel("log10(光度 L/Lo)")
    ax.set_title("HR図: 恒星カテゴリの分布(全240件)")
    ax.invert_xaxis()  # HR図の伝統的な向き(高温が左)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/hr_diagram.png", dpi=130)
    plt.close()
    print("保存: results/hr_diagram.png")


def plot_confusion_matrices():
    cm_knn = np.load("results/knn_confusion.npy")
    cm_nn = np.load("results/nn_confusion.npy")
    with open("results/knn_accuracy.txt") as f:
        acc_knn = float(f.read().strip())
    with open("results/nn_accuracy.txt") as f:
        acc_nn = float(f.read().strip())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, cm, title, acc in [
        (axes[0], cm_knn, f"k-NN (k=5)  正解率={acc_knn:.1%}", acc_knn),
        (axes[1], cm_nn, f"NN(自前実装)  正解率={acc_nn:.1%}", acc_nn),
    ]:
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels([c[:8] for c in CLASSES], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([c[:8] for c in CLASSES], fontsize=8)
        ax.set_xlabel("予測")
        ax.set_ylabel("正解")
        ax.set_title(title)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    plt.tight_layout()
    plt.savefig("results/confusion_matrices.png", dpi=130)
    plt.close()
    print("保存: results/confusion_matrices.png")


if __name__ == "__main__":
    plot_learning_curve()
    plot_hr_diagram()
    plot_confusion_matrices()
