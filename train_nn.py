"""
train_nn.py
NumPyのみで実装したニューラルネットワーク(neural_network.py)を使い、
恒星カテゴリの分類器を学習する。data_prep.pyで作った
k-NNと全く同じtrain/testスプリットを使用する。
"""
import numpy as np
from data_prep import load_dataset, CLASSES
from neural_network import NeuralNetwork

EPOCHS = 300
BATCH_SIZE = 32
LR = 5e-3
HIDDEN = [32, 16]
SEED = 42


def one_hot(y, n_classes):
    oh = np.zeros((len(y), n_classes))
    oh[np.arange(len(y)), y] = 1
    return oh


def main():
    d = load_dataset()
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test = d["X_test"], d["y_test"]
    n_classes = len(CLASSES)
    Y_train_oh = one_hot(y_train, n_classes)

    net = NeuralNetwork([X_train.shape[1]] + HIDDEN + [n_classes], seed=SEED)
    rng = np.random.default_rng(SEED)

    n = X_train.shape[0]
    loss_history, train_acc_history, test_acc_history = [], [], []

    for epoch in range(1, EPOCHS + 1):
        perm = rng.permutation(n)
        Xs, Ys = X_train[perm], Y_train_oh[perm]
        epoch_loss = 0.0
        n_batches = max(1, n // BATCH_SIZE)
        for b in range(n_batches):
            xb = Xs[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            yb = Ys[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            if len(xb) == 0:
                continue
            epoch_loss += net.train_step(xb, yb, lr=LR)
        epoch_loss /= n_batches

        train_pred = net.predict(X_train)
        test_pred = net.predict(X_test)
        train_acc = np.mean(train_pred == y_train)
        test_acc = np.mean(test_pred == y_test)

        loss_history.append(epoch_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    final_pred = net.predict(X_test)
    final_acc = np.mean(final_pred == y_test)
    print(f"\n最終テスト正解率: {final_acc:.4f} ({np.sum(final_pred==y_test)}/{len(y_test)})")

    np.savez("results/nn_history.npz",
             loss=loss_history, train_acc=train_acc_history, test_acc=test_acc_history)
    np.save("results/nn_pred.npy", final_pred)
    with open("results/nn_accuracy.txt", "w") as f:
        f.write(f"{final_acc:.4f}\n")

    # 混同行列
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_test, final_pred):
        cm[t, p] += 1
    np.save("results/nn_confusion.npy", cm)
    print("\n混同行列 (行=正解, 列=予測):")
    print("\t" + "\t".join(c[:6] for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(c[:10] + "\t" + "\t".join(str(v) for v in cm[i]))


if __name__ == "__main__":
    main()
