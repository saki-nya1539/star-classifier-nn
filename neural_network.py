"""
neural_network.py
NumPyのみを用いた全結合ニューラルネットワーク(多クラス分類版)の自前実装。

- 隠れ層の活性化関数: tanh
- 出力層: softmax
- 損失関数: 交差エントロピー(cross-entropy)
- 最適化: ミニバッチAdam

PyTorch/TensorFlow等の自動微分フレームワークは使用せず、
順伝播・逆伝播の勾配計算式をすべて手で導出してNumPy配列演算のみで実装している。
(neural_pendulumプロジェクトの回帰用NN実装を、多クラス分類向けに拡張したもの)
"""
import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_grad(a):
    return 1.0 - a ** 2


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)  # オーバーフロー対策
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


class NeuralNetwork:
    """
    layer_sizes = [入力次元, 隠れ層1, ..., 出力次元(=クラス数)]
    出力層はsoftmax、損失は交差エントロピー(多クラス分類用)。
    """

    def __init__(self, layer_sizes, seed=0):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        rng = np.random.default_rng(seed)

        self.W, self.b = [], []
        for i in range(self.n_layers):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.b.append(np.zeros((1, fan_out)))

        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, X):
        self.z, self.a = [], [X]
        for i in range(self.n_layers):
            z = self.a[-1] @ self.W[i] + self.b[i]
            self.z.append(z)
            if i < self.n_layers - 1:
                a = tanh(z)
            else:
                a = softmax(z)
            self.a.append(a)
        return self.a[-1]

    def backward(self, Y_onehot):
        """
        softmax + 交差エントロピーの組み合わせは、出力層の誤差が
        (予測確率 - 正解one-hot) というシンプルな形になる(標準的な結果)。
        """
        m = Y_onehot.shape[0]
        dW, db = [None] * self.n_layers, [None] * self.n_layers

        delta = (self.a[-1] - Y_onehot) / m

        for i in reversed(range(self.n_layers)):
            dW[i] = self.a[i].T @ delta
            db[i] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                da_prev = delta @ self.W[i].T
                delta = da_prev * tanh_grad(self.a[i])
        return dW, db

    def adam_step(self, dW, db, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for i in range(self.n_layers):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * dW[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (dW[i] ** 2)
            mW_hat = self.mW[i] / (1 - beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - beta2 ** self.t)
            self.W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * db[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (db[i] ** 2)
            mb_hat = self.mb[i] / (1 - beta1 ** self.t)
            vb_hat = self.vb[i] / (1 - beta2 ** self.t)
            self.b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def train_step(self, X_batch, Y_onehot_batch, lr=1e-3):
        pred = self.forward(X_batch)
        dW, db = self.backward(Y_onehot_batch)
        self.adam_step(dW, db, lr=lr)
        eps = 1e-12
        loss = -np.mean(np.sum(Y_onehot_batch * np.log(pred + eps), axis=1))
        return loss

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
