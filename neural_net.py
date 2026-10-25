"""
Feed-forward neural network for binary classification, written from scratch
in NumPy (no autodiff / NN libraries).

Architecture:  input -> 1 or 2 hidden layers -> 1 sigmoid output unit
Loss:          binary cross-entropy
Hidden act.:   sigmoid | tanh | relu
Init:          Xavier/Glorot (sigmoid, tanh) or He (relu)
Optimizers:    plain mini-batch SGD or Adam

Shapes convention: a mini-batch X is (m, n_features); layer l has weights
W[l] of shape (n_in, n_out) and bias b[l] of shape (1, n_out), so every layer
is a single matrix product Z = A_prev @ W + b over the whole batch.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ----------------------------------------------------------------------------
# Activation functions and their derivatives (all element-wise, vectorized)
# ----------------------------------------------------------------------------

def sigmoid(z):
    # 0.5 * (1 + tanh(z/2)) equals 1 / (1 + e^-z) but never overflows.
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    return 1.0 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    # Sub-gradient: 0 at z == 0.
    return (z > 0).astype(z.dtype)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
    "relu": (relu, relu_grad),
}


# ----------------------------------------------------------------------------
# Optimizers. Each one updates the parameter list in place given the gradients.
# ----------------------------------------------------------------------------

class SGD:
    """Plain mini-batch gradient descent: theta <- theta - lr * grad."""

    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


class Adam:
    """Adam (Kingma & Ba, 2015).

    m <- beta1 * m + (1 - beta1) * g            (1st moment, like momentum)
    v <- beta2 * v + (1 - beta2) * g^2          (2nd moment, like RMSProp)
    m_hat = m / (1 - beta1^t),  v_hat = v / (1 - beta2^t)   (bias correction)
    theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:  # lazily create one moment buffer per parameter
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            p -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)


OPTIMIZERS = {"sgd": SGD, "adam": Adam}


# ----------------------------------------------------------------------------
# The network
# ----------------------------------------------------------------------------

class NeuralNetwork:
    """Binary classifier: preprocess -> train -> predict -> evaluate."""

    def __init__(self, hidden=(64,), activation="relu", optimizer="adam",
                 lr=0.001, epochs=50, batch_size=64, beta1=0.9, beta2=0.999,
                 eps=1e-8, threshold=0.5, seed=42):
        if not 1 <= len(hidden) <= 2:
            raise ValueError("hidden must list 1 or 2 layer sizes")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(ACTIVATIONS)}")
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {list(OPTIMIZERS)}")

        self.hidden = list(hidden)
        self.activation = activation
        self.act, self.act_grad = ACTIVATIONS[activation]
        self.optimizer_name = optimizer
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.adam_kwargs = dict(beta1=beta1, beta2=beta2, eps=eps)
        self.threshold = threshold  # P(>50K) at or above this -> predict class 1
        self.seed = seed
        # One generator drives both weight init and batch shuffling, so a
        # given seed reproduces a run exactly.
        self.rng = np.random.default_rng(seed)
        self.transformer = None
        self.W, self.b = [], []

    # ---------------------------------------------------------------- data --
    def preprocess(self, X_train, X_val, X_test):
        """Clean and encode the raw DataFrames; returns NumPy float arrays.

        * Missing values: Adult marks them '?'. In the categorical columns
          they become their own 'Unknown' category (no rows dropped).
          Numeric columns have no missing values in Adult, but a median
          imputer is included for safety.
        * Categorical features: one-hot encoded. Categories that only appear
          in val/test are ignored (all-zero row) rather than crashing.
        * Numeric features: standardised to zero mean / unit variance.

        Every imputer/encoder/scaler is fit on the TRAINING set only and then
        applied unchanged to validation and test, so no information leaks.
        """
        def clean(df):
            return df.replace("?", "Unknown")

        X_train, X_val, X_test = clean(X_train), clean(X_val), clean(X_test)

        numeric = X_train.select_dtypes(include="number").columns.tolist()
        categorical = [c for c in X_train.columns if c not in numeric]

        self.transformer = ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ])
        Xtr = self.transformer.fit_transform(X_train)  # fit on train only
        Xva = self.transformer.transform(X_val)
        Xte = self.transformer.transform(X_test)
        return Xtr.astype(np.float64), Xva.astype(np.float64), Xte.astype(np.float64)

    # -------------------------------------------------------------- params --
    def _init_params(self, n_in):
        """Xavier/Glorot for sigmoid & tanh, He for ReLU (normal distribution).

        Xavier: Var(W) = 2 / (fan_in + fan_out) keeps activation and gradient
                variance roughly constant for symmetric, ~linear-at-0 units.
        He:     Var(W) = 2 / fan_in compensates for ReLU zeroing half its inputs.
        The output layer is a sigmoid unit, so it always uses Xavier.
        Biases start at zero.
        """
        sizes = [n_in] + self.hidden + [1]
        self.W, self.b = [], []
        n_layers = len(sizes) - 1
        for l in range(n_layers):
            fan_in, fan_out = sizes[l], sizes[l + 1]
            is_output = (l == n_layers - 1)
            if self.activation == "relu" and not is_output:
                std = np.sqrt(2.0 / fan_in)
            else:
                std = np.sqrt(2.0 / (fan_in + fan_out))
            self.W.append(self.rng.normal(0.0, std, size=(fan_in, fan_out)))
            self.b.append(np.zeros((1, fan_out)))

    # ------------------------------------------------------ forward/backward --
    def _forward(self, X):
        """Returns the output logit z_out (m, 1) plus cached Z's and A's."""
        A = X
        Zs, As = [], [X]
        for l in range(len(self.W) - 1):          # hidden layers
            Z = A @ self.W[l] + self.b[l]
            A = self.act(Z)
            Zs.append(Z)
            As.append(A)
        z_out = A @ self.W[-1] + self.b[-1]       # output layer (logit)
        return z_out, Zs, As

    @staticmethod
    def _bce_from_logits(z, y):
        """Mean binary cross-entropy computed from logits (numerically stable).

        -[y log s(z) + (1-y) log(1-s(z))] == max(z,0) - z*y + log(1 + e^-|z|)
        """
        return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))

    def _backward(self, y, z_out, Zs, As):
        """Backpropagation; returns gradients in the same order as params.

        With a sigmoid output and BCE loss, dL/dz_out simplifies to
        (y_hat - y) / m. From there, for each layer going backwards:
            dW = A_prev^T @ dZ,   db = sum_rows(dZ),
            dA_prev = dZ @ W^T,   dZ_prev = dA_prev * act'(Z_prev)
        """
        m = y.shape[0]
        dZ = (sigmoid(z_out) - y) / m
        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        for l in range(len(self.W) - 1, -1, -1):
            dW[l] = As[l].T @ dZ
            db[l] = dZ.sum(axis=0, keepdims=True)
            if l > 0:  # propagate into the previous hidden layer
                dZ = (dZ @ self.W[l].T) * self.act_grad(Zs[l - 1])
        return dW, db

    def loss(self, X, y):
        z_out, _, _ = self._forward(X)
        return self._bce_from_logits(z_out, y.reshape(-1, 1))

    # ---------------------------------------------------------------- train --
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """Mini-batch training. Returns a per-epoch history dict.

        Each epoch the training set is reshuffled and walked through in
        mini-batches of size batch_size (the last batch may be smaller).
        After every epoch the loss/accuracy on the full train and validation
        sets is recorded for the learning curves.
        """
        y_train = y_train.reshape(-1, 1).astype(np.float64)
        self._init_params(X_train.shape[1])
        opt_cls = OPTIMIZERS[self.optimizer_name]
        opt = opt_cls(self.lr, **self.adam_kwargs) if opt_cls is Adam else opt_cls(self.lr)

        history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
        m = X_train.shape[0]
        for epoch in range(1, self.epochs + 1):
            perm = self.rng.permutation(m)
            for start in range(0, m, self.batch_size):
                idx = perm[start:start + self.batch_size]
                Xb, yb = X_train[idx], y_train[idx]
                z_out, Zs, As = self._forward(Xb)
                dW, db = self._backward(yb, z_out, Zs, As)
                opt.step(self.W + self.b, dW + db)

            history["train_loss"].append(self.loss(X_train, y_train))
            history["train_acc"].append(accuracy_score(y_train.ravel(), self.predict(X_train)))
            if X_val is not None:
                history["val_loss"].append(self.loss(X_val, y_val))
                history["val_acc"].append(accuracy_score(y_val, self.predict(X_val)))
            if verbose:
                msg = (f"epoch {epoch:3d}  train_loss {history['train_loss'][-1]:.4f}"
                       f"  train_acc {history['train_acc'][-1]:.4f}")
                if X_val is not None:
                    msg += (f"  val_loss {history['val_loss'][-1]:.4f}"
                            f"  val_acc {history['val_acc'][-1]:.4f}")
                print(msg)
        return history

    # -------------------------------------------------------------- predict --
    def predict_proba(self, X):
        """P(income > 50K) for each row, shape (m,)."""
        z_out, _, _ = self._forward(X)
        return sigmoid(z_out).ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)

    # ------------------------------------------------------------- evaluate --
    def evaluate(self, X, y):
        """Accuracy, macro-F1 (classes are imbalanced ~76/24) and BCE loss."""
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "macro_f1": f1_score(y, y_pred, average="macro"),
            "loss": self.loss(X, y),
        }
