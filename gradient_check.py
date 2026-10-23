"""
Numerical gradient check: compares backprop gradients with central finite
differences  (L(theta + h) - L(theta - h)) / 2h  on a small random problem.
A relative error below ~1e-6 means the backward pass is correct.

    python gradient_check.py
"""

import numpy as np

from neural_net import NeuralNetwork


def check(hidden, activation, seed=0, h=1e-5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(20, 6))
    y = rng.integers(0, 2, size=(20, 1)).astype(float)

    net = NeuralNetwork(hidden=hidden, activation=activation, seed=seed)
    net._init_params(X.shape[1])
    # Small random biases so no pre-activation sits exactly on ReLU's kink at
    # z = 0 (with zero biases, a row whose layer-1 units are all off gives
    # z = 0 exactly in layer 2, where finite differences are one-sided).
    net.b = [rng.normal(0, 0.1, size=b.shape) for b in net.b]
    z_out, Zs, As = net._forward(X)
    dW, db = net._backward(y, z_out, Zs, As)

    worst = 0.0
    for params, grads in ((net.W, dW), (net.b, db)):
        for p, g in zip(params, grads):
            num = np.zeros_like(p)
            for i in np.ndindex(p.shape):   # loop is fine here: test code only
                old = p[i]
                p[i] = old + h
                lp = net.loss(X, y)
                p[i] = old - h
                lm = net.loss(X, y)
                p[i] = old
                num[i] = (lp - lm) / (2 * h)
            rel = np.linalg.norm(num - g) / (np.linalg.norm(num) + np.linalg.norm(g) + 1e-12)
            worst = max(worst, rel)
    return worst


if __name__ == "__main__":
    ok = True
    for hidden in ([5], [5, 4]):
        for act in ("sigmoid", "tanh", "relu"):
            err = check(hidden, act)
            ok &= err < 1e-6
            print(f"hidden={str(hidden):7s} act={act:8s} max relative error = {err:.2e}")
    print("PASS" if ok else "FAIL")
