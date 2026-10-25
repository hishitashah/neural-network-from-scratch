"""
Train and evaluate one network configuration from the command line.

    python train.py                                    # defaults (see --help)
    python train.py --hidden 64 32 --activation tanh --optimizer sgd --lr 0.01
    python train.py --test --save results/run.json     # final config: also test

The test set is only evaluated when --test is passed, so tuning runs never
look at it.
"""

import argparse
import json
import os
import time

from data import load_adult, split_data
from neural_net import ACTIVATIONS, OPTIMIZERS, NeuralNetwork

# Config keys passed straight to NeuralNetwork.__init__.
MODEL_KEYS = ("hidden", "activation", "optimizer", "lr", "epochs",
              "batch_size", "beta1", "beta2", "eps", "threshold", "seed")


def build_parser():
    p = argparse.ArgumentParser(description="Feed-forward NN on UCI Adult (NumPy, from scratch)")
    p.add_argument("--hidden", type=int, nargs="+", default=[64],
                   help="neurons per hidden layer, 1 or 2 values (e.g. --hidden 64 32)")
    p.add_argument("--activation", choices=list(ACTIVATIONS), default="relu",
                   help="hidden-layer activation")
    p.add_argument("--optimizer", choices=list(OPTIMIZERS), default="adam")
    p.add_argument("--lr", type=float, default=0.001, help="learning rate")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--beta1", type=float, default=0.9, help="Adam 1st-moment decay")
    p.add_argument("--beta2", type=float, default=0.999, help="Adam 2nd-moment decay")
    p.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="decision threshold: predict >50K when P(>50K) >= threshold")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for the split, weight init and shuffling")
    p.add_argument("--split", type=float, nargs=3, default=[0.70, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"), help="train/val/test ratios")
    p.add_argument("--test", action="store_true",
                   help="also evaluate on the test set (final configurations only)")
    p.add_argument("--save", metavar="PATH", help="write config, history and metrics as JSON")
    p.add_argument("--quiet", action="store_true", help="no per-epoch log")
    return p


def prepare_data(split, seed):
    """Download Adult once and return the raw (X_train, X_val, X_test, y_train, y_val, y_test)."""
    X, y = load_adult()
    return split_data(X, y, ratios=tuple(split), seed=seed)


def run(config, data, verbose=True, eval_test=False):
    """Train one configuration on pre-split raw data and return a results dict."""
    X_train, X_val, X_test, y_train, y_val, y_test = data
    net = NeuralNetwork(**{k: config[k] for k in MODEL_KEYS})
    Xtr, Xva, Xte = net.preprocess(X_train, X_val, X_test)

    start = time.perf_counter()
    history = net.train(Xtr, y_train, Xva, y_val, verbose=verbose)
    train_seconds = time.perf_counter() - start

    result = {
        "config": dict(config),
        "history": history,
        "train": net.evaluate(Xtr, y_train),
        "val": net.evaluate(Xva, y_val),
        "train_seconds": train_seconds,
    }
    if eval_test:
        result["test"] = net.evaluate(Xte, y_test)
    return result


def to_jsonable(obj):
    """Recursively convert NumPy scalars to plain Python numbers for json.dump."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "item"):  # NumPy scalar
        return obj.item()
    return obj


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= len(args.hidden) <= 2:
        parser.error("--hidden takes 1 or 2 layer sizes")

    config = vars(args).copy()
    for k in ("test", "save", "quiet"):  # run options, not hyperparameters
        config.pop(k)
    print("config:", json.dumps(config))

    data = prepare_data(args.split, args.seed)
    result = run(config, data, verbose=not args.quiet, eval_test=args.test)

    print(f"\ntrained in {result['train_seconds']:.1f}s (seed {args.seed})")
    for split in ("train", "val", "test"):
        if split in result:
            m = result[split]
            print(f"{split:5s}  accuracy {m['accuracy']:.4f}  macro_f1 {m['macro_f1']:.4f}"
                  f"  loss {m['loss']:.4f}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(to_jsonable(result), f, indent=2)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
