"""
Run the experiment sweep defined in a JSON config and write the results table.

    python run_experiments.py                       # uses experiments.json -> results/

The config has two parts:
  "defaults": hyperparameters shared by every run (epochs, batch size, seed, ...)
  "grid":     lists of values; every combination becomes one configuration
              (e.g. 3 activations x 2 architectures x 2 optimizers x 3 lrs = 36)

Outputs (in --out):
  runs/<id>_....json   config, per-epoch history and metrics of each run
  results.csv          one row per configuration
  results.md           the same table in Markdown

Only train and validation data are used here; the test set is not touched.
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from train import prepare_data, run, to_jsonable

# Column order of the results table.
TABLE_COLUMNS = [
    "id", "activation", "hidden", "optimizer", "lr", "epochs", "batch_size", "seed",
    "train_acc", "train_macro_f1", "val_acc", "val_macro_f1",
    "train_loss", "val_loss", "best_val_loss_epoch", "train_seconds",
]


def expand_grid(spec):
    """Turn {"defaults": {...}, "grid": {...}} into a list of full configs."""
    keys = list(spec["grid"])
    configs = []
    for values in itertools.product(*(spec["grid"][k] for k in keys)):
        config = dict(spec["defaults"])
        config.update(zip(keys, values))
        configs.append(config)
    return configs


def check_coverage(configs):
    """Fail early if the sweep does not meet the assignment's requirement 5."""
    problems = []
    if len(configs) < 12:
        problems.append(f"only {len(configs)} configurations (need >= 12)")
    if {c["activation"] for c in configs} != {"sigmoid", "tanh", "relu"}:
        problems.append("must include all three activations (sigmoid, tanh, relu)")
    if len({c["lr"] for c in configs}) < 2:
        problems.append("need at least two learning rates")
    if not {1, 2} <= {len(c["hidden"]) for c in configs}:
        problems.append("need both a 1-hidden-layer and a 2-hidden-layer network")
    if not {"sgd", "adam"} <= {c["optimizer"] for c in configs}:
        problems.append("need both sgd and adam")
    if problems:
        raise SystemExit("config does not cover requirement 5:\n  - " + "\n  - ".join(problems))


def train_command(config):
    """The equivalent single-run command, so any row can be reproduced alone."""
    return ("python train.py"
            f" --hidden {' '.join(map(str, config['hidden']))}"
            f" --activation {config['activation']} --optimizer {config['optimizer']}"
            f" --lr {config['lr']} --epochs {config['epochs']}"
            f" --batch-size {config['batch_size']}"
            f" --beta1 {config['beta1']} --beta2 {config['beta2']} --eps {config['eps']}"
            f" --threshold {config['threshold']}"
            f" --seed {config['seed']} --split {' '.join(map(str, config['split']))}")


def run_name(run_id, config):
    hidden = "x".join(map(str, config["hidden"]))
    return f"{run_id:02d}_{config['activation']}_{hidden}_{config['optimizer']}_lr{config['lr']}"


def table_row(run_id, result):
    c, h = result["config"], result["history"]
    return {
        "id": run_id,
        "activation": c["activation"],
        "hidden": "-".join(map(str, c["hidden"])),
        "optimizer": c["optimizer"],
        "lr": c["lr"],
        "epochs": c["epochs"],
        "batch_size": c["batch_size"],
        "seed": c["seed"],
        "train_acc": result["train"]["accuracy"],
        "train_macro_f1": result["train"]["macro_f1"],
        "val_acc": result["val"]["accuracy"],
        "val_macro_f1": result["val"]["macro_f1"],
        "train_loss": result["train"]["loss"],
        "val_loss": result["val"]["loss"],
        # 1-based epoch with the lowest validation loss; far below `epochs`
        # means validation loss went back up later (a sign of overfitting).
        "best_val_loss_epoch": int(np.argmin(h["val_loss"])) + 1,
        "train_seconds": result["train_seconds"],
    }


def to_markdown(df):
    """Format a DataFrame as a Markdown table (avoids the optional tabulate dependency)."""
    def fmt(col, v):
        if col == "lr":
            return f"{v:g}"
        if col == "train_seconds":
            return f"{v:.1f}"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(c, v) for c, v in zip(df.columns, row)) + " |")
    return "\n".join(lines) + "\n"


def best_by_val(df):
    """Best row: highest validation accuracy, ties broken by validation macro-F1."""
    return df.sort_values(["val_acc", "val_macro_f1"], ascending=False).iloc[0]


def main():
    p = argparse.ArgumentParser(description="Run the experiment sweep from a JSON config")
    p.add_argument("--config", default="experiments.json")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--verbose", action="store_true", help="print the per-epoch log of every run")
    args = p.parse_args()

    with open(args.config) as f:
        spec = json.load(f)
    configs = expand_grid(spec)
    check_coverage(configs)

    # All runs share one split; download and split the data only once.
    split_keys = {(tuple(c["split"]), c["seed"]) for c in configs}
    if len(split_keys) != 1:
        raise SystemExit("all configurations must share the same split and seed")
    split, seed = split_keys.pop()
    data = prepare_data(split, seed)

    runs_dir = os.path.join(args.out, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    rows = []
    for run_id, config in enumerate(configs, start=1):
        result = run(config, data, verbose=args.verbose)
        result["command"] = train_command(config)
        with open(os.path.join(runs_dir, run_name(run_id, config) + ".json"), "w") as f:
            json.dump(to_jsonable(result), f, indent=2)
        row = table_row(run_id, result)
        rows.append(row)
        print(f"[{run_id:2d}/{len(configs)}] {row['activation']:7s} hidden={row['hidden']:6s}"
              f" {row['optimizer']:4s} lr={row['lr']:<6} val_acc {row['val_acc']:.4f}"
              f"  val_f1 {row['val_macro_f1']:.4f}  ({row['train_seconds']:.1f}s)")

    df = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    df.to_csv(os.path.join(args.out, "results.csv"), index=False)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(to_markdown(df))
    print(f"\nwrote {args.out}/results.csv, {args.out}/results.md and {len(rows)} run files")

    print("\nbest by validation accuracy (ties: validation macro-F1)")
    groups = [("activation", a) for a in ("sigmoid", "tanh", "relu")] + \
             [("optimizer", o) for o in ("sgd", "adam")]
    for col, val in groups:
        b = best_by_val(df[df[col] == val])
        print(f"  {col}={val:7s} -> run {b['id']:2d}: {b['activation']} hidden={b['hidden']}"
              f" {b['optimizer']} lr={b['lr']}  val_acc {b['val_acc']:.4f}"
              f"  val_f1 {b['val_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
