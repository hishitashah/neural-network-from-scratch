"""
Final evaluation, run after run_experiments.py:

  * picks the best configuration per activation (validation accuracy,
    ties broken by validation macro-F1) from results/results.csv
  * plots train/val loss vs epoch for those runs (from the saved histories)
  * plots SGD vs Adam validation loss and writes an optimizer comparison table
  * retrains the 3 best configurations and evaluates them on the TEST set
  * trains a logistic-regression baseline on the same split and features

    python final_evaluation.py

This is the only script that looks at the test set.
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")  # write PNGs, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from neural_net import NeuralNetwork
from run_experiments import best_by_val, to_markdown
from train import prepare_data, run

ACTIVATION_ORDER = ["sigmoid", "tanh", "relu"]

# Plot styling: two validated categorical hues (blue, orange) plus recessive
# chrome, chosen to stay distinguishable for colour-blind readers and in print.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


# ----------------------------------------------------------------- helpers --

def load_run(results_dir, run_id):
    paths = glob.glob(os.path.join(results_dir, "runs", f"{run_id:02d}_*.json"))
    if len(paths) != 1:
        raise SystemExit(f"expected one run file for id {run_id}, found {paths}")
    with open(paths[0]) as f:
        return json.load(f)


def describe(config):
    hidden = "-".join(map(str, config["hidden"]))
    return f"{config['activation']}, hidden {hidden}, {config['optimizer']}, lr {config['lr']:g}"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def end_label(ax, x, y, text, color, dy=0):
    """Direct label just right of a line's last point (text stays in ink).
    dy (points) nudges the label up/down so two close end points don't collide."""
    ax.plot([x], [y], "o", color=color, markersize=4)
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=9, color=INK_2)


def plot_learning_curve(ax, run_result):
    """Train and validation BCE loss vs epoch on the same axes."""
    h = run_result["history"]
    epochs = np.arange(1, len(h["train_loss"]) + 1)
    ax.plot(epochs, h["train_loss"], color=BLUE, linewidth=2, label="training loss")
    ax.plot(epochs, h["val_loss"], color=ORANGE, linewidth=2, label="validation loss")
    # The higher end point's label goes up, the lower one's down.
    up = 5 if h["val_loss"][-1] >= h["train_loss"][-1] else -5
    end_label(ax, epochs[-1], h["train_loss"][-1], "train", BLUE, dy=-up)
    end_label(ax, epochs[-1], h["val_loss"][-1], "val", ORANGE, dy=up)

    best = int(np.argmin(h["val_loss"])) + 1
    ax.axvline(best, color=MUTED, linestyle=":", linewidth=1.2)
    ax.annotate(f"lowest val loss\n(epoch {best})", (best, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -4), textcoords="offset points", va="top", fontsize=8, color=MUTED)

    style_axes(ax)
    ax.set_xlim(0, len(epochs) + 6)  # room for the end labels
    ax.set_xlabel("epoch")
    ax.set_ylabel("binary cross-entropy loss")
    ax.set_title(describe(run_result["config"]), fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper center")


def save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {path}")


def first_epoch_reaching(values, threshold):
    hits = np.nonzero(np.asarray(values) >= threshold)[0]
    return int(hits[0]) + 1 if len(hits) else None


# -------------------------------------------------------------------- main --

def main():
    p = argparse.ArgumentParser(description="Learning curves, test results and baseline")
    p.add_argument("--results", default="results", help="folder written by run_experiments.py")
    p.add_argument("--baseline-C", type=float, default=1.0,
                   help="logistic regression inverse regularisation strength")
    p.add_argument("--baseline-max-iter", type=int, default=1000,
                   help="logistic regression solver iterations")
    p.add_argument("--target-acc", type=float, default=0.85,
                   help="val accuracy used to measure convergence speed in the optimizer table")
    p.add_argument("--compare-hidden", type=int, nargs="+", default=[64, 32],
                   help="architecture shown in the SGD vs Adam plot")
    p.add_argument("--compare-lr", type=float, default=0.01,
                   help="learning rate shown in the SGD vs Adam plot")
    args = p.parse_args()

    df = pd.read_csv(os.path.join(args.results, "results.csv"))
    plots_dir = os.path.join(args.results, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Best configuration per activation (validation data only).
    best_ids = {act: int(best_by_val(df[df["activation"] == act])["id"]) for act in ACTIVATION_ORDER}
    best = {act: load_run(args.results, run_id) for act, run_id in best_ids.items()}
    print("best per activation:", {a: f"run {i}" for a, i in best_ids.items()})

    # 2. Learning curves: one file per activation, plus a combined figure
    #    with a shared y-axis for side-by-side comparison.
    for act in ACTIVATION_ORDER:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_learning_curve(ax, best[act])
        save(fig, os.path.join(plots_dir, f"learning_curve_{act}.png"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, act in zip(axes, ACTIVATION_ORDER):
        plot_learning_curve(ax, best[act])
    for ax in axes[1:]:
        ax.set_ylabel("")
    save(fig, os.path.join(plots_dir, "learning_curves.png"))

    # 3a. SGD vs Adam: same activation, architecture (--compare-hidden) and
    #     lr (--compare-lr); only the optimizer differs between the two lines.
    cmp_hidden = "-".join(map(str, args.compare_hidden))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, act in zip(axes, ACTIVATION_ORDER):
        for opt, color in (("sgd", BLUE), ("adam", ORANGE)):
            match = df[(df.activation == act) & (df.hidden == cmp_hidden)
                       & (df.optimizer == opt) & np.isclose(df.lr, args.compare_lr)]
            if match.empty:
                raise SystemExit(f"no run with hidden {cmp_hidden}, lr {args.compare_lr} in results.csv")
            row = match.iloc[0]
            h = load_run(args.results, int(row["id"]))["history"]
            epochs = np.arange(1, len(h["val_loss"]) + 1)
            ax.plot(epochs, h["val_loss"], color=color, linewidth=2, label=f"{opt.upper()} (run {row['id']})")
            end_label(ax, epochs[-1], h["val_loss"][-1], opt.upper(), color)
        style_axes(ax)
        ax.set_xlim(0, len(epochs) + 7)
        ax.set_xlabel("epoch")
        ax.set_title(f"{act}, hidden {cmp_hidden}, lr {args.compare_lr:g}", fontsize=10,
                     color=INK, loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    axes[0].set_ylabel("validation BCE loss")
    save(fig, os.path.join(plots_dir, "sgd_vs_adam.png"))

    # 3b. Optimizer comparison table over every matched SGD/Adam pair.
    comp_rows = []
    for (act, hidden, lr), group in df.groupby(["activation", "hidden", "lr"], sort=False):
        row = {"activation": act, "hidden": hidden, "lr": lr}
        for opt in ("sgd", "adam"):
            r = group[group.optimizer == opt].iloc[0]
            h = load_run(args.results, int(r["id"]))["history"]
            row[f"{opt}_epoch_to_{args.target_acc:g}_val_acc"] = first_epoch_reaching(h["val_acc"], args.target_acc)
            row[f"{opt}_min_val_loss"] = float(np.min(h["val_loss"]))
            row[f"{opt}_min_val_loss_epoch"] = int(np.argmin(h["val_loss"])) + 1
            row[f"{opt}_final_val_acc"] = r["val_acc"]
            row[f"{opt}_final_val_macro_f1"] = r["val_macro_f1"]
        comp_rows.append(row)
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(os.path.join(args.results, "optimizer_comparison.csv"), index=False)
    print(f"saved {os.path.join(args.results, 'optimizer_comparison.csv')}")

    # 4. Test-set evaluation of the 3 best configurations. Weights were not
    #    saved, so each is retrained; a fixed seed reproduces the same model,
    #    which the validation-accuracy check below confirms.
    first = best[ACTIVATION_ORDER[0]]["config"]
    data = prepare_data(first["split"], first["seed"])
    final_rows = []
    for act in ACTIVATION_ORDER:
        saved = best[act]
        result = run(saved["config"], data, verbose=False, eval_test=True)
        if not np.isclose(result["val"]["accuracy"], saved["val"]["accuracy"], rtol=0, atol=1e-12):
            raise SystemExit(f"retrained {act} run does not reproduce its saved validation accuracy")
        final_rows.append({
            "model": f"NN best {act} (run {best_ids[act]})",
            "config": describe(saved["config"]),
            "train_acc": result["train"]["accuracy"],
            "val_acc": result["val"]["accuracy"],
            "test_acc": result["test"]["accuracy"],
            "test_macro_f1": result["test"]["macro_f1"],
            "command": saved["command"] + " --test",
        })

    # 5. Logistic-regression baseline: identical features (same preprocess,
    #    fit on train only) and identical split.
    X_train, X_val, X_test, y_train, y_val, y_test = data
    Xtr, Xva, Xte = NeuralNetwork(seed=first["seed"]).preprocess(X_train, X_val, X_test)
    logreg = LogisticRegression(C=args.baseline_C, max_iter=args.baseline_max_iter,
                                random_state=first["seed"])
    logreg.fit(Xtr, y_train)

    def lr_predict(X):  # same decision threshold as the networks
        return (logreg.predict_proba(X)[:, 1] >= first["threshold"]).astype(int)

    final_rows.append({
        "model": "Logistic regression (baseline)",
        "config": f"C {args.baseline_C:g}, max_iter {args.baseline_max_iter}",
        "train_acc": accuracy_score(y_train, lr_predict(Xtr)),
        "val_acc": accuracy_score(y_val, lr_predict(Xva)),
        "test_acc": accuracy_score(y_test, lr_predict(Xte)),
        "test_macro_f1": f1_score(y_test, lr_predict(Xte), average="macro"),
        "command": (f"python final_evaluation.py --baseline-C {args.baseline_C:g}"
                    f" --baseline-max-iter {args.baseline_max_iter}"),
    })

    final = pd.DataFrame(final_rows)
    final.to_csv(os.path.join(args.results, "final.csv"), index=False)
    with open(os.path.join(args.results, "final.md"), "w") as f:
        f.write(to_markdown(final))
    print(f"saved {os.path.join(args.results, 'final.csv')} and final.md\n")
    print(final.drop(columns="command").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nmajority-class test accuracy (always <=50K): {1 - y_test.mean():.4f}")


if __name__ == "__main__":
    main()
