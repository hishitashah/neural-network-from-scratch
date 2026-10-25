# Feed-Forward Neural Network from Scratch — UCI Adult (Census Income)

A feed-forward neural network with one or two hidden layers. The forward pass, backpropagation and the optimizers are written by hand in NumPy, with no PyTorch, TensorFlow, autograd or similar. It is trained to predict whether a person earns more than $50K a year.

## Dataset

- **UCI Adult / Census Income:** https://archive.ics.uci.edu/dataset/20/census+income
- The code downloads `adult.data` and `adult.test` from the public UCI URL every time it runs, so the data is not included in this repo.
- The two files are merged into 48,842 rows, with 12 features after dropping two columns (see Assumptions).
- This is a binary task: `>50K` is labelled 1 and `<=50K` is labelled 0.
- The classes are imbalanced, with about 23.9% positive, so macro-F1 is reported alongside accuracy.

## Installation

Requires Python 3.x (developed with 3.13).

```bash
pip install -r requirements.txt
```

## Project structure

| File | Contents |
|---|---|
| `data.py` | Downloads and merges the raw files, cleans the labels, and makes the stratified train/val/test split |
| `neural_net.py` | The `NeuralNetwork` class (`preprocess`, `train`, `predict`, `evaluate`), the activations, and the `SGD` / `Adam` optimizers |
| `gradient_check.py` | Checks backprop's gradients against finite-difference estimates |
| `train.py` | Command-line script that trains and evaluates one configuration |
| `experiments.json` | The experiment sweep config: shared defaults plus a grid of values to combine |
| `run_experiments.py` | Runs every configuration in the sweep and writes the results table to `results/` |
| `final_evaluation.py` | Learning curves, the SGD vs Adam comparison, test results for the best run per activation, and the logistic regression baseline |

## Implementation summary

- **Preprocessing (`NeuralNetwork.preprocess`):**
  - Missing values (`?`) in categorical columns become their own `Unknown` category.
  - Categorical features are one-hot encoded.
  - Numeric features are standardized.
  - Every transformer is fit on the training set only and then applied to validation and test.
- **Split:** stratified 70 / 15 / 15 (train / validation / test), giving 34,188 / 7,327 / 7,327 rows.
- **Network:**
  - 1 or 2 hidden layers of any size, and every neuron has a bias.
  - Hidden activation is one of `sigmoid`, `tanh` or `relu`.
  - The output is a single sigmoid unit trained with binary cross-entropy.
- **Initialization:** Xavier/Glorot for sigmoid and tanh layers, He for ReLU layers. The sigmoid output layer always uses Xavier, and biases start at zero.
- **Optimizers:** plain mini-batch SGD and Adam.
- **Vectorization:** each mini-batch is processed as one matrix, with no Python loop over examples or neurons.
- **Reproducibility:** the seed (default 42) controls the data split, weight initialization and mini-batch shuffling.

## Running

Check that backpropagation is correct:

```bash
python gradient_check.py
```

This prints the largest relative error between the analytic and numerical gradients for each activation, with 1 and 2 hidden layers. All cases currently pass, with errors of about 1e-8 or less.

### Single run

```bash
python train.py                                   # defaults: --hidden 64, relu, adam, lr 0.001
python train.py --hidden 64 32 --activation tanh --optimizer sgd --lr 0.01
python train.py --hidden 64 32 --test --save results/run.json
```

The script prints the configuration it is using, including the seed, then a log line per epoch with train/val loss and accuracy. At the end it prints a summary of accuracy, macro-F1 and loss.

| Flag | Default | Meaning |
|---|---|---|
| `--hidden` | `64` | Neurons per hidden layer, 1 or 2 values (e.g. `--hidden 64 32`) |
| `--activation` | `relu` | Hidden activation: `sigmoid`, `tanh` or `relu` |
| `--optimizer` | `adam` | `sgd` or `adam` |
| `--lr` | `0.001` | Learning rate |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` | `64` | Mini-batch size |
| `--beta1` / `--beta2` / `--eps` | `0.9` / `0.999` / `1e-8` | Adam settings (SGD ignores them) |
| `--seed` | `42` | Random seed for the split, weight initialization and shuffling |
| `--split` | `0.70 0.15 0.15` | Train / validation / test ratios |
| `--test` | off | Also evaluate on the test set. Use this only for final configurations. |
| `--save PATH` | — | Write the config, per-epoch history and metrics to a JSON file |
| `--quiet` | off | Turn off the per-epoch log |

Two runs with the same arguments give identical results.

### Running the experiments

```bash
python run_experiments.py                 # reads experiments.json, writes to results/ (~3 min)
```

`experiments.json` has two parts:
- `defaults` holds the settings every run shares: 50 epochs, batch size 64, Adam settings, seed 42 and the 70/15/15 split.
- `grid` holds lists of values. Every combination becomes one configuration: 3 activations × 2 architectures (`[64]`, `[64, 32]`) × 2 optimizers (SGD, Adam) × 3 learning rates (0.001, 0.01, 0.1) = **36 runs**.

Before training, the script checks that the grid meets requirement 5 of the assignment. The data is downloaded and split once, and every run uses the same split. The test set is not used.

| Output | Contents |
|---|---|
| `results/results.csv`, `results/results.md` | One row per configuration: parameters, train/val accuracy, train/val macro-F1, final train/val loss, the epoch with the lowest validation loss, and training time. All metrics are from the final (50th) epoch. |
| `results/runs/<id>_<activation>_<hidden>_<optimizer>_lr<lr>.json` | One file per run with its config, per-epoch history and metrics, plus the `train.py` command that reproduces that run on its own |

### Final evaluation

Run this after `run_experiments.py`:

```bash
python final_evaluation.py               # ~30 s; optional: --baseline-C 1.0 --baseline-max-iter 1000
```

1. For each activation, picks the best configuration from `results/results.csv` using validation data only.
2. Plots the learning curves from the saved histories.
3. Retrains the three best configurations with the same seed and evaluates them on the test set. The script checks that each retrained model matches its saved validation accuracy exactly.
4. Trains a logistic regression baseline (scikit-learn) on the same split and the same preprocessed features.

This is the only script that uses the test set.

| Output | Contents |
|---|---|
| `results/final.md`, `results/final.csv` | Train/val/test accuracy and test macro-F1 for the best run per activation and for logistic regression, with the command that reproduces each row |
| `results/plots/learning_curve_<activation>.png` | Training and validation loss vs epoch for the best run of each activation. A dotted line marks the epoch with the lowest validation loss. |
| `results/plots/learning_curves.png` | The three learning curves side by side, with the same y-axis |
| `results/plots/sgd_vs_adam.png` | Validation loss of SGD vs Adam for each activation, with hidden 64-32 and lr 0.01. Only the optimizer differs between the two lines. |
| `results/optimizer_comparison.csv` | For every matched SGD/Adam pair: the first epoch reaching 0.85 validation accuracy, the lowest validation loss and the epoch it happened, and final validation accuracy and macro-F1 |

## Status

- [x] Data loading and stratified splitting
- [x] Preprocessing, network, backpropagation, SGD and Adam
- [x] Gradient check
- [x] `train.py`: command-line interface for a single run
- [x] Experiment sweep (`experiments.json`) and results table
- [x] Learning curves and test accuracy for the best configuration per activation
- [x] Logistic regression baseline
- [ ] Report

## Assumptions

- `adult.data` and `adult.test` are merged and then re-split, instead of using the official test file as the test set.
- Missing values are kept as an `Unknown` category; no rows are dropped.
- `fnlwgt` is dropped because it is a census sampling weight, not a trait of the person. `education` is dropped because it duplicates `education-num` one-to-one.
- Duplicate rows in the raw data are kept.
- A predicted probability of 0.5 or more is classified as `>50K`.
- The "best" configuration is the one with the highest validation accuracy, with validation macro-F1 breaking ties. The assignment says to use the validation set for tuning and names accuracy as the reported metric; macro-F1 is reported alongside it.
- Every run trains for a fixed 50 epochs with no early stopping, and the reported metrics are from the last epoch.
