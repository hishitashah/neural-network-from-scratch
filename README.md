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

## Status

- [x] Data loading and stratified splitting
- [x] Preprocessing, network, backpropagation, SGD and Adam
- [x] Gradient check
- [x] `train.py`: command-line interface for a single run
- [ ] Experiment sweep (`experiments.json`) and results table
- [ ] Learning curves and test accuracy for the best configuration per activation
- [ ] Logistic regression baseline
- [ ] Report

## Assumptions

- `adult.data` and `adult.test` are merged and then re-split, instead of using the official test file as the test set.
- Missing values are kept as an `Unknown` category; no rows are dropped.
- `fnlwgt` is dropped because it is a census sampling weight, not a trait of the person. `education` is dropped because it duplicates `education-num` one-to-one.
- Duplicate rows in the raw data are kept.
- A predicted probability of 0.5 or more is classified as `>50K`.
