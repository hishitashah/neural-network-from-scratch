| model | config | train_acc | val_acc | test_acc | test_macro_f1 | command |
|---|---|---|---|---|---|---|
| NN best sigmoid (run 10) | sigmoid, hidden 64-32, adam, lr 0.001 | 0.8617 | 0.8583 | 0.8601 | 0.7898 | python train.py --hidden 64 32 --activation sigmoid --optimizer adam --lr 0.001 --epochs 50 --batch-size 64 --beta1 0.9 --beta2 0.999 --eps 1e-08 --threshold 0.5 --seed 42 --split 0.7 0.15 0.15 --test |
| NN best tanh (run 21) | tanh, hidden 64-32, sgd, lr 0.1 | 0.8665 | 0.8581 | 0.8649 | 0.8029 | python train.py --hidden 64 32 --activation tanh --optimizer sgd --lr 0.1 --epochs 50 --batch-size 64 --beta1 0.9 --beta2 0.999 --eps 1e-08 --threshold 0.5 --seed 42 --split 0.7 0.15 0.15 --test |
| NN best relu (run 26) | relu, hidden 64, sgd, lr 0.01 | 0.8576 | 0.8561 | 0.8567 | 0.7843 | python train.py --hidden 64 --activation relu --optimizer sgd --lr 0.01 --epochs 50 --batch-size 64 --beta1 0.9 --beta2 0.999 --eps 1e-08 --threshold 0.5 --seed 42 --split 0.7 0.15 0.15 --test |
| Logistic regression (baseline) | C 1, max_iter 1000 | 0.8531 | 0.8534 | 0.8491 | 0.7778 | python final_evaluation.py --baseline-C 1 --baseline-max-iter 1000 |
