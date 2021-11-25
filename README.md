# Robustness with Intersection Sets
The two folders contain experiments for supervised learning and bandits (`supervised_cb`), and reinforcement learning (`rl`).

## Run Instructions
The two folders have run scripts with the extension `*.sh`.

To run supervised learning experiment, run
```
cd supervised_cb
bash run_supervised.sh
```

For bandits, run
```
cd supervised_cb
bash run_cb.sh
bash run_warfarin.sh
```

For reinforcement learning, run
```
cd rl
bash run_batch_cliff.sh
bash run_batch_sepsis.sh
```

Uncomment the code in each script to run with the full dataset or for multiple runs.

## Packages Required
Instal by running
`pip install torch numpy scipy scikit-learn pandas joblib pymdptoolbox matplotlib seaborn tqdm`



# Acknowledgements

The code includes publicly available scripts from the following three codebases.

## Supervised Learning and Contextual Bandits

Code for the experiments is adapted from [Srivastava et al.](https://bit.ly/uvdro-codalab)'s implementation.

## Reinforcement Learning
Code for the experiments is adapted from [Roy et al.](https://github.com/royaurko/rl-mismatch)'s implementation.

Sepsis simulator is written by [Oberst and Sontag](https://github.com/clinicalml/gumbel-max-scm).
