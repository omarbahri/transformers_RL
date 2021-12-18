# Transformers for Reinforcement Learning

The motivation of most classical reinforcement learning algorithms is to learn an optimal, single-step policy by capitalizing on the Markov property. However, by taking a step back and redefining the problem as sequence modeling, where the goal is to predict a sequence of actions based on a previous sequence of states-actions-rewards, it becomes very tempting to leverage state-of-the-art sequence-to-sequence models. Janner et al. [1] and Chen et al. [2] did exactly that by training a transformer architecture on offline benchmark datasets. Their results show the superiority of this new approach. In this project, I aim to 1) learn about transformers and the attention mechanism, 2) explore and understand the work done in [1] and [2], 3) generate my own offline dataset on a simple problem, in compliance with the D4RL benchmark [3] format, and 4) evaluate Chen et al.’s Decision Transformer performance on this dataset, with the motivation of verifying whether Decision Transformer can achieve better results using a dataset generated using a simple algorithm?

[1] M. Janner, Q. Li, and S. Levine, “Reinforcement Learning as One Big Sequence Modeling Problem,” 2021.

[2] L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mordatch, “Decision Transformer: Reinforcement Learning via Sequence Modeling,” jun 2021.

[3] J. Fu, A. Kumar, O. Nachum, G. Brain, G. T. Google Brain, and S. Levine, “D4RL: Datasets for Deep Data-Driven Reinforcement Learning,”

## Steps to Reproduce
I have used code from https://github.com/kzl/decision-transformer/tree/master/gym and https://github.com/zhihanyang2022/pytorch-sac. However, please run the files I provided in this repository, as I have introduced several modifications.

1. Follow instructions in https://github.com/kzl/decision-transformer/tree/master/gym to install environment.
2. Follow instructions in https://github.com/zhihanyang2022/pytorch-sac to install evironment.
3. Generate pendulum-random using scripts/generate_random_pendulum.py.
4. Generate pendulum-medium-replay using pytorch-sac-main/generate_sac_pendulum.py.
5. Generate pendulum-medium using pytorch-sac-main/generate_trained_pendulum.py.
6. Convert datasets to D4RL format using scripts/convert_d4rl_dataset.py
7. Train decision transformer using decision-transformer/gym/experiment.py.
8. Evaluate trained models using decision-transformer/gym/evaluate_dt.
