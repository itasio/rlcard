import os
import argparse
import numpy as np
from sklearn.model_selection import ParameterGrid
import shutil
import pandas as pd

import rlcard
from rlcard.agents import QLAgent, RandomAgent, ThresholdAgent, ThresholdAgent2
from rlcard.utils import set_seed, tournament, Logger, plot_curve

def train(args, alpha, gamma):
    env = rlcard.make(
        'new-limit-holdem',
        config={
            'seed': 0,
            'allow_step_back': True,
        }
    )
    eval_env = rlcard.make(
        'new-limit-holdem',
        config={
            'seed': 0,
        }
    )

    set_seed(args.seed)

    # Delete the previous model if it exists
    model_path = os.path.join(args.log_dir, 'ql_model')
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    agent = QLAgent(
        env,
        os.path.join(
            args.log_dir,
            'ql_model',
        ),
        alpha,
        gamma
    )
    agent.load()

    eval_env.set_agents([
        agent,
        ThresholdAgent2(num_actions=env.num_actions),
    ])

    env.set_agents([
        agent,
        ThresholdAgent2(num_actions=env.num_actions),
    ])

    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            agent.train()
            print('\rIteration {}'.format(episode), end='')
            if episode % args.evaluate_every == 0:
                agent.save()
                logger.log_performance(
                    episode,
                    tournament(
                        eval_env,
                        args.num_eval_games
                    )[0]
                )

        csv_path, fig_path = logger.csv_path, logger.fig_path

    plot_curve(csv_path, fig_path, 'Q-learning')
    df = pd.read_csv(csv_path)
    avg_score = df['reward'].tail(args.evaluate_every).mean()

    return avg_score

def evaluate_hyperparameters(args, alpha, gamma):
    avg_score = train(args, alpha, gamma)
    print(f"Alpha: {alpha}, Gamma: {gamma}, Avg Score: {avg_score}")
    return avg_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Q-Learning Agent example in RLCard")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=2800)
    parser.add_argument('--num_eval_games', type=int, default=2000)
    parser.add_argument('--evaluate_every', type=int, default=400)
    parser.add_argument('--log_dir', type=str, default='experiments/new_limit_holdem_ql_result/')

    args = parser.parse_args()

    param_grid = {
        'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
        'gamma': [0.1, 0.3, 0.5, 0.7, 0.9],
    }

    best_score = -np.inf
    best_params = None

    for params in ParameterGrid(param_grid):
        score = evaluate_hyperparameters(args, params['alpha'], params['gamma'])
        if score > best_score:
            best_score = score
            best_params = params

    print("Best hyperparameters found: ", best_params)


