''' Find all the optimal hyperparameters and compare for q learning algorithms on multiple opponents
q-learning and sarsa
'''

import os
import argparse
import numpy as np
from sklearn.model_selection import ParameterGrid
import shutil
import pandas as pd

import rlcard
from rlcard.agents import QLAgent, RandomAgent, ThresholdAgent, ThresholdAgent2, SARSAAgent
from rlcard.utils import set_seed, tournament, Logger, plot_curve

def train(args, agent1, alpha, gamma, epsilon_decay=0):
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
    model_path = os.path.join(args.log_dir, 'sarsa_model')
    if os.path.exists(model_path):
        shutil.rmtree(model_path)


    agent = SARSAAgent(
        env,
        os.path.join(
            args.log_dir,
            'sarsa_model',
        ),
        alpha,
        gamma,
    )
    agent.load()

    if agent1 == 'random':
        eval_env.set_agents([
            agent,
            RandomAgent(num_actions=env.num_actions),
        ])

        env.set_agents([
            agent,
            RandomAgent(num_actions=env.num_actions),
        ])
    elif agent1 == 'th1':
        eval_env.set_agents([
            agent,
            ThresholdAgent(num_actions=env.num_actions),
        ])

        env.set_agents([
            agent,
            ThresholdAgent(num_actions=env.num_actions),
        ])
    elif agent1 == 'th2':
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

    df = pd.read_csv(csv_path)
    avg_score = df['reward'].tail(args.evaluate_every).mean()

    return csv_path, avg_score

def evaluate_hyperparameters(args, agent, alpha, gamma):
    #path, avg_score = train(args, agent, alpha, gamma, e)
    path, avg_score = train(args, agent, alpha, gamma)
    #print(f"Alpha: {alpha}, Gamma: {gamma}, Epsilon_decay: {e} Avg Score: {avg_score}")
    print(f"Alpha: {alpha}, Gamma: {gamma}, Avg Score: {avg_score}")
    # Create a new folder for storing the plots
    plot_dir = os.path.join(args.log_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Save the plot for each parameter set
    plot_path = os.path.join(plot_dir, f'AGENT_{agent}_alpha_{alpha}_gamma_{gamma}.png')
    plot_curve(path, plot_path, f'AGENT_{agent}_alpha_{alpha}_gamma_{gamma}.png')

    return avg_score


if __name__ == '__main__':
    agents = []
    parameters = {}
    agents.append('random')
    agents.append('th1')
    agents.append('th2')
    for agent in agents:
        parser = argparse.ArgumentParser("Q-Learning Agent example in RLCard")
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--num_episodes', type=int, default=3000)
        parser.add_argument('--num_eval_games', type=int, default=2000)
        parser.add_argument('--evaluate_every', type=int, default=150)
        parser.add_argument('--log_dir', type=str, default='experiments/new_limit_holdem_ql_result/')

        args = parser.parse_args()

        param_grid = {
            'alpha': [0.1, 0.3, 0.5, 0.7],
            'gamma':  [0.1, 0.3, 0.5, 0.7],
            #'epsilon': [0.9, 0.95, 0.99, 0.995],  # for epsilon decay
        }

        best_score = -np.inf
        best_params = None

        for params in ParameterGrid(param_grid):
            score = evaluate_hyperparameters(args, agent, params['alpha'], params['gamma'])#, params['epsilon'])
            if score > best_score:
                best_score = score
                best_params = params
                parameters[agent] = [best_params, best_score]

    print("Best hyperparameters found: /n", parameters)