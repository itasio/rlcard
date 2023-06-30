import collections

import numpy as np
from numpy import random
from scipy.special import softmax
import os
import pickle

from rlcard.utils.utils import *


class PIAgent:
    ''' Implement policy - iteration algorithm
    '''

    def __init__(self, env, g=1):
        ''' Initialize Agent
dp
         Args:
         env (Env): Env class
         converge se 4 iterations
        '''

        self.gamma = g
        self.agent_id = 0
        self.use_raw = False
        self.env = env
        self.rank_list = ['A', 'T', 'J', 'Q', 'K']
        self.card_prob = 0.2

        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict(list)
        self.state_values = collections.defaultdict(list)
        self.iteration = 0
        self.flag = 0
        self.flag2 = 0
        self.rank = None
        self.public_ranks = None

    def train(self, episodes=None):
        ''' Find optimal policy
        '''
        while True:
            #k += 1
            self.iteration += 1
            print('Current iteration: %d' % self.iteration)
            old_policy = self.policy.copy()
            self.evaluate_policy()
            if self.compare_policys(old_policy, self.policy):
                break
            if self.iteration == 10:
                break
        print('Optimal policy found: State space length: %d after %d iterations' % (len(self.policy), self.iteration))
        self.remake_policy()

    def remake_policy(self):
        ''' Take the policy that has key: tuple(obs, opponent_card, public_cards) and for every obs compute
        average policy for all possible opponent cards in key: obs
        We take the first key then we find a match and store it to send to policy sum
        '''
        new_policy = collections.defaultdict(list)
        old_policy = self.policy

        for key1 in old_policy:
            if isinstance(key1, tuple):
                obs1 = key1[0]
            else:
                obs1 = key1

            if obs1 in new_policy:
                continue

            same_obs_values = []
            for key2 in old_policy:
                if isinstance(key2, tuple):
                    obs2 = key2[0]
                else:
                    obs2 = key2
                if obs1 == obs2:
                    same_obs_values.append(old_policy[key2])

            if same_obs_values:
                new_policy = self._policy_sum(new_policy, same_obs_values, obs1)
            else:
                print('10')
        self.policy = new_policy


    @staticmethod
    def _policy_sum(policy, same_obs_values, obs):
        ''' Gets the average mean policy when given different policies
        args:
        policy: contains the local policy
        same_obs_values: list of np arrays containing probs for every action
        obs: key for policy
        '''
        average_values = np.mean(same_obs_values, axis=0)
        policy[obs] = average_values
        return policy

    def compare_policys(self, p1, p2):
        ''' Compare the policies given
        If they have different number of keys they are different
        Else check every policy's probs to be the same
        Get the total different state policys and print it
        '''
        if p1.keys() != p2.keys():
            print('Dif number of policy keys')
            return False
        count = 0
        for key in p1:
            if not np.array_equal(p1[key], p2[key]):
                count += 1
        if count > 0:
            print('Changes in policy: %d' % count)
            return False
        return True

    def find_agent(self):
        ''' Check if our agent starts first or
        '''
        agents = self.env.get_agents()
        for id, agent in enumerate(agents):
            if isinstance(agent, PIAgent):
                self.agent_id = id
                break

    def evaluate_policy(self):
        '''We traverse the tree for every possible combination of cards(agent card, public cards and opponent card)
        and also every starting possition so we can explore all the state space
        We the total Value of our iteration with current policy so we can know if it is working
        '''
        self.find_agent()
        suit = 'S'
        Vtotal = 0
        for rank1 in self.rank_list:
            for rank2 in self.rank_list:
                for rank3 in self.rank_list:
                    for rank4 in self.rank_list:
                        self.env.reset(self.agent_id, self.agent_id, Card(suit, rank1), Card(suit, rank2),
                                       Card(suit, rank3), Card(suit, rank4))
                        self.rank = rank4
                        self.public_ranks = (rank2, rank3)
                        Vtotal += self.traverse_tree()
        player = (self.agent_id + 1) % self.env.num_players
        for rank1 in self.rank_list:
            for rank2 in self.rank_list:
                for rank3 in self.rank_list:
                    for rank4 in self.rank_list:
                        self.env.reset(player, self.agent_id, Card(suit, rank1), Card(suit, rank2),
                                       Card(suit, rank3), Card(suit, rank4))
                        self.rank = rank4
                        self.public_ranks = (rank2, rank3)
                        Vtotal += self.traverse_tree()
        print('Total value: %d' % Vtotal)
        return Vtotal


    def traverse_tree(self):
        if self.env.is_over():
            chips = self.env.get_payoffs()
            return chips[self.agent_id]

        self.roundzero()

        current_player = self.env.get_player_id()
        # compute the q of previous state
        if not current_player == self.agent_id:
            obs, legal_actions = self.get_state(current_player)
            state = self.env.get_state(current_player)
            action_probs = self.env.agents[current_player].get_action_probs(state, self.env.num_actions)
            Vstate = 0
            for action in legal_actions:
                prob = action_probs[action]
                # Keep traversing the child state
                self.env.step(action)
                v = self.traverse_tree()
                Vstate += v * prob
                self.env.step_back()
            return Vstate*self.gamma

        if current_player == self.agent_id:
            quality = {}
            Vstate = 0
            obs, legal_actions = self.get_state(current_player)
            # if first time we encounter state initialize qualities
            action_probs = self.action_probs(obs, legal_actions, self.policy)

            for action in legal_actions:
                prob = action_probs[action]
                # Keep traversing the child state
                self.env.step(action)
                v = self.traverse_tree()
                self.env.step_back()

                quality[action] = v  # Qvalue
                Vstate += v*prob

            self.state_values[obs] = Vstate
            ''' alter policy by choosing the action with the max value'''
            self.roundzero()
            self.improve_policy(obs, quality, legal_actions)

        return Vstate * self.gamma

    def improve_policy(self, obs, quality, legal_actions):
        '''Change the policy according to the new Q values
        '''
        # best_action = max(quality, key=quality.get)
        #
        # new_policy = np.array([0 for _ in range(self.env.num_actions)])
        # new_policy[best_action] = 1

        q = np.array([-np.inf for _ in range(self.env.num_actions)])
        for i in quality:
            q[i] = quality[i]

        new_policy = softmax(q)
        if self.flag2 == 0:
            obs = (obs, self.rank)
        elif self.flag2 == 1:
            obs = (obs, self.rank, self.public_ranks)

        if obs in self.policy.keys():
            self.policy[obs] = new_policy
        else:
            print('error') # just a check

    def action_probs(self, obs, legal_actions, policy):
        ''' Obtain the action probabilities(policy) of the current state or initialize a random policy

        Args:
            obs (str): state_str
            legal_actions (list): List of leagel actions
            player_id (int): The current player
            policy (dict): The used policy
            action_values (dict): The action_values of policy

        Returns:
            (tuple) that contains:
                action_probs(numpy.array): The action probabilities
                legal_actions (list): Indices of legal actions
        '''
        if self.flag2 == 0:
            obs1 = (obs, self.rank)
        elif self.flag2 == 1:
            obs1 = (obs,  self.rank, self.public_ranks)
        # if new state initialize policy
        if obs not in policy.keys() and obs1 not in policy.keys():
            best_action = random.choice(legal_actions)
            # best_action = np.argmax(tactions)
            action_probs = np.array([0 for action in range(self.env.num_actions)])
            action_probs[best_action] = 1
            self.policy[obs1] = action_probs
        elif obs1 not in policy.keys():
            action_probs = policy[obs].copy()
        else:
            action_probs = policy[obs1].copy()
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs


    def eval_step(self, state):
        ''' Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
            info (dict): A dictionary containing information
        '''

        probs = self.action_probs(state['obs'].tostring(), list(state['legal_actions'].keys()), self.policy)
        #action = np.random.choice(len(probs), p=probs)
        action = np.argmax(probs)

        # if np.random.rand() < self.epsilon:
        #     action = np.random.choice(list(state['legal_actions'].keys()))
        # else:
        #     action = np.argmax(probs)
        #
        # self.decay_epsilon()

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: float(probs[list(state['legal_actions'].keys())[i]]) for i in
                         range(len(state['legal_actions']))}

        return action, info

    def step(self, state):
        '''step = eval.step
        '''
        return self.eval_step(state)


    def get_state(self, player_id):
        ''' Get state_str of the player

        Args:
            player_id (int): The player id

        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        '''
        state = self.env.get_state(player_id)
        return state['obs'].tostring(), list(state['legal_actions'].keys())

    def roundzero(self):
        '''Check if the game is still in first round
        if yes it raises the round flag
        '''
        if self.env.first_round():
            self.flag2 = 1
        else:
            self.flag2 = 0