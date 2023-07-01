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

    def __init__(self, env, model_path='./pi_model', g=1):
        ''' Initialize Agent
dp
         Args:
         env (Env): Env class
         converge se 4 iterations
        '''

        self.public_card_prob = None  # prob of having this set of public cards
        self.hand_card_prob = None    # prob of having this set of hand cards
        self.gamma = g                # gamma value
        self.agent_id = 0             # starting possition of agent
        self.rank_list = ['A', 'T', 'J', 'Q', 'K']
        self.use_raw = False
        self.env = env
        self.model_path = model_path


        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict(list)
        self.iteration = 0
        self.flag2 = 0               # flag that indicates weather we are in round 0 or 1
        self.rank = None
        self.public_ranks = None

    def train(self, episodes=None):
        ''' Find optimal policy
        '''
        while True:
            #k += 1
            self.iteration += 1
            print('-----------------------------------------------------------------------')
            print('Current iteration: %d' % self.iteration)
            old_policy = self.policy.copy()
            self.evaluate_policy()
            if self.compare_policys(old_policy, self.policy):
                break
            if self.iteration == 10:
                break
        print('=============================================================================')
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
                if obs1 in new_policy:
                    continue
                same_obs_values = []
                probs = np.array([])
                for key2 in old_policy:
                    if isinstance(key2, tuple):
                        obs2 = key2[0]
                        if obs1 == obs2:
                            same_obs_values.append(old_policy[key2])
                            probs = np.append(probs, key2[1])
                if same_obs_values:
                    new_policy = self._policy_sum(new_policy, same_obs_values, obs1, probs)
                else:
                    print('10')

        self.policy = new_policy


    @staticmethod
    def _policy_sum(policy, same_obs_values, obs, probs=None):
        ''' Gets the average mean policy when given different policies
        args:
        policy: contains the local policy
        same_obs_values: list of np arrays containing probs for every action
        obs: key for policy
        '''
        if probs is None:
            average_values = np.mean(same_obs_values, axis=0)
            policy[obs] = average_values
        else:
            average_values = np.average(same_obs_values, axis=0, weights=probs)
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
                        self.get_public_card_probs(rank1, rank2, rank3, rank4)
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
                        self.get_public_card_probs(rank1, rank2, rank3, rank4)
                        Vtotal += self.traverse_tree()
        print('Total value: %d' % Vtotal)
        return Vtotal


    def traverse_tree(self):
        ''' We traverse the game tree for a specific set of hand card, opponent card and public cards
        If end game return the chips won or lost(reward)
        If opponent turn get the probs for every move and play every action (except those with prob = 0):
        Vstate = Sum (action_prob*Value of next state) for every action
        If our agent plays get the the probs for every action(policy) play every possible action and update the policy
        according to the new Qvalues
        and return Vstate = Sum (action_prob*Value of next state) where action_probs based on previous policy.
        '''
        if self.env.is_over():
            chips = self.env.get_payoffs()
            return chips[self.agent_id]

        self.round_zero()

        current_player = self.env.get_player_id()
        # compute the q of previous state
        if not current_player == self.agent_id:
            obs, legal_actions = self.get_state(current_player)
            state = self.env.get_state(current_player)
            action_probs = self.env.agents[current_player].get_action_probs(state, self.env.num_actions)
            Vstate = 0
            for action in legal_actions:
                prob = action_probs[action]
                if prob == 0:
                    continue
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

            ''' alter policy by choosing the action with the max value'''
            self.round_zero()
            self.improve_policy(obs, quality)

        return Vstate * self.gamma

    def improve_policy(self, obs, quality):
        ''' Change the policy according to the new Q values
        Args:
            obs: observable state
            quality: new Qvalues, np array

        flag2 = flag that indicates that we are behind the unveal of public cards

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
            obs = (obs, self.hand_card_prob, self.rank)
        elif self.flag2 == 1:
            obs = (obs, self.public_card_prob, self.rank, self.public_ranks)

        if obs in self.policy.keys():
            #print(self.policy[obs])
            self.policy[obs] = new_policy
            #print(self.policy[obs])
        else:
            print('error')  # just a check

    def action_probs(self, obs, legal_actions, policy, eval=False):
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
        if eval:
            if obs in self.policy:
                return self.policy[obs]
            else:
                print('sok')

        if self.flag2 == 0:
            obs1 = (obs, self.hand_card_prob, self.rank)
        elif self.flag2 == 1:
            obs1 = (obs, self.public_card_prob, self.rank, self.public_ranks)
        # if new state initialize policy
        if obs not in policy.keys() and obs1 not in policy.keys():
            best_action = random.choice(legal_actions)
            # best_action = np.argmax(tactions)
            action_probs = np.array([0 for action in range(self.env.num_actions)])
            action_probs[best_action] = 1
            self.policy[obs1] = action_probs
        elif obs not in policy.keys():
            action_probs = policy[obs1].copy()
        else:
            print('1')
            action_probs = policy[obs].copy()
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

        probs = self.action_probs(state['obs'].tostring(), list(state['legal_actions'].keys()), self.policy, True)
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
        ''' step = eval.step
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

    def round_zero(self):
        ''' Check if the game is still in first round
        if yes it raises the round flag
        '''
        if self.env.first_round():
            self.flag2 = 1
        else:
            self.flag2 = 0

    def get_public_card_probs(self, handcard, pcard1, pcard2, opcard):
        ''' Set the probability of getting those public cards/handcard-opponent card
        Args: handcard: rank of ourcard
              pcard1, pcard2: public cards
              opcard: card of the opponent
        '''
        total_cards = 20
        prob1 = 4/total_cards
        total_cards = 19
        if opcard == handcard:
            prob2 = 3/total_cards
        else:
            prob2 = 4/total_cards
        self.hand_card_prob = prob1 * prob2

        total_cards = 18  # 20 - 2 already handed at first round
        available_cards = 4  # 4 for each rank
        # probability of first public card
        if pcard1 == handcard:
            available_cards -= 1
        if pcard1 == opcard:
            available_cards -= 1
        prob1 = available_cards/total_cards
        # probability of second public card
        total_cards = 17
        available_cards = 4  # 4 for each rank
        if pcard2 == handcard:
            available_cards -= 1
        if pcard2 == opcard:
            available_cards -= 1
        if pcard2 == pcard1:
            available_cards -= 1
        prob2 = available_cards/total_cards
        self.public_card_prob = self.hand_card_prob * prob1 * prob2

    def save(self):
        '''  Save model
        '''

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'), 'wb')
        pickle.dump(self.policy, policy_file)
        policy_file.close()

    def load(self):
        ''' Load model
        '''

        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'), 'rb')
        self.policy = pickle.load(policy_file)
        policy_file.close()

