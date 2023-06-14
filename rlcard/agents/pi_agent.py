import collections
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
        '''

        self.gamma = g
        self.agent_id = 0
        self.use_raw = False
        self.env = env


        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict(list)

        self.iteration = 0

    def train(self):
        ''' Find optimal policy
        '''
        while True:
            self.iteration += 1
            old_policy = self.policy.copy()
            self.evaluate_policy()
            if self.compare_policys(old_policy, self.policy):
                break
        print('Optimal policy found: State space length: %d after %d iterations' % (len(self.policy), self.iteration))

    def compare_policys(self, p1, p2):
        if p1.keys() != p2.keys():
            return False

        for key in p1:
            if not np.array_equal(p1[key], p2[key]):
                return False

        return True
    def find_agent(self):
        agents = self.env.get_agents()
        for id, agent in enumerate(agents):
            if isinstance(agent, PIAgent):
                self.agent_id = id
                break


    def evaluate_policy(self):
        self.find_agent()
        suit = 'S'
        rank_list = ['A', 'T', 'J', 'Q', 'K']
        for rank1 in rank_list:
            for rank2 in rank_list:
                for rank3 in rank_list:
                    self.env.reset(self.agent_id, self.agent_id, Card(suit,rank1), Card(suit,rank2), Card(suit,rank3))
                    self.traverse_tree()
        for rank in rank_list:
            player = (self.agent_id + 1) % self.env.num_players
            for rank1 in rank_list:
                for rank2 in rank_list:
                    for rank3 in rank_list:
                        self.env.reset(player, self.agent_id, Card(suit, rank1), Card(suit, rank2),
                                       Card(suit, rank3))
                        self.traverse_tree()


    def traverse_tree(self):
        if self.env.is_over():
            chips = self.env.get_payoffs()
            return chips[self.agent_id]

        current_player = self.env.get_player_id()
        # compute the quality of previous state
        if not current_player == self.agent_id:
            state = self.env.get_state(current_player)
            # other agent move
            obs, legal_actions = self.get_state(current_player)
            action_probs = self.action_probs2(legal_actions)
            Vstate = 0
            for action in legal_actions:
                prob = action_probs[action]
                # Keep traversing the child state
                self.env.step(action)
                Vstate += self.traverse_tree()*prob
                self.env.step_back()
            return Vstate

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
            self.improve_policy(obs, quality, legal_actions)

        return Vstate * self.gamma

    def improve_policy(self, obs, quality, legal_actions):
        best_action = max(quality, key=quality.get)

        new_policy = np.array([0 for action in range(self.env.num_actions)])
        new_policy[best_action] = 1

        self.policy[obs] = new_policy


    def action_probs(self, obs, legal_actions, policy):
        ''' Obtain the action probabilities of the current state

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
        # if new state initialize qualities and policy
        if obs not in policy.keys():
            tactions = np.array([-np.inf for action in range(self.env.num_actions)])
            for action in range(self.env.num_actions):
                if action in legal_actions:
                    tactions[action] = 0
            best_action = np.argmax(tactions)
            action_probs = np.array([0 for action in range(self.env.num_actions)])
            action_probs[best_action] = 1
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs].copy()
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def action_probs2(self, legal_actions):
        '''Get the action probs of the opponent agent
        '''

        ''' random agent'''
        i = len(legal_actions)
        probs = np.zeros(self.env.num_actions)
        for a in legal_actions:
            probs[a] = 1/i
        return probs

    def eval_step(self, state):
        ''' Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
            info (dict): A dictionary containing information
        '''

        probs = self.action_probs(state['obs'].tostring(), list(state['legal_actions'].keys()), self.policy)
        # action = np.random.choice(len(probs), p=probs)
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

