import pprint as pp
import numpy as np
import collections
from scipy.special import softmax
import os
import pickle

from rlcard.utils.utils import *
'''
P is the state space that we need for implementing value iteration
P["state1","raise"] for example captures what happens if at state1 I take action: raise.
next state is about the next state the agent will be. It's not about the other player's state
'''

# P = {

# "state1": {
#     "raise": [  [0.9, "state2", 0.0, 9],           # prob of next state(ctr/sum_of_ctrs_for_this_action), next state, reward of next state, ctr(num of time visited next state)
#                 [0.1, "state3", 0.5, 1]
#     ],
#     "check": [ [0.1, "state2", 1.0, 1],
#                [0.8, "state3", -1.0, 8],
#                [0.1, "state4", 0.0, 1]
#     ]
# },

# "state2": {
#     "call": [[0.9, "state4", 0.0, 9], # prob of next state, next state, reward of next state, ctr(num of time visited next state)
#         [0.1, "state3", 0.5, 1]
#     ],
#     "fold": [[0.1, "state3", 1.0, 1],
#         [0.8, "state3", -1.0, 8],
#         [0.1, "state4", 0.0, 1]
#     ]
# }

# }

class ValueIterAgent:
    ''' An agent that will play according to value iteration algorithm,
        in order to find optimal policy
    '''

    def __init__(self, env, model_path='./vi_model', e=0.8, conv_limit=1e-10):
        ''' Initilize the value iteration agent

        Args:
            env (Env): Env class
        '''
        self.use_raw = 2
        self.env = env
        self.conv_limit = conv_limit
        self.discount_factor = e
        self.agent_id = 0
        self.model_path = model_path
        self.iteration = 0
        self.P = collections.__dict__               # state space
        self.V = collections.defaultdict(float)    # value function for each state
        self.Q = collections.defaultdict(list)     # Q table
    

    def train(self):
        ''' Do one iteration of value iteration
        '''
        self.iteration += 1
        self.env.reset()
        self.find_agent()
        self.traverse_tree()

    def find_agent(self):
        agents = self.env.get_agents()
        for id, agent in enumerate(agents):
            if isinstance(agent, ValueIterAgent):
                self.agent_id = id
                break

    def traverse_tree(self):
        if self.env.is_over():
            chips = self.env.get_payoffs()
            return chips[self.agent_id]

        current_player = self.env.get_player_id()
        # compute the quality of previous state
        if not current_player == self.agent_id:
            state = self.env.get_state(current_player)
            # other agent move
            action = self.env.agents[current_player].step(state)

            # Keep traversing the child state
            self.env.step(action)
            Vstate = self.traverse_tree()
            self.env.step_back()
            return Vstate
        
        if current_player == self.agent_id:
            obs, legal_actions = self.get_state(current_player)
            # if first time we encounter state initialize qualities
            self.update_P_and_Q(obs, legal_actions, self.P, self.Q)

    @staticmethod
    def step(state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        return np.random.choice(list(state['legal_actions'].keys()))

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        probs = [0 for _ in range(self.num_actions)]
        for i in state['legal_actions']:
            probs[i] = 1/len(state['legal_actions'])

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}

        return self.step(state), info
    
    def update_P_and_Q(self, obs, legal_actions):
        '''
        For State Space P:
            1) add new state and actions for it, or
            2) update list of legal actions for existing state (add actions that are not already in the list)
                    
        For Q table:
            1) Add new state and rewards for its legal actions(set to zero) or
            2) Update rewards (set to zero) of legal actions for existing state 

        Args:
            obs (str): state_str
            legal_actions (list): List of legel actions
        '''
        # if existing state
        if obs in self.P.keys() and obs in self.Q.keys():
            for action in legal_actions:            # check all listed actions that can be done in this state
                if action not in self.P[obs].keys():    # if any not listed in P so far add it
                    # initialize now will change later
                    self.P[obs][action] = [[0,0,0,0]]   #prob_next_st, next_st, rew_next_st, num_visited_next_st
                # now reset rewards of Q table for legal actions, will be recalculated 
                self.Q[obs][action] = 0
        else:
            # new state found, add it to dicts and set appropriate values
            self.P[obs] = {}
            self.Q[obs] = [-np.inf, -np.inf, -np.inf, -np.inf]
            for action in legal_actions:
                self.P[obs][action] = [[0,0,0,0]]
                self.Q[obs][action] = 0
            

        if obs not in policy.keys() and obs not in self.qualities.keys():
            tactions = np.array([-np.inf for action in range(self.env.num_actions)])
            for action in range(self.env.num_actions):
                if action in legal_actions:
                    tactions[action] = 0
            self.qualities[obs] = tactions
            action_probs = softmax(tactions)
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs].copy()
        action_probs = remove_illegal(action_probs, legal_actions)
    
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

if __name__ == '__main__':
    x = collections.defaultdict(list)
    x["state1"].append(2)
    x["state1"].append(3)
    x["state1"].append(4)
    x["state1"].append(2)

    x["state2"].append(2)
    x["state2"].append(5)
    x["state2"].append(2)
    x["state2"].append(6)
    x["state2"][0]=10
    x["ff"]=[]
    ll = x.keys()
    print(x)
    print(x.keys())
    for i in x:
        print(i)
        print(x[i])
        print(max(x[i]), x[i].index(max(x[i])))
        # print max of each state and the index of it
    # print(x)
    # print(x["state1"])
    # print(x["state2"])
    # Q = np.zeros((3, 4))
    # print(Q)
    # x[2]=3
    # print(x.keys())
    # print(x.values())
    # print(x.items())
    # while True:
    #     for i in range(len(x)):
    #         print(x[i])
    #         if i == len(x)-1:
    #             x[len(x)]= 4
    #             # print(x[3])
    P = collections.__dict__
    P = {

    "state1": {
        "raise": [[0.9, "state2", 0.0, 9],           # prob of next state(ctr/sum_of_ctrs_for_this_action), next state, reward of next state, ctr(num of time visited next state)
            [0.1, "state3", 0.5, 1]
        ],
        "check": [[0.1, "state2", 1.0, 1],
            [0.8, "state3", -1.0, 8],
            [0.1, "state4", 0.0, 1]
        ]
    },

    "state2": {
        "call": [[0.9, "state4", 0.0, 9], # prob of next state, next state, reward of next state, ctr(num of time visited next state)
            [0.1, "state3", 0.5, 1]
        ],
        "fold": [[0.1, "state3", 1.0, 1],
            [0.8, "state3", -1.0, 8],
            [0.1, "state4", 0.0, 1]
        ]
    }

    }
    P["state3"] = 5
    print(P.keys())
    i = "state3"
    if i in P:
        print("yes")
    pp.pprint(P)
    k = P["state1"]["raise"]    # list
    print(k)

    if "state2" in k[0]:
        print("yes")
    else:
        print("no")
    if "bla" not in P["state1"].keys():
        P["state1"]["bla"]=[[0,0,0,0]]
    P["state1"]["bla"].append([1,1,1,1])
    P["state1"]["bla"][0][0] = "hey"
    pp.pprint(P)
    # Q = np.zeros((2, 4), dtype=np.float64)
    # Q[0][0] = 2
    # Q[0][3] = 3
    # Q[1][0] = 4

    # print(np.max(Q, axis=1))   #returns the max of each row i.e. [3,4]
    