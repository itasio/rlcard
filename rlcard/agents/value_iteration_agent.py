import pprint as pp
import numpy as np
import collections
from scipy.special import softmax
import os
import pickle

#from rlcard.utils.utils import *
'''
P is the state space that we need for implementing value iteration
P["state1","raise"] for example captures what happens if at state1 I take action: raise.
next state is about the next state the agent will be. It's not about the other player's state
'''

# P = {

# "state1": {
#     "raise": [ { "state2": [0.9, 0.0, 9],           # next state: [prob of next state, reward of next state, num of times visited next state right after state1] 
#                  "state3": [0.1, 0.5, 1]
#                }, ctr                           ctr: (num of times action raise was taken when in state1)
#     ],
#     "check": [ { "state2": [0.1, 1.0, 1],
#                  "state3": [0.8, -1.0, 8],
#                  "state4": [0.1, 0.0, 1]
#                }, ctr
#     ]
# },

# "state2": {
#     "call": [ { "state6": [0.5, 0.0, 5],           # next state: [prob of next state, reward of next state, num of times visited next state right after state1] 
#                  "state3": [0.5, 0.5, 5]
#                }, ctr                           ctr: (num of times action raise was taken when in state1)
#     ],
#     "fold": [ { "state5": [0.1, 1.0, 1],
#                  "state3": [0.8, -1.0, 8],
#                  "state4": [0.1, 0.0, 1]
#                }, ctr
#     ]
# }

# }

class ValueIterAgent:
    ''' An agent that will play according to value iteration algorithm,
        in order to find optimal policy
    '''

    def __init__(self, env, model_path='./vi_model', gamma=0.8, conv_limit=1e-10):
        ''' Initilize the value iteration agent

        Args:
            env (Env): Env class
        '''
        self.use_raw = 2
        self.env = env
        self.conv_limit = conv_limit
        self. conv = False
        self.gamma = gamma
        self.agent_id = 0
        self.model_path = model_path
        self.iteration = 0
        self.states_discovered = 0
        self.P = collections.defaultdict(dict)              # state space
        self.V = collections.defaultdict(float)    # value function for each state (expected return of the best action for each state)
        self.Q = collections.defaultdict(list)     # Q table
    

    def train(self):
        ''' Do one iteration of value iteration
        '''
        self.iteration += 1
        self.states_discovered = len(self.P)
        self.env.reset()
        self.find_agent()
        self.traverse_tree()
        if self.conv:
            print('Value iteration converged after {} iterations'.format(self.iteration))

    def find_agent(self):
        agents = self.env.get_agents()
        for id, agent in enumerate(agents):
            if isinstance(agent, ValueIterAgent):
                self.agent_id = id
                break

    def traverse_tree(self):
        if self.env.is_over():
            chips = self.env.get_payoffs()
            current_player = self.env.get_player_id()   # TODO if opponent folded who is curr player?
            if current_player == self.agent_id:
                obs, legal_actions = self.get_state(current_player)
                return chips[self.agent_id], obs
            else:
                return chips[self.agent_id], "other player"

        current_player = self.env.get_player_id()
        # compute the quality of previous state
        if not current_player == self.agent_id:
            state = self.env.get_state(current_player)
            # other agent move
            action = self.env.agents[current_player].step(state)

            # Keep traversing the child state
            self.env.step(action)
            Vstate, next_state = self.traverse_tree()
            if self.conv:
                return Vstate, next_state
            self.env.step_back()
            return Vstate, next_state
        
        if current_player == self.agent_id:
            obs, legal_actions = self.get_state(current_player)
            # update state space, V and Q table (initializing only)
            self.update_P_and_Q_and_V(obs, legal_actions)
            q_mean = 0  # For holding mean q between legal actions in a state, that will be transferred to parent state of tree

            for action in legal_actions:
                # Keep traversing the child state
                self.env.step(action)
                q, next_state = self.traverse_tree()    # I want my next state, not opponent's state
                q_mean += q
                if next_state == "other player":
                    # this is my last state, game is finished and i took last action e.g. fold
                    #next_state = obs    # Next state is my current state TODO here the action taken in this state is not recorded!
                    next_state, _ = self.get_state(current_player)  # this way we pass in next state the info about action taken
                if self.conv:
                    return q, next_state
                self.env.step_back()
                
                self.P[obs][action][1] +=1  #took action when in state obs one more time
                if next_state not in self.P[obs][action][0].keys():   #next state first time recorded for current state
                    self.P[obs][action][0][next_state] = [0, q, 1] #prob of next state, reward for this state, times visited this state 
                else:   # I have visited again next state, after current state obs
                    self.P[obs][action][0][next_state][2] += 1 
                    self.P[obs][action][0][next_state][1] = (self.P[obs][action][0][next_state][1] + q) / 2     # q_new = (q_old + q) / 2
                
                for i in self.P[obs][action][0]:    #calculate again probabilities of each recorded next state when in current state obs and taken certain action
                    self.P[obs][action][0][i][0] = self.P[obs][action][0][i][2] / self.P[obs][action][1]    #times visited next state/sum of all visits

                for item in self.P[obs][action][0].items(): # for every next state after current state obs taking certain action 
                    prob_next_st, rew_next_st, ctr = item[1]    
                    nxt_st = item[0]
                    self.Q[obs][action] += prob_next_st * (rew_next_st + self.gamma * self.V[nxt_st])
                
                #equivalent with the above
                # for nxt_st in self.P[obs][action][0]:   # for every next state after current state obs taking certain action 
                #     prob_next_st, rew_next_st, ctr = self.P[obs][action][0][nxt_st]
                #     self.Q[obs][action] += prob_next_st * (rew_next_st + self.gamma * self.V[nxt_st])

                ll = list(self.Q.values())  # list of lists with Q values of each action per state
                q_vals = np.max(ll, axis = 1)    #maximum expected reward for each state as calculated in Q table
                v_vals = list(self.V.values())
                if np.max(np.abs(np.subtract(q_vals,v_vals))) < self.conv_limit and self.states_discovered == len(self.P):
                    # converged and also no new states discovered in last training round
                    self. conv = True
                    break   # found convergence must stop
                # Since i have not converged, i set new V(s)
                for i, st in enumerate(self.Q):       # Setting V value for each state
                   # self.V[st] = np.max(self.Q[st])
                    self.V[st] = k[i]   # TODO check if it holds
            return q_mean/len(legal_actions), obs

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
    
    def update_P_and_Q_and_V(self, obs, legal_actions):
        '''
        For State Space P:
            1) add new state and actions for it, or
            2) update list of legal actions for existing state (add actions that are not already in the list)
                    
        For Q table:
            1) Add new state and rewards for its legal actions(set to zero) or
            2) Update rewards (set to zero) of legal actions for existing state 

        For V: If new state found add it to V with expected return 0
        Args:
            obs (str): state_str
            legal_actions (list): List of legel actions
        '''
        # if existing state
        if obs in self.P.keys() and obs in self.Q.keys():
            for action in legal_actions:            # check all listed actions that can be done in this state
                if action not in self.P[obs].keys():    # if any not listed in P so far add it
                    # initialize now will change later
                    #self.P[obs][action] = [[0,0,0,0]]   # {next_st: [prob_next_st, rew_next_st, num_visited_next_st]}
                    self.P[obs][action] =[{},0] # so far zero times 
                # now reset rewards of Q table for legal actions, will be recalculated 
                self.Q[obs][action] = 0
        else:
            # new state found, add it to dicts and set appropriate values
            self.Q[obs] = [-np.inf, -np.inf, -np.inf, -np.inf]

            for action in legal_actions:
                #self.P[obs][action] = [[0,0,0,0]]
                self.P[obs][action] =[{},0] # so far zero times 
                self.Q[obs][action] = 0
            self.V[obs] = 0
    
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
    v = collections.defaultdict(dict)           # P table
    v["st1"]["raise"] = [{},0]
    v["st1"]["call"] = [{},0]
    print(v)
    v["st1"]["raise"][1] +=1
    v["st1"]["raise"][0]["st2"] =[0,2,3,4]
    v["st1"]["raise"][0]["st3"] =[10,50,60,70]
    print(v)
    v["st1"]["raise"][0]["st2"][0] =1
    pp.pprint(v)
    v["st2"]["raise"] = [{},0]
    pp.pprint(v)
    print(v["st1"].keys())
    for nxt_st in v["st1"]["raise"][0]:
        prob, rew, ctr,cc = v["st1"]["raise"][0][nxt_st]
        print(nxt_st, prob, rew, ctr,cc)
    for i in v["st1"]["raise"][0].items():
        prob, rew, ctr,cc = i[1]
        print(i[0])
        print(i[1])


    qq = collections.defaultdict(list)      # Q table
    qq["st1"] = [1,2,3,4]
    qq["st2"] = [5,6,2,3]
    ll = list(qq.values())
    print(ll)
    k = np.max(ll, axis = 1)    #to be assigned to V
    print(k)

    fl = collections.defaultdict(float)     # V table
    fl["st1"] = 0
    fl["st2"] = 0
    for st in qq:
        fl[st]  =np.max(qq[st])
    print(fl)


