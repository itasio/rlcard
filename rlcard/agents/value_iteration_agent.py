import pprint as pp
import numpy as np
import collections

'''
P is the state space that we need for implementing value iteration
P["state1","raise"] for example captures what happens if at state1 I take action: raise.
next state is about the next state the agent will be. It's not about the other player's state
'''

# P = {

# "state1": {
#     "raise": [(0.9, "state2", 0.0, 9),           # prob of next state(ctr/sum_of_ctrs_for_this_action), next state, reward of next state, ctr(num of time visited next state)
#         (0.1, "state3", 0.5, 1)
#     ],
#     "check": [(0.1, "state2", 1.0, 1),
#         (0.8, "state3", -1.0, 8),
#         (0.1, "state4", 0.0, 1)
#     ]
# },

# "state2": {
#     "call": [(0.9, "state4", 0.0, 9), # prob of next state, next state, reward of next state, ctr(num of time visited next state)
#         (0.1, "state3", 0.5, 1)
#     ],
#     "fold": [(0.1, "state3", 1.0, 1),
#         (0.8, "state3", -1.0, 8),
#         (0.1, "state4", 0.0, 1)
#     ]
# }

# }

class ValueIterAgent:
    ''' A random agent. Random agents is for running toy examples on the card games
        Initially will play some rounds randomly to estimate 
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
        self.P = collections.__dict__
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

    def traverse_tree():
        pass

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
    

if __name__ == '__main__':
    x = collections.defaultdict(list)
    x[0]=1
    x[1].append(2)
    print(x)
    print(x[0])
    print(x[1])
    # Q = np.zeros((3, 4))
    # print(Q)
    x[2]=3
    print(x.keys())
    print(x.values())
    print(x.items())
    # while True:
    #     for i in range(len(x)):
    #         print(x[i])
    #         if i == len(x)-1:
    #             x[len(x)]= 4
    #             # print(x[3])
    P = collections.__dict__
    P = {

    "state1": {
        "raise": [(0.9, "state2", 0.0, 9),           # prob of next state(ctr/sum_of_ctrs_for_this_action), next state, reward of next state, ctr(num of time visited next state)
            (0.1, "state3", 0.5, 1)
        ],
        "check": [(0.1, "state2", 1.0, 1),
            (0.8, "state3", -1.0, 8),
            (0.1, "state4", 0.0, 1)
        ]
    },

    "state2": {
        "call": [(0.9, "state4", 0.0, 9), # prob of next state, next state, reward of next state, ctr(num of time visited next state)
            (0.1, "state3", 0.5, 1)
        ],
        "fold": [(0.1, "state3", 1.0, 1),
            (0.8, "state3", -1.0, 8),
            (0.1, "state4", 0.0, 1)
        ]
    }

    }
    P["state3"] = 5
    print(P.keys())
    i = "state3"
    if i in P.keys():
        print("yes")
    pp.pprint(P)
    k = P["state1"]["raise"]    # list
    print(k)

    if "state2" in k[0]:
        print("yes")
    else:
        print("no")
    Q = np.zeros((2, 4), dtype=np.float64)
    print(Q)
