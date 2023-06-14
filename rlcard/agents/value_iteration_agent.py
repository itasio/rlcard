import numpy as np
import collections

class ValueIterAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
        Initially will play some rounds randomly to estimate 
    '''

    def __init__(self, env, model_path='./vi_model', e=0.8, conv_limit=1e-10):
        ''' Initilize the value iteration agent

        Args:
            env (Env): Env class
        '''
        self.use_raw = False
        self.env = env
        self.conv_limit = conv_limit
        self.discount_factor = e
        self.model_path = model_path
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            if is_done:
                self.state = self.env.reset() 
            else: 
                self.state = new_state

    def train(self):
        self.play_n_random_steps

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
    x.keys().__len__
    for i in range():
        print(i)
        if i == 1:
            x[3]= 4
            # print(x[3])
