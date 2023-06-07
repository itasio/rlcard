import numpy as np


class ThresholdAgent(object):
    ''' A threshold agent. He will be playing new-limit-holdem.
        Will bet the maximum amount allowed in each round, provided that it has at least a high enough "combination".
        In round 1 it will always bet/raise with a K or Ace.
        In round 2 it will always bet/raise with any pair.
    '''

    def __init__(self, num_actions):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        # return np.random.choice(list(state['legal_actions'].keys()))
        legal_actions = state['raw_legal_actions'];

        if len(state['raw_obs']['public_cards']) == 0:  #we are on 1st round i.e. no public cards
            hand = state['raw_obs']['hand'][0]
            if hand == 'SA' or hand == 'HA' or hand == 'DA' or hand == 'CA' or\
            hand == 'SK' or hand == 'HK' or hand == 'DK' or hand == 'CK':
                #high enough combination
                #i will play as offensive as I can i.e. 1raise 2. call 3. check 
                if 'raise' in legal_actions:
                    return 1
                elif 'call' in legal_actions:
                    return 0
                elif 'check' in legal_actions:
                    return 3
                else:   #fold
                    return 2
            else:   # play randomly
                x = np.random.choice(list(state['legal_actions'].keys()))
                return x
        else:   #we are on 2nd round and no matter the card i have i play the same style as before
            if 'raise' in legal_actions:
                return 1
            elif 'call' in legal_actions:
                return 0
            elif 'check' in legal_actions:
                return 3
            else:   #fold
                return 2

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the threshold agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        # probs = [0 for _ in range(self.num_actions)]
        # for i in state['legal_actions']:
        #     probs[i] = 1/len(state['legal_actions'])

        info = {}
        #info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}

        return self.step(state), info
