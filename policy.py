import numpy as np
from enum import Enum

class Action(Enum):
    BUY = 1
    SELL = -1

# e.g) stock A,B are explained by 2-quantile [val,growth]
# state = [[1,1],[1,2]] 
# rewards = [[[0.5,0.3,-0.2,0]],[[-0.1,0.3, 0,0.5]]]
def eps_greedy(states, rewards, epsilon=0.1):
    import random
    actions = [] # [Up,down,...]
    trans_probs = [] # [{'Up':1},{'Up':0.5,'Down':0.5},...]
    q_num = len(rewards[0])

    # to-consider : epsilon-greedy for all stocks
    if random.random() < epsilon:
        for i in np.random.uniform(size=len(states)):
            if i >= 0.5:
                actions.append(Action.BUY)
            else :
                actions.append(Action.SELL)
            trans_probs.append({'BUY':0.5, 'SELL':0.5})
    else:
        for i in range(0,len(states)):
            if rewards[i][states[i][0] - 1][states[i][1] - 1] >= 0:
                actions.append(Action.BUY)
                trans_probs.append({'BUY':1.0})
            else:
                actions.append(Action.SELL)
                trans_probs.append({'SELL':1.0})
    return actions, trans_probs


if __name__ == '__main__':
    states = [[1,1]]
    rewards = [[[0.5,0.3],[-0.2,0]]]
    for i in range(0,5):
        print(eps_greedy(states,rewards))
        
    states = [[1,1],[1,2]]
    rewards = [[[0.5,0.3],[-0.2,0]],[[-0.1,0.3], [0,0.5]]]
    for i in range(0,5):
        print(eps_greedy(states,rewards))