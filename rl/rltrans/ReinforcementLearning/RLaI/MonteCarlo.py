import random.random as random
import random.sample as sample
from numpy import argmax


# first-visit MC prediction estimating state values of a policy
def fvmc(policy, states):
    # 1.initialization
    values = {state:random() for state in states}
    returns = {state:[] for state in states}

    # Loop each episode
    while True:
        states, actions, rewards = generate_episode(policy)
        G = 0
        T = len(episode) / 3
        for t  in reverse(range(0,T)):
            G = G + rewards[t]
            if states[t] in states[:t]:
                returns[states[t]].append(G)
                values[states[t]] = sum(returns[states[t]]) / len(returns[states[t]])
    return values


# on-policy first-visit MC control(for epsilon-soft policies)
def onfvmc(actions, epsilon):
    """
    Input:
        actions: the dict of states:available-actions
        epsilon: the threshold of a 
    Output:
        return an approxmation of a epsilon-greedy policy
    """
    # Initialize
    states = actions.keys()
    values = {s:{a:random() for a in actions[s]} for s in states}
    returns = {s:{a:[] for a in actions[s]} for s in states}
    def epsilon_policy(state): # Arbitrary epsilon greedy policy
        if random() <= epsilon:
            return sample(actions[state])
        else:
            maxind = argmax(values[state])
            return actions[state][maxind]

    # Improvement
    while True:
        S, A, R = generate_episode()
        G = 0
        for t in reverse(range(len(S))):
            G = G + R[t]
            if (S[t],A[t]) in [(S[i]:A[i]) for i in range(t)]:
                returns[S[t]][A[t]].append(G)
                values[S[t]][A[t]] = sum(returns[S[t]][A[t]]) / len(returns[S[t]][A[t]])
