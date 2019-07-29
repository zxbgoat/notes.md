#! -*- coding:utf-8 -*-

from math import abs

gamma = 0.1
theta = 1e-7

def iter_policy_eva(pi, p, S, A, R):
    """
        Usage: evaluate how good is a policy
        Invoke: V = iter_policy_eva(pi, p)
        Parameters: pi is the policy, determining pr(a | s)
                    p is the dynamics of the enviroment,
                        determining pr(nx_s,r | s, a)
                    S: list of states
                    A: list of actions
                    R: list of rewards
        Result: return the array of the value of all states
    """

    # initialize
    V = [0 for s in S]
    # iterate
    while True:
        delta = 0
        for s in S:
            v = V[s]
            V[s] = 0
            for a in actions:
                pi_a_s = pi(a,s) # get the probability of pr(a|s)
                for nx_s in S:
                    for r in R:
                        tmps = p(nx_s,r,s,a)*[r+gamma*V[nx_s]]
                tmpa = pi_a_s * tmps
            V[s] += tmpa
            delta = max(delta, math.abs(V[s]-v))

        # judge if stop iteration
        if delta < theta:
            break
