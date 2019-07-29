import random as rd
import math

INFINITY = 1000000

def nstep_TDestimate(pi, SA, n, ter_s, alpha=0.1, gamma=1, ini_s=None):
    """
    Use n-step TD to estimate the state values of a policy
    parameters:
        pi: a policy function to be estimated
        SA: a dict of state:actions
        n: step number to look ahead
        ter_s = a list of terminal states
        alpha: step size
        ini_s: a list of start states
        gamma: the discount
    """

    S = SA.keys()

    def init():
        V = {s:rd.random() for s in SA.keys()}
        if ini_s == None:
            st = rd.sample(SA.keys)
            while st in ters:
                st = rd.sample(SA.keys)
        else:
            st = rd.sample(ini_s)
        St = []
        Rt = []

    while True:
        St.append(st)
        T = INFINITY
        t = 0
        while st not in ter_s:
            if t < T:
                rt_, st_ = pi(st)
                Rt.append(rt_)
                St.append(st_)
                if st_ in ter_s:
                    T = t+1
            tau = t-n+1
            if tau >= 0:
                G = sum([mt.pow(gamma, i-tau-1) for i in min(tau+n, T)])
                if tau+n < T:
                    G += mt.pow(gamma, n)*V[St[tau+n]]
                    V[St[tau]] += alpha * (G-V[St[tau]])

def nstep_sarsa()
