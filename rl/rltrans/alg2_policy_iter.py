#！ -*- coding:utf-8 -*-

import numpy as np
from math import abs
import random

gamma = 0.1
theta = 1e-7

def policy_iter():
    """
    """

    # initialize
    V = [0 for s in S]
    pi = []

    # policy iterate
    while True:
        # policy evaluate
        while True:
            delta = 0
            for s in S:
                v = V[s]
                V[s] = 0
                for nx_s in S:
                    for r in R:
                        tmp = p(nx_s,r,s,a) * (r + gamma * V[nx_s])
                        V[s] += tmp
                delta = max(delat, abs(V[s]-v))
            if delta < theta:
                break

        # policy improvement
        while True:
            stable = True
            for s in S:
                old_a = pi(s)
