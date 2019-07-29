import random as rd


alpha = 0.5
epsilon = 0.1
EP_NUM = 8000
W = 10
H = 7
O = (3, 0)
G = (0, 7)
A = ('up', 'down', 'left', 'right')


def init():
    Q = [[{a:rd.random() for a in A} for i in range(H)] for j in range(W)]
    for key in Q[G[0]][G[1]].keys():
        Q[G[0]][G[1]][key] = 0.0


def keymax(d):
    from numpy import argmax
    keys = d.keys()
    values = d.values()
    return keys[argmax(values)]


def choose_action(S):
    if rd.random() < epsilon:
        return rd.sample(A)
    else:
        return keymax(Q[S[0]][S[1]])


def take_action(S, A):
    if A=='up':
        S[0] += 1
    elif A == 'down':
        S[0] -= 1
    elif A == 'left':
        S[1] -= 1
    else:
        S[1] += 1
    if S[1] in [3,4,5,8]:
        S[0] += 1
    elif S[1] in [6, 7]:
        S[0] += 2
    if S == G:
        return 0, S
    else:
        return -1, S


init()
for eps in range(EPS_NUM):
    S = O
    while S != G:
        A = choose_action(S)
        R, S_ = take_action(A)
        A_ = choose_action(S_)
        Q[S[0]][S[1]][A] += alpha * (R+Q[S_[0]][S_[1]][A_]-Q[S[0]][S[1]][A])
        S = S_
        A = A_
