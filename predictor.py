#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 08:35:50 2018

@author: oster
"""
import numpy as np
#epsilon - greedy - methods with
#value_funciton: (approximated) value function
#moves: list of possible moves/next states
#eps: epsilon determining whhen to take a random not optimal step for exploration
#minmax: spicifies if value function should be minimized or maximized
def eps_greed(value_function, moves, eps, minmax):
    v=-2
    action=[]
    for m in moves:
        if v<= minmax*value_function(m):
           v=minmax*value_function(m)
           action.append(m)
    if (np.random.rand(1,1)>eps):
        final_move= action[np.random.randint(0,len(action))]
    else: 
        final_move = moves[np.random.randint(0,len(moves))]
    return np.matrix(final_move)

#rondom action as comparsion model
def randomAction(moves):
    return np.matrix(moves[np.random.randint(0,len(moves))])