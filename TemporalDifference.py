#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 08:27:01 2018

@author: oster
"""
import numpy as np
# Temporal Difference learning with TD(0) update
#with value_function: (approximated) value function
#state_0: current state
#state_1: next choosen state
#reward: immeadiate reward of next choosen state
#alpha: step-size/learning parameter
#gamma: discount factor
def TD_0_update(value_function, state_0, state_1, reward, alpha, gamma):
    return value_function(state_0)+alpha*(reward+gamma*value_function(state_1)-value_function(state_0))


#n-step TD model learning (assumming opponent uses same policy)
#with value_function: (approximated) value function
#state_0: current state
#state_follow: list of next n choosen states
#reward: list of immeadiate reward of next n choosen state
#alpha: step-size/learning parameter
#gamma: discount factor
#
def n_step_TD(value_function, state_0, state_follow, reward,alpha,gamma):
    n=len(reward)
    G=0
    for i in range(n):
        G+=gamma**(i)*reward[i]
    G=G+gamma**n*value_function(state_follow[n-1])
    val=value_function(state_0)+alpha*(G-value_function(state_0))
    return val
                
                
# semi gradient TD(0) with function approximation
def semi_gradient_TD_0(value_function, state_0, state_1, reward, alpha, gamma,weights):
    x_0 = np.array(np.reshape(state_0,[1,9]))
    x_0=x_0[0]
    grad = np.zeros((9,9,9))
    for i in range(9):
        for j in range(9):
            for k in range(9):
                grad[i][j][k]=x_0[i]*x_0[j]*x_0[k]
    delta_weights=weights+alpha*(reward+gamma*value_function(state_1)-value_function(state_0))*grad
    return delta_weights
    
def tensor_alternativ(value_function, state_0, state_1, reward, alpha, gamma,weights):
    grad = np.zeros((3,3,3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            grad[i,j,k,l,m,n] = state_0[0,i]*state_0[1,j]*state_0[2,k]*state_0[l,0]*state_0[m,1]*state_0[n,2]
    delta_weights=weights+alpha*(reward+gamma*value_function(state_1)-value_function(state_0))*grad
    return delta_weights