#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 08:27:01 2018

@author: oster
"""

# Temporal Difference learning with TD(0) update
#with value_function: (approximated) value function
#state_0: current state
#state_1: next choosen state
#reward: immeadiate reward of next choosen state
#alpha: step-size/learning parameter
#gamma: discount factor
def TD_0_update(value_function, state_0, state_1, reward, alpha, gamma):
    return value_function(state_0)+alpha*(reward+gamma*value_function(state_1)-value_function(state_0))


#n-step TD
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
                
                
        