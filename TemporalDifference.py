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
