#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:43:51 2018

@author: oster
"""
import TicTacToe as ttt
import numpy as np
#comparing different methods
def gameFight(method1,method2):
    p=np.random.randint(0,2)
    startPlayer = (-1)**p
    secondPlayer= -1*startPlayer
    incomplete = True
    state= np.matrix([[0,0,0],[0,0,0],[0,0,0]])
    while(incomplete):
        if(np.count_nonzero(state)==9):
            return 0

        state=ttt.executeMove(state,startPlayer,ttt.value_func, method1)
        if(ttt.winX(state)!=0):
            return 1
        
        if(np.count_nonzero(state)==9):
            return 0
        state=ttt.executeMove(state,secondPlayer,ttt.value_func,method2)
        if(ttt.winX(state)!=0):
            return -1
        if(np.count_nonzero(state)==9):
            return 0
def run():
    summe=0
    for i in range(1000):
        if(gameFight('TD(0)','random')==-1):
            summe+=1
    return summe  
print(run())