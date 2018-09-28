#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:43:51 2018

@author: oster
"""
import TicTacToe as ttt
import numpy as np
#comparing different methods
def gameFight(method1,method2, valfunc1,valfunc2):
    p=np.random.randint(0,2)
    startPlayer = (-1)**p
    secondPlayer= -1*startPlayer
    incomplete = True
    state= np.matrix([[0,0,0],[0,0,0],[0,0,0]])

    while(incomplete):
        if(np.count_nonzero(state)==9):
            return 0

        state=ttt.executeMove(state,startPlayer,valfunc1, method1,0.,0)
        if(ttt.winX(state)!=0):
            return ttt.winX(state)
        
        if(np.count_nonzero(state)==9):
            return 0
        state=ttt.executeMove(state,secondPlayer,valfunc2,method2,0.,0)
        if(ttt.winX(state)!=0):
            return ttt.winX(state)
        if(np.count_nonzero(state)==9):
            return 0
def run():
    summe=0
    ttt.readValue_n_stepTD()
    ttt.readValueTD0()  
    ttt.readWeights()
    summ0=0
    print('zero game')
    for i in range(1000):
        if(gameFight('random','random',ttt.value_funcTD0,ttt.value_funcTD0)==1):
            summ0+=1
    summ1=0    
    print('first game')
    for i in range(1000):
        if(gameFight('random','TD(0)',ttt.value_funcTD0,ttt.value_funcTD0)==1):
            summe+=1
    summ1=0
    print('second game')
    for i in range(1000):
        if(gameFight('random','n-step_TD',ttt.value_func_n_stepTD,ttt.value_func_n_stepTD)==1):
            summ1+=1    
    summ1b=0
    print('third game')
    for i in range(1000):
        if(gameFight('random','semi_gradient_TD0',ttt.value_func_semi_gradient_TD0,ttt.value_func_semi_gradient_TD0)==1):
            summ1b+=1   
    summe1c=0
    print('thirdA game')
    for i in range(1000):
        if(gameFight('random','tensor_alternativ',ttt.value_funcTD0,ttt.value_func_tensor_alternativ)==1):
            summe1c+=1
    summe2=0
    print('fourth game')
    for i in range(1000):
        if(gameFight('TD(0)','n-step_TD',ttt.value_funcTD0,ttt.value_func_n_stepTD)==1):
            summe2+=1
    summe2b=0
    print('fith game')
    for i in range(1000):
        if(gameFight('TD(0)','semi_gradient_TD0',ttt.value_funcTD0,ttt.value_func_semi_gradient_TD0)==1):
            summe2b+=1
    return summ0,summe ,summ1,summ1b, summe1c ,summe2, summe2b 

print(run())