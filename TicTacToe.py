#!/usr/bin/env py+thon3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:36:31 2018

@author: oster
"""
import numpy as np
import copy as co
import TemporalDifference as TD
import predictor


# state is a 3x3 Matrix with 1=x, 2 = o and 0 empty
# value funktion is a list of length 3^9
value=.5*np.ones((19683))
#learning parameter
alpha=0.5
#discount parameter
gamma=1
#epsilon for greedy policy
eps=0.05
#iteration for training
MaxIt=10


#define winning
# return 1 if x wins and -1 if o wins

#subroutine for checking horizontal wins
def winX_not_diagonal(state):
    
    for i in range(3):
        win = np.mean(state[i,:])
        if np.abs(win)==1:
            return np.sign(win)
    return 0
#subroutine for checking main diagonal wins
def winX_diagonal(state):
    if(state[0,0] != 0):
        if(state[0,0]==state[1,1] and state[0,0]==state[2,2]):
            return state[0,0]
    return 0
#checking total wins
def winX(state):
    dummy1 = winX_not_diagonal(state)
    dummy2 = winX_not_diagonal(state.T)     #checking for vertical wins
    dummy3 = winX_diagonal(state)
    dummy4 = winX_diagonal(np.fliplr(state)) #checking for minor diagonal wins

    if dummy1 !=0:
        return dummy1
    elif dummy2 !=0:
        return dummy2
    
    elif dummy3 !=0:
        return dummy3
    elif dummy4 !=0:
        return dummy4

    return 0

#making a move
#define an action to be an index pair with indicator 1 or 2
def move(state, action):
    if state[action[0],action[1]] ==0:
        state[action[0],action[1]] = state[action[0],action[1]] +action[2]
        if winX(state)!=0:
            value[stateID(state)]=winX(state)
        return state,True
    else:
        return [],False
#generates possible moves for player
def moveGenerator(current_state, player):
    moves=[]
    for i in range(3):
        for j in range(3):
            if (move(co.copy(current_state),[i,j,player])[1]):
                moves.append(move(co.copy(current_state),[i,j,player])[0])
    return moves
# execution of a eps-greedy move
def executeMove(state,player,value_function):
    return learn(state, value_function,player)
    

# evaluating a state: interprete state as ternary representation
def stateID(state):
    newstate= (np.mod(co.copy(state)+3,3))
    return int(''.join(str(e) for e in np.array(np.reshape(newstate,(1,9)))[0]),3)
def value_func(state):
    return value[stateID(state)]

# learning via temporal differenct (comparwe "Reinforcment learning: An Introduction" by Sutton and Barto pp. 7)
def learn(state, value_function,player):
    state_1=predictor.eps_greed(value_function, moveGenerator(state,player),eps,player)
    value[stateID(state)]=TD.TD_0_update(value_function,state,state_1,0,alpha,gamma)
    return state_1
#define one game of tic tac toe
def game():
    p=np.random.randint(0,2)
    startPlayer = (-1)**p
    secondPlayer= -1*startPlayer
    incomplete = True
    state= np.matrix([[0,0,0],[0,0,0],[0,0,0]])
    while(incomplete):
        if(np.count_nonzero(state)==9):
            break
        
        state=executeMove(state,startPlayer,value_func)

        if(winX(state)!=0):
            break
        
        if(np.count_nonzero(state)==9):
            break
        state=executeMove(state,secondPlayer,value_func)

        if(winX(state)!=0):
            break
        if(np.count_nonzero(state)==9):
            break
#define a training session
def training():
    for i in range(MaxIt):
        game()
     


