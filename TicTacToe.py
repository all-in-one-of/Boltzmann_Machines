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
value_n_step=.5*np.ones((19683))

#learning parameter
#alpha=0.02
#discount parameter
gamma=1
#epsilon for greedy policy
#eps=0.1
#iteration for training
MaxIt=1000
#weights for semi gradient TD(0)

weights = 1*np.ones((9,9,9))
weightsAlt = np.ones((3,3,3,3,3,3))

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
#method: from list [random, TD(0), n-step_TD , semi_gradient_TD0] pf strings
def executeMove(state,player,value_function, method,alpha,eps):
    if method == "TD(0)":
        return learn_TD0(state, value_function,player,alpha,eps)
    if method == 'n-step_TD':
        return learn_n_step_TD(state,value_function, player,3,alpha,eps)
    if method == 'random':
        return predictor.randomAction(moveGenerator(state,player))
    if method =='semi_gradient_TD0':
        return learn_semi_gradient_TD0(value_function, state,player,weights,alpha,eps)
    if method == 'tensor_alternativ':
        return learn_tensor_alternativ(value_function, state,player,weightsAlt,alpha,eps)

#build a list of steps and rewards for n-step TD    
def builtChain_n_step_TD(state,player,value_function,n,eps):
    notwinning=True
    if winX(state)!=0:
        notwinning = False
        return [],[]
    i=0
    chain = []
    reward = []
    while(notwinning and i<n and np.count_nonzero(state)!=9):    
        dummy=predictor.eps_greed(value_function,moveGenerator(state,player),eps,player)
        chain.append(dummy)
        reward.append(winX(dummy))
        if(winX(dummy)!=0):
            notwinning = False
            break
        state = dummy
        if(np.count_nonzero(state)==9):
            break
        dummy=predictor.eps_greed(value_function,moveGenerator(state,-1*player),eps,-1*player)
        chain.append(dummy)
        reward.append(winX(dummy))
        if(winX(dummy)!=0):
            notwinning = False
            break
        state = dummy
        if(np.count_nonzero(state)==9):
            break        
        i+=1
    return chain,reward
 

       
# evaluating a state: interprete state as ternary representation
def stateID(state):
    newstate= (np.mod(co.copy(state)+3,3))
    return int(''.join(str(e) for e in np.array(np.reshape(newstate,(1,9)))[0]),3)
def value_funcTD0(state):
    return value[stateID(state)]
def value_func_n_stepTD(state):
    return value_n_step[stateID(state)]
# uses a multilinear form T on the states, i.e an order 3 tensor T(s,s,s)
def value_func_semi_gradient_TD0(state):
    
    x=np.array(np.reshape(state,[9,1]))
    ret = np.tensordot(np.tensordot(np.tensordot(weights,x,1),x,2),x,1)
    return ret
# as value_func_semi_gradientTD0 but with rows and columns as input, i.e. an order 6 tensor T(r1,r2,r3,c1,c2,c3)
def value_func_tensor_alternativ(state):
    a1=np.tensordot(weightsAlt,np.reshape(state[0,:],(3,1)),1)
    a2=np.tensordot(a1,state[1,:],(1,1))
    a3=np.tensordot(a2,np.reshape(state[2,:],(1,3)),(2,1))
    a3.shape=(3,3,3)
    a4=np.tensordot(a3,np.reshape(state[:,0],(3,1)),1)
    a5=np.tensordot(a4,np.reshape(state[:,1],(1,3)),(1,1))
    a5.shape=1,3
    dummy = np.inner(a5,np.reshape(state[:,2],(1,3)))[0]
    return np.asscalar(dummy[0])



# learning via temporal differenct (comparwe "Reinforcment learning: An Introduction" by Sutton and Barto pp. 7)
def learn_TD0(state, value_function,player,alpha,eps):
    state_1=predictor.eps_greed(value_function, moveGenerator(state,player),eps,player)
    value[stateID(state)]=TD.TD_0_update(value_function,state,state_1,winX(state),alpha,gamma)
    return state_1
#learn with n-step TD
def learn_n_step_TD(state, value_function,player,n,alpha,eps):
    state_follow, reward = builtChain_n_step_TD(state,player, value_function, n,eps)
    value_n_step[stateID(state)]=TD.n_step_TD(value_function, state, state_follow, reward , alpha, gamma )
    return predictor.eps_greed(value_function, moveGenerator(state,player),eps,player)
def learn_semi_gradient_TD0(value_function, state, player,w,alpha,eps):
    state_1=predictor.eps_greed(value_function, moveGenerator(state,player),eps,player)
    for i in range(len(w)):
        weights[i]=TD.semi_gradient_TD_0(value_function,state,state_1,winX(state),alpha,gamma,w)[i]
    return state_1
def learn_tensor_alternativ(value_function, state, player,w,alpha,eps):
    state_1 = predictor.eps_greed(value_function, moveGenerator(state,player),eps,player)
    for i in range(len(w)):
        weightsAlt[i] =  TD.tensor_alternativ(value_function,state,state_1, winX(state),alpha,gamma,w)[i]
    return state_1

#define one game of tic tac toe
def game(method,value_function,alpha,eps):
    p=np.random.randint(0,2)
    startPlayer = (-1)**p
    secondPlayer= -1*startPlayer
    incomplete = True
    state= np.matrix([[0,0,0],[0,0,0],[0,0,0]])
    while(incomplete):
        if(np.count_nonzero(state)==9):
            break
        state=executeMove(state,startPlayer,value_function, method,alpha,eps)
        if(winX(state)!=0):
            break
        if(np.count_nonzero(state)==9):
            break
        state=executeMove(state,secondPlayer,value_function,method,alpha,eps)
        if(winX(state)!=0):
            break
        if(np.count_nonzero(state)==9):
            break
#define a training session
def training():
    readValue_n_stepTD()
    eps1 = .1
    eps2=.1
    eps3 = .1
    eps4 = .1
    decrease = .99
    for i in range(MaxIt):
        game('n-step_TD',value_func_n_stepTD,.1/(i+1)**(2/3),eps1)
        eps1*=decrease
    safeValue_n_stepTD()
    readValueTD0()
    print("Training n step finished")
    for i in range(MaxIt):
        game('TD(0)', value_funcTD0,.1/(i+1)**(2/3),eps2)
        eps2*=decrease
    safeValueTD0()  
    print('training TD(0) finished')
    readWeights()
    for i in range(MaxIt):
        game('semi_gradient_TD0',value_func_semi_gradient_TD0,.1/(i+1)**(2/3),eps3)
        eps3*=decrease
    safeWeights()
    print('Training semi gradient finished')
    readWeightsAlt()
    for i in range(MaxIt):
        game('tensor_alternativ',value_func_tensor_alternativ,.1/(i+1)**(2/3),eps4)
        eps4*=decrease
    safeWeightsAlt()
    print('Training t4nsor alternativ finsihed')
def safeValueTD0():
    file = open('valueFunctionTD0','w')
    for v in value:
        file.write(str(v)+'\n')
    file.close()
def readValueTD0():
    file = open('valueFunctionTD0','r')
    value2=[]
    for f in file:
        value2.append(f)
    value=np.array(value2)    
    file.close()
def safeValue_n_stepTD():
    file = open('valueFunction_n_step_TD','w')
    for v in value_n_step:
        file.write(str(v)+'\n')
    file.close()
def readValue_n_stepTD():
    file = open('valueFunction_n_step_TD','r')
    value2=[]
    for f in file:
        value2.append(f)
    value_n_step=np.array(value2)    
    file.close()  
def safeWeights():
    file = open('weights','w')
    file.write(str(list(np.reshape(weights,(9**3,)))))
    file.close()
def readWeights():
    file = open('weights','r')
    dummy = np.fromstring(file.read().strip('[').strip(']'),dtype=float,sep=',')
    dummy.shape=(9,9,9)
    for i in range(9):
        weights[i] =dummy[i]
def safeWeightsAlt():
    file = open('weightsAlt','w')
    file.write(str(list(np.reshape(weightsAlt,(3**6,)))))
    file.close()
def readWeightsAlt():
    file = open('weightsAlt','r')
    dummy = np.fromstring(file.read().strip('[').strip(']'),dtype=float,sep=',')
    dummy=np.reshape(dummy,(3,3,3,3,3,3))
    for i in range(3):
        weightsAlt[i]=dummy[i]
    file.close()
    
safeValueTD0()
safeValue_n_stepTD()
safeWeights()
safeWeightsAlt()
print('Training started')
training()
print('Training finished')
