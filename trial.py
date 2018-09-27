#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 09:07:16 2018

@author: oster
"""

import numpy as np

A=np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(A)
x=np.array([1,1])
x.shape = (2,1)
print(x)
B=np.tensordot(np.tensordot(np.tensordot(A,x,1),x,2),x,1)
print(B)
C=np.array([[1,2],[3,4]])
print(C)
print(C[:,0])
print(C[:,1])
weightsAlt = np.array([[[1,1],[1,1]],[[1,1],[1,1]]])
state = np.array([[1,1],[1,1]])
a1 = np.tensordot(weightsAlt,state[0,:],1)
print(a1)
a2 = np.tensordot(a1,state[1,:],1)
print(a2)
a3=np.tensordot(a2,np.reshape(state[:,0],(2,1)),1)
print(a3)
#a4 = np.tensordot(a3,state[:,0],1)
#a5 = np.tensordot(a4,state[:,1],1)
#dummy = np.tensordot(a5,state[:,2],1)

#C= np.tensordot(B,x,2)
#
#print(C)
#D=np.tensordot(C,x,1)
#print(D)