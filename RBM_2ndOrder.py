import numpy as np


#RBM - bi partite
def RBM_2( hid,vis):
    hidden = len(hid)    #number of hidden units
    visible = len(vis)  #number of visible

    #initialize weights and biases
    weight =  np.random.rand(visible,hidden)
    bias_v = np.random.rand(visible,1)
    bias_h = np.random.rand(visible,1)

    # define Energy

    energy = - np.dot(bias_v,vis)-np.dot(bias_h,hid)- np.dot(vis,np.dot(weight,hid))
