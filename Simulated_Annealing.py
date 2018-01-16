import numpy as np
import copy
import tensorflow as tf
#define Simulated Annealing

def sim_an(X,h0,rbm_vb,rbm_hb,rbm_w,rbm_w3,clamped,T0, energy):
    #define starting parameters x0 and T0
    x1 = X
    x2 = h0

    T = T0
    tau = 1.2
    i=0
    maxIt = 10
    y1=copy.copy(X)
    y2 = copy.copy(x2)

    # Annealing loop
    while(i<maxIt):
        # state update loop
        j = 0
        M=10
        while(j<M):
            # choose new state randomly
            y2 = rand_choice(y2)
            if (not clamped):
                y1 = rand_choice(y1)
            delta_energy  = energy(y1,y2,rbm_vb,rbm_hb,rbm_w,rbm_w3)-energy(x1,x2,rbm_vb,rbm_hb,rbm_w,rbm_w3)
            if(all (delta_energy<0)):
                x2=copy.copy(y2)
                if(not clamped):
                    x1=copy.copy(y1)
                j +=1
            else:
                # choose x=y with certain probability
                if((np.exp(-delta_energy/T))>np.random.rand(1,1)):
                    x2 = copy.copy(y2)
                    if (not clamped):
                        x1 = copy.copy(y1)
                    j+=1
                else:
                    j+=1
        i +=1
        T = (1/tau)*T
    print(energy(x1,x2,rbm_vb,rbm_hb,rbm_w,rbm_w3))
    return x1,x2

# define how to randomly choose new state
def rand_choice(input):
    output = (tf.reshape(input,[-1]))
    m = np.random.randint(tf.to_int32(tf.shape(output)[0]))
    output[m] = np.remainder(output[m]+1,2)
    return tf.reshape(output,tf.shape(input))
