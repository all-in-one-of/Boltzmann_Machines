import numpy as np

#define Simulated Annealing
def sim_an(x0, T0, energy):
    #define starting parameters x0 and T0
    x = x0
    T = T0
    tau = 1.2
    i=0
    maxIt = 1000

    # Annealing loop
    while(i<maxIt):
        # state update loop
        j = 0
        M=1000
        while(j<M):
            # choose new state randomly
            y = rand_choice(x)
            delta_energy  = energy(y)-energy(x)
            if(delta_energy<0):
                x=y
                j +=1
            else:
                # choose x=y with certain probability
                if(1/(1+np.exp(delta_energy/T))>np.random.rand(1,1)):
                    x=y
                    j+=1
                else: j+=1
        i +=1
        T = (1/tau)*T
    return x

# define how to randomly choose new state
def rand_choice(x):
    m = np.random.randint(len(x),1)
    x[m] = np.remainder(x[m]+1,2)
    return m

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
