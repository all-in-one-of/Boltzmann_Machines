import numpy as np
import copy
#define Simulated Annealing
def sim_an(x0, T0, energy):
    #define starting parameters x0 and T0
    x = x0
    T = T0
    tau = 1.2
    i=0
    maxIt = 1000
    y=copy.copy(x0)

    # Annealing loop
    while(i<maxIt):
        # state update loop
        j = 0
        M=1000
        while(j<M):
            # choose new state randomly
            rand_choice(y)
            delta_energy  = energy(y)-energy(x)
            if(delta_energy<0):
                x=copy.copy(y)
                j +=1
                print(energy(x))
            else:
                # choose x=y with certain probability
                if((np.exp(-delta_energy/T))>np.random.rand(1,1)):
                    x=copy.copy(y)
                    j+=1
                    print(energy(x))
                else:
                    j+=1
                    print(energy(x))
        i +=1
        T = (1/tau)*T


    return x

# define how to randomly choose new state
def rand_choice(input):
    output = input
    m = np.random.randint(len(output))
    output[m] = np.remainder(output[m]+1,2)
    return output

def energy(x):
    en = 0
    for i in range(len(x)):
        en+=+x[i]**2+x[i]**3+12*x[i]**4
    return en

x0 = [1,1,0,1,0,0,1,0,0,1,0,1,0,1,1,1,1,1,1,0,1,1,0,0,0,1]
t0 = 10000
print(energy(sim_an(x0,t0,energy)))