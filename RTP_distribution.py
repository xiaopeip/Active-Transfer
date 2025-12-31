#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import numba as nb
import random
import matplotlib.pyplot as plt
#from scipy.stats import gaussian_kde

import h5py



# In[57]:


L = 10.  # Box size





active_force = 3.  # strength of active force
tumble_rate = 4.5  # tumble rate (0 = never)
int_force=2.4 #strength of interaction
medium_friction=1.
N_colloids =  5  # number of particles


dt = 0.005  # time step size


mass=15. # mass of the probe
radius =0.5

N_steps = 5000_000_000  # number of time steps to integrate
N_eq = 2000_000  # number of time steps to let the system equilibrate
save_rate = 4000  # save rate: the algorithm saves probe velocity every 'save_rate' steps
peak=0.7 # location of velocity peaks


@nb.jit(nopython=True)
def force(colloid_x,probe_x):
    dis=mod2(colloid_x-probe_x)
    int_m = np.where(np.abs(dis)<radius,int_force*np.sin(dis/radius*np.pi),0.)
    return int_m 

@nb.jit(nopython=True)
def mod(x ):
    y=x%L
    if y >L/2:
        y=y-L
    return y


@nb.jit(nopython=True)
def mod2(x):
    # 使用 numpy.mod 将 x 限制在 [0, 2) 区间
    result = np.mod(x, L)
    return np.where(result > L/2, result - L, result)



@nb.jit(nopython=True)
def sweep_path(N, colloid_x, colloid_s,probe_x,probe_v):
    '''
    Simulate the composite dynamics, Record the probe velocity path, Count the transition between velocity peaks.
    Performs N time steps, initial values should be given.
    '''
    tumble_times = np.random.exponential(scale=1/tumble_rate,size=N_colloids)

    transition_time=[]
    previous=0 # "previous = 1,-1" means probe velocity in positive/negative peak area, 0 for initial
    path_v=np.empty(N//save_rate)

    for i in range(N):
        if i%save_rate==0:
            path_v[i//save_rate]=probe_v


        int_m = force(colloid_x, probe_x)
        if_tumbles = tumble_times < dt 

        Delta_x1=((int_m + active_force * colloid_s ) * dt+if_tumbles*(-2*dt+2*tumble_times)*colloid_s*active_force)/medium_friction
        d_probex1 = probe_v *dt
        d_probev1 = -np.sum(int_m)*dt/mass
        


        colloid_s[if_tumbles] = -colloid_s[if_tumbles]

        tumble_times=tumble_times-dt
        #for no flip particles, tumble times decrease by dt, for particles with flip, new tumble_times is negative
        tumble_times[if_tumbles]=np.random.exponential(scale=1/tumble_rate,size=np.sum(if_tumbles))+tumble_times[if_tumbles]
        #for flipped particles, generate new tumble times

        int_m2=force(colloid_x+Delta_x1,probe_x+d_probex1)

        Delta_x2 =  Delta_x1+(int_m2-int_m)*dt/medium_friction
        d_probex2 = (probe_v+d_probev1)*dt
        d_probev2 = -np.sum(int_m2)*dt/mass

        colloid_x = mod2(colloid_x+(Delta_x1+Delta_x2)/2)
        probe_x = mod(probe_x+(d_probex1+d_probex2)/2)
        probe_v = mod(probe_v+(d_probev1+d_probev2)/2)
        #second order Ruge-Kutta method

        if probe_v<-peak:
            if previous!=-1:
                previous=-1
                transition_time.append(i*dt)
                # record transition time from positive to negative peak
        if probe_v>peak:
            if previous!=1:
                previous=1
                transition_time.append(i*dt)
                # record transition time from positive to negative peak

        
        if i%save_rate==0:
            path_v[i//save_rate]=probe_v
            if i%3000000==0:
                print(i/N_steps)



    return colloid_x, colloid_s,probe_x, probe_v, transition_time, path_v


        
    


# In[58]:


colloid_s = np.random.choice([-1,+1],N_colloids)
#colloid_x = np.array([steady_dis(s) for s in colloid_s])
colloid_x = np.random.random(N_colloids)*L-L/2

probe_x = 0.
probe_v = 0.
colloid_x, colloid_s,probe_x, probe_v=sweep_path(N_eq, colloid_x, colloid_s, probe_x, probe_v)[:4]



colloid_x, colloid_s, probe_x, probe_v, transitions,path_v=sweep_path(N_steps, colloid_x, colloid_s, probe_x, probe_v)


# time intervals between transitions
waiting_time=np.empty(len(transitions)-1)
for j in range(len(transitions)-1):
    waiting_time[j]=transitions[j+1]-transitions[j]
    




len(waiting_time)


# In[61]:


with h5py.File("RTP_path_{}_{}_{}.h5".format(int_force,tumble_rate,mass), "w") as opt:
    opt["L"]=L
    opt["alpha"]=tumble_rate
    opt["mass"]=mass
    opt["N_medium"]=N_colloids
    opt["interaction"]=int_force
    opt["radius"]=radius
    opt["dt"]=dt
    opt["time_step"]=save_rate*dt
    opt["v0"]=active_force
    opt["passage_time"]=waiting_time
    opt["path_v"]=path_v



