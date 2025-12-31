#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import numba as nb
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
#from scipy.stats import gaussian_kde

import h5py




# In[57]:



cpus=48

L = 10.  # Box size



active_force = 3.  # strength of active force
tumble_rate = 1.6  # tumble rate (0 = never)
int_force=3.3 #strength of interaction
medium_friction=1.
N_colloids =  5  # number of particles


dt = 0.005  # time step size
loop_per_cpu=20 # number of simulation loops per cpu


mass=30.
radius =0.5

N_steps = 20000_000  # number of time steps to integrate
N_eq = 5000_000  # number of time steps to let the system equilibrate
save_rate = 100_000  # save rate: the algorithm saves every 'save_rate' steps to save memory
peak=0.5


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
def sweep_composite(N, colloid_x, colloid_s,probe_x,probe_v):
    '''
    Simulate the composite dynamics.
    Performs N time steps, and calculates the mean square displacement.
    colloid_x, colloid_s,probe_x,probe_v: Initial values.
    '''

    probe_x0=probe_x

    tumble_times = np.random.exponential(scale=1/tumble_rate,size=N_colloids)

    # transition_time=[]
    # previous=0
    msd = np.empty(N//save_rate)

    for i in range(N):



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
        probe_x = probe_x+(d_probex1+d_probex2)/2
        # no periodic boundary condition for position
        probe_v = mod(probe_v+(d_probev1+d_probev2)/2)
        #second order Ruge-Kutta method

        # if probe_v<-peak:
        #     if previous!=-1:
        #         previous=-1
        #         transition_time.append(i*dt)
        # if probe_v>peak:
        #     if previous!=1:
        #         previous=1
        #         transition_time.append(i*dt)
        
        if i%3000000==0:
            print(i/N_steps)

        if i%save_rate==0:

            msd[i//save_rate] = (probe_x-probe_x0)**2

    return colloid_x, colloid_s,probe_x, probe_v,msd

        
    


# In[58]:

def calculate():

    msd_percpu=np.empty((loop_per_cpu,N_steps//save_rate))
    for i in range(loop_per_cpu):

        colloid_s = np.random.choice([-1,+1],N_colloids)
        #colloid_x = np.array([steady_dis(s) for s in colloid_s])
        colloid_x = np.random.random(N_colloids)*L-L/2
        #print(colloid_s)

        probe_x = 0.
        probe_v = 0.
        colloid_x, colloid_s,probe_x, probe_v=sweep_composite(N_eq, colloid_x, colloid_s, probe_x, probe_v)[:4]

        msd_percpu[i] =sweep_composite(N_steps, colloid_x, colloid_s, probe_x, probe_v)[-1]

    return msd_percpu




# In[59]:


results = [None] * cpus

with ProcessPoolExecutor(max_workers=cpus) as executor:
    # 提交所有任务
    futures = [executor.submit(calculate) for k in range(cpus)]
    # 收集结果
    for i, future in enumerate(futures):
        results[i] = future.result()



msd=np.array([result for result in results])
save_time=np.arange(0,N_steps*dt,save_rate*dt)

diffusion=msd[:,:,-1]/(2*save_time[-1])

# In[61]:


with h5py.File("RTP_msd_{}_{}_{}.h5".format(int_force,tumble_rate,mass), "w") as opt:
    opt["L"]=L
    opt["alpha"]=tumble_rate
    opt["mass"]=mass
    opt["N_medium"]=N_colloids
    opt["interaction"]=int_force
    opt["radius"]=radius
    opt["dt"]=dt
    opt["time_step"]=save_rate*dt
    opt["v0"]=active_force
    opt["msd"]=msd
    opt["diffusion"]=diffusion
    opt["time"]=save_time
    opt["N_steps"]=N_steps



