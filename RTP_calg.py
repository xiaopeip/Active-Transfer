#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import numba as nb
import random
from concurrent.futures import ProcessPoolExecutor

from scipy import integrate

import h5py




# In[3]:



cpus=48

L = 10.  # Box size


active_force = 3.  # strength of active force
tumble_rate = 4.5  # tumble rate (0 = never)
int_force=2.4 #strength of interaction
medium_friction=1.
N_colloids =  100000  # number of particles



dt = 0.005  # time step size

N_steps = 700_00  # number of time steps to integrate
N_eq = 200_00  # number of time steps to let the system equilibrate

count=N_eq//10 # algorithm records the time correlation function up to this time step difference
save_rate = 2  # save rate: the algorithm saves every 'save_rate' steps to save memory
count_index=count//save_rate

radius =0.5 # interaction range




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
def direction(m,p):
    if (m-p) % L < L/2:
        return -int_force
    else:
        return +int_force

@nb.jit(nopython=True)
def force(colloid_x,probe_x):
    dis=mod2(colloid_x-probe_x)
    int_m = np.where(np.abs(dis)<radius,int_force*np.sin(dis/radius*np.pi),0.)
    return int_m 

@nb.jit(nopython=True)
def sweep(N, colloid_x, colloid_s,drift):
    '''
    Performs N time steps, and pre-calculates noise and tumbles.
    Initial values of colloid_x and colloid_s should be given.
    meanfx: mean interaction force under current parameters.
    drift: external drift force, resulting from the change of frame to the probe frame.
    '''

    sum_int=0.

    tumble_times = np.random.exponential(scale=1/tumble_rate,size=N_colloids)

    for i in range(N):
        # second order Ruge-Kutta method

        int_m = force(colloid_x, 0.)
        if_tumbles = tumble_times < dt 

        Delta_x1=((int_m + active_force * colloid_s-drift ) * dt+if_tumbles*(-2*dt+2*tumble_times)*colloid_s*active_force)/medium_friction
        colloid_x2 = mod2(colloid_x + Delta_x1)
        colloid_s[if_tumbles] = -colloid_s[if_tumbles]

        tumble_times=tumble_times-dt
        #for no flip particles, tumble times decrease by dt, for particles with flip, tumble_times is negative
        tumble_times[if_tumbles]=np.random.exponential(scale=1/tumble_rate,size=np.sum(if_tumbles))+tumble_times[if_tumbles]
        #for flipped particles, generate new tumble times

        int_m2=force(colloid_x2,0.)

        Delta_x2 =  Delta_x1+(int_m2-int_m)*dt/medium_friction
        colloid_x = mod2(colloid_x+(Delta_x1+Delta_x2)/2)
        #second order Ruge-Kutta method

        sum_int+= np.mean(int_m)/(N)

    return colloid_x, colloid_s,sum_int


@nb.jit(nopython=True)
def sweep_gv(N, colloid_x,  colloid_s,meanfx,drift):
    '''
    Performs N time steps, record correlation function that needed to calculate G(v).
    '''

    
    seq_int=np.empty((count_index,N_colloids)) 
    # queue to store interaction forces at different time steps, up to last "count_index*save_rate" time steps

    time_corr=np.zeros(count_index)


    rear = 0   # 队尾指针
    isfull=False

    tumble_times = np.random.exponential(scale=1/tumble_rate,size=N_colloids)

    for i in range(N):
        int_m = force(colloid_x,0.)
        if_tumbles = tumble_times < dt 

        Delta_x1=((int_m + active_force * colloid_s-drift ) * dt+if_tumbles*(-2*dt+2*tumble_times)*colloid_s*active_force)/medium_friction
        colloid_x2 = mod2(colloid_x + Delta_x1)

        colloid_s[if_tumbles] = -colloid_s[if_tumbles]

        tumble_times=tumble_times-dt
        #for no flip particles, tumble times decrease by dt, for particles with flip, tumble_times is negative
        tumble_times[if_tumbles]=np.random.exponential(scale=1/tumble_rate,size=np.sum(if_tumbles))+tumble_times[if_tumbles]
        #for flipped particles, generate new tumble times

        int_m2=force(colloid_x2,0.)

        Delta_x2 =  Delta_x1+(int_m2-int_m)*dt/medium_friction
        colloid_x = mod2(colloid_x+(Delta_x1+Delta_x2)/2)


        if i%save_rate==0:
            saveint=int_m

            seq_int[rear]=saveint

            if rear==count_index-1:
                isfull=True
                for j in range(count_index):
                    time_corr[j]=np.mean((seq_int[j,:]-meanfx)*seq_int[0,:])
                break
            rear = (rear+1)%count_index
        isfull=False

    print(isfull)
    return  colloid_x, colloid_s, time_corr


@nb.jit(nopython=True)
def steady_dis(s):
    if tumble_rate>0 and active_force>1:
        v=active_force
        a=tumble_rate
        k=int_force
        x=random.random()
        if x<0.5-k*s/(v*2):
            return np.log((v-k*s)/(2*v*x-(-v+2*v*x+k*s)*np.exp(-2*np.pi*k*a/(v**2-k**2))))*(-1)*(v**2-k**2)/(2*k*a)
        else:
            return np.log((v+k*s)/(2*v*(1-x)+(-v+2*v*x+k*s)*np.exp(-2*np.pi*k*a/(v**2-k**2))))*(v**2-k**2)/(2*k*a)
    else:
        return random.random()*L-L/2
        


# In[4]:

# load previous results
with h5py.File("RTP_landscape_{}_{}.h5".format(int_force,tumble_rate)) as ipt:
    tab_drift2 = ipt["v"][...]
    Bv = ipt["noise"][()]
    fv = ipt["friction"][()]



# In[5]:


tab_drift = tab_drift2
dv=tab_drift[1]-tab_drift[0]




def calculate():
    '''
    Calculates G(v) for specific velocity v
    '''

    Gresult= np.empty(len(tab_drift))
    colloid_s = np.random.choice([-1,+1],N_colloids)
    colloid_x = np.random.random(N_colloids)*L-L/2


    colloid_x, colloid_s,useless=sweep(count*5, colloid_x, colloid_s,0.)
    colloid_x, colloid_s,useless=sweep(count, colloid_x, colloid_s,0.-dv)

    for i in range(len(tab_drift)):

        # calculate correlator with dynamics at v but with initial steady distribution at v-dv
        colloid_x, colloid_s,cor_left=sweep_gv(N_steps, colloid_x, colloid_s,fv[i],tab_drift[i])

        # equilibrate the system at v+dv
        colloid_x, colloid_s,useless=sweep(count, colloid_x, colloid_s,tab_drift[i]+dv)

        # calculate correlator with dynamics at v but with initial steady distribution at v+dv
        colloid_x, colloid_s,cor_right=sweep_gv(N_steps, colloid_x, colloid_s,fv[i],tab_drift[i])

        Gs=(cor_right-cor_left)/(2*dv)
        
        Gresult[i]=integrate.simpson(Gs, dx=dt*save_rate)

    return Gresult


Gv=np.zeros(len(tab_drift))


with ProcessPoolExecutor(max_workers=cpus) as executor:
    # 向每个CPU分别提交任务
    futures = [executor.submit(calculate) for mp in range(cpus)]
    # 收集结果
    for future in futures:
        Gv += future.result()/len(futures)


# In[6]:


with h5py.File("RTP_landscape_all_{}_{}para.h5".format(int_force,tumble_rate),"w") as opt:
    opt["L"]=L
    opt["alpha"]=tumble_rate
    opt["interaction"]=int_force
    opt["radius"]=radius
    opt["v0"]=active_force
    opt["v"]=tab_drift
    opt["noise"]=Bv
    opt["friction"]=fv
    opt["Gv"]=Gv






