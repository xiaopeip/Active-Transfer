# coding: utf-8

# In[3]:


import numpy as np
import numba as nb
from concurrent.futures import ProcessPoolExecutor

from scipy import integrate

import h5py


# In[4]:


L = 10.  # Box size


cpus= 48 # number of CPUs to use

active_force = 3.  # strength of active force
tumble_rate = 4.5  # diffusion of the orientation
int_force= 2.4 # strength of interaction
medium_mobility=1. # mobility of the medium particles
N_colloids = 50000  # number of medium particles

N_task = N_colloids*cpus # number of particles per CPU

dt = 0.003  # time step size

#driving = 0.5  # driving strength
radius=.5 # range of the interaction

save_rate=2 # save rate for calculating the correlation function


N_steps = 500_000  # number of time steps to integrate
N_eq = 25_00  # number of time steps to let the system equilibrate

count=N_eq # number of time steps when calculating the integrated correlation function
count_index=count//save_rate 


# In[5]:


@nb.jit(nopython=True)
def mod2(x):
    # 使用 numpy.mod 将某个数组中每个元素限制在 [-L/2, L/2) 区间
    result = np.mod(x, L)
    return np.where(result > L/2, result - L, result)

@nb.jit(nopython=True)
def force(colloid_x,colloid_y):
    #Lennard-jones
    distance=np.sqrt(colloid_x**2+colloid_y**2) 

    return 169/42*((13/7)**(1/6))*((distance/radius)**(-13)-(distance/radius)**(-7))*int_force*colloid_x/distance, 169/42*((13/7)**(1/6))*((distance/radius)**(-13)-(distance/radius)**(-7))*int_force*colloid_y/distance


@nb.jit(nopython=True)
def sweep_mean(N, colloid_x, colloid_y, colloid_s,driving):
    '''
    Simulate the medium dynamics in the frame moving with the probe.
    Performs N time steps, in order to let system equilibrate.
    '''
    
    # m_intx=0.
    # m_inty=0.

    for i in range(N):


        int_x, int_y = force(colloid_x,colloid_y)
        d_colloid_x= (active_force*np.cos(colloid_s)+ int_x -driving) * dt*medium_mobility
        d_colloid_y = (active_force*np.sin(colloid_s)+ int_y ) * dt*medium_mobility

        colloid_s += np.random.normal(0,np.sqrt(2*tumble_rate*dt),N_colloids) 
        # update the orientation of the colloid

        int_x2,int_y2= force(mod2(colloid_x+d_colloid_x),mod2(colloid_y+d_colloid_y))
        d_colloid_x2= (active_force*np.cos(colloid_s)+ int_x2 -driving) * dt*medium_mobility
        d_colloid_y2 = (active_force*np.sin(colloid_s)+ int_y2 ) * dt*medium_mobility

        colloid_x = mod2(colloid_x + (d_colloid_x+d_colloid_x2)/2)
        colloid_y = mod2(colloid_y + (d_colloid_y+d_colloid_y2)/2)
        # update the position using second-order Runge-Kutta method

        # m_intx+=np.mean(int_x)
        # m_inty+=np.mean(int_y)

    return  colloid_x,colloid_y, colloid_s


@nb.jit(nopython=True)
def sweep_G(N, colloid_x, colloid_y, colloid_s,meanfx,driving):
    '''
    Simulate the medium dynamics in the frame moving with the probe.
    Performs N time steps, and calculates the time correlation function.
    driving: drift force, resulting from the moving frame.
    meanfx: mean interaction force under current parameters.
    '''
    
    seq_int=np.empty((count_index,N_colloids,2),dtype=np.float64)
    # circular queue to store interaction forces at different time steps, up to last "count_index*save_rate" time steps


    time_corr=np.zeros(count_index,dtype=np.float64)
    # array to save time correlation function 

    meanfy=0.

    rear = 0   # 队尾指针
    isfull=False

    for i in range(N):
        int_x, int_y = force(colloid_x,colloid_y)
        d_colloid_x= (active_force*np.cos(colloid_s)+ int_x -driving) * dt*medium_mobility
        d_colloid_y = (active_force*np.sin(colloid_s)+ int_y ) * dt*medium_mobility

        colloid_s += np.random.normal(0,np.sqrt(2*tumble_rate*dt),N_colloids) 

        int_x2,int_y2= force(mod2(colloid_x+d_colloid_x),mod2(colloid_y+d_colloid_y))
        d_colloid_x2= (active_force*np.cos(colloid_s)+ int_x2 -driving) * dt*medium_mobility
        d_colloid_y2 = (active_force*np.sin(colloid_s)+ int_y2 ) * dt*medium_mobility

        colloid_x = mod2(colloid_x + (d_colloid_x+d_colloid_x2)/2)
        colloid_y = mod2(colloid_y + (d_colloid_y+d_colloid_y2)/2)
        # update according to the second-order Runge-Kutta method


        if i%save_rate==0:
            # save data every save_rate steps
            seq_int[rear,:,0]=int_x
            seq_int[rear,:,1]=int_y

            if rear==count_index-1:
                # if the queue is full, calculate the time correlation function and break the loop
                isfull=True
                for j in range(count_index):
                    time_corr[j]=np.mean((seq_int[j,:,0]-meanfx)*seq_int[0,:,0])
                break
            rear = (rear+1)%count_index
        isfull=False
        


    return  colloid_x,colloid_y, colloid_s, time_corr



# In[6]:

# Load the data from the h5 file
with h5py.File("ABM_landscape_{}_{}.h5".format(int_force,tumble_rate)) as ipt:
    tab_drift2 = ipt["v"][...]
    Bx = ipt["Bx"][()]
    By = ipt["By"][()]
    fy = ipt["fy"][()]
    fx = ipt["fx"][()]


tab_drift =tab_drift2
dv=tab_drift[1]-tab_drift[0]

# In[ ]:
def calculate():
    '''
    Calculates G(v) for specific velocity v
    '''

    colloid_s = np.random.random(N_colloids)*2*np.pi
    colloid_x = mod2(np.random.random(N_colloids)*(L-2*radius)+radius)
    colloid_y = mod2(np.random.random(N_colloids)*(L-2*radius)+radius)
    # randomly initialize the position and orientation of the colloid

    colloid_x, colloid_y, colloid_s =sweep_mean(N_eq*6, colloid_x, colloid_y, colloid_s,0.)[:3]
    colloid_x, colloid_y, colloid_s =sweep_mean(N_eq, colloid_x, colloid_y, colloid_s,0.-dv)[:3]

    # equilibrate the system

    G_result = np.zeros(len(tab_drift),dtype=np.float64)
    for i in range(len(tab_drift)):

        print(i/len(tab_drift))

        # calculate correlator with dynamics at v but with initial steady distribution at v-dv
        colloid_x, colloid_y, colloid_s,Gleft=sweep_G(N_steps, colloid_x, colloid_y, colloid_s,fx[i],tab_drift[i])
        
        # equilibrate the system at v+dv
        colloid_x, colloid_y, colloid_s =sweep_mean(N_eq, colloid_x, colloid_y, colloid_s,tab_drift[i]+dv)[:3]

        # calculate correlator with dynamics at v but with initial steady distribution at v+dv
        colloid_x, colloid_y, colloid_s,Gright=sweep_G(N_steps, colloid_x, colloid_y, colloid_s,fx[i],tab_drift[i])



        Gs=(Gright-Gleft)/(2*dv)
        # differential of the time correlation function
        
        G_result[i]=integrate.simpson(Gs, dx=dt*save_rate)
        # integral over time

    return G_result


# In[43]:
Gv=np.zeros(len(tab_drift))


with ProcessPoolExecutor(max_workers=cpus) as executor:
    # 向每个CPU分别提交任务
    futures = [executor.submit(calculate) for mp in range(cpus)]
    # 收集结果
    for future in futures:
        Gv += future.result()/len(futures)



# In[8]: save results to file


with h5py.File("ABM_landscape_all_{}_{}.h5".format(int_force,tumble_rate),"w") as opt:
    opt["L"]=L
    opt["alpha"]=tumble_rate
    opt["interaction_max"]=int_force
    opt["radius_lj0"]=radius
    opt["v0"]=active_force
    opt["v"]=tab_drift
    opt["fx"]=fx
    opt["fy"]=fy
    opt["Bx"]=Bx
    opt["By"]=By
    opt["G"]=Gv
    opt["dt"]=dt


# In[ ]:




