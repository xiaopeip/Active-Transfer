# coding: utf-8

# In[40]:


import numpy as np
import numba as nb
import h5py
import sys

from scipy import integrate
from concurrent.futures import ProcessPoolExecutor



# In[41]:


L = 10  # Box size
active_force = 3.  # strength of active force
tumble_rate = 3.0  # diffusion of the orientation
int_force= 1.5 #strength of interaction
medium_mobility=1.
N_colloids =  50  # number of particles
#mass = 16.

radius=.5 # range of the interaction

dt = 0.003  # time step size
N_steps = 500_000  # number of time steps to integrate
N_eq = 100_000  # number of time steps to let the system equilibrate
save_rate=2 # save rate for calculating the correlation function

count=N_eq//50
count_index=count//save_rate

cpus= 48 # number of CPUs to use

if len(sys.argv)==3:
    int_force=float(sys.argv[1])
    tumble_rate=float(sys.argv[2])
    for i, arg in enumerate(sys.argv[1:], 1):
        print(f"  参数{i}: {arg}")
        print("landscape{}_LJ{}.h5".format(int_force,tumble_rate))
else:
    print("no input")


@nb.jit(nopython=True)
def mod2(x):
    # 使用 numpy.mod 将某个数组中每个元素限制在 [-L/2, L/2) 区间
    result = np.mod(x, L)
    return np.where(result > L/2, result - L, result)

@nb.jit(nopython=True)
def force(colloid_x,colloid_y):
    #Lennard-Jones
    distance=np.sqrt(colloid_x**2+colloid_y**2) 
    return 169/42*((13/7)**(1/6))*((distance/radius)**(-13)-(distance/radius)**(-7))*int_force*colloid_x/distance, 169/42*((13/7)**(1/6))*((distance/radius)**(-13)-(distance/radius)**(-7))*int_force*colloid_y/distance




@nb.jit(nopython=True)
def sweep_mean(N, colloid_x, colloid_y, colloid_s,driving):
    '''
    Simulate the medium dynamics in the frame moving with the probe.
    driving: drift force, resulting from the moving frame.
    Performs N time steps, and calculate friction from the mean value of force.
    '''
    
    m_intx=0.
    m_inty=0.

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
        # update according to second-order Runge-Kutta method

        m_intx+=np.mean(int_x)
        m_inty+=np.mean(int_y)
        # calculate the mean value of the force

    return  colloid_x,colloid_y, colloid_s, m_intx/(N),m_inty/(N)






@nb.jit(nopython=True)
def sweep_noise(N, colloid_x, colloid_y, colloid_s,meanfx,driving):
    '''
    Performs N time steps, and calculates noise intensity.
    '''
    
    seq_int=np.empty((count_index,N_colloids,2),dtype=np.float64)
    # 存储相互作用力的环形队列，只存储最近的count_index个值

    time_corrx=np.zeros(count_index,dtype=np.float64)
    time_corry=np.zeros(count_index,dtype=np.float64)
    # time_xy=np.zeros(count_index,dtype=np.float64)
    # time_yx=np.zeros(count_index,dtype=np.float64)

    meanfy=0. # mean value of the force in y-direction
    record_number=0 # 标记存储次数

    rear = 0   # 队尾指针
    isfull=False # 队列是否已满

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

        if i%save_rate==0:
            saveint=np.empty((N_colloids,2))
            saveint[:,0]=int_x
            saveint[:,1]=int_y

            seq_int[rear]=saveint

            if rear==count_index-1:
                isfull=True
            if isfull:
                # if the queue is full, calculate the time correlation function
                record_number+=1
                for j in range(count_index):

                    time_corrx[j]+=np.mean((seq_int[rear,:,0]-meanfx)*(seq_int[(rear-j)%count_index,:,0]-meanfx))
                    time_corry[j]+=np.mean((seq_int[rear,:,1]-meanfy)*(seq_int[(rear-j)%count_index,:,1]-meanfy))
                    #time_yx[j]+=np.mean((seq_int[rear,:,1]-meanfy)*(seq_int[(rear-j)%count_index,:,0]-meanfx))
                    #time_xy[j]+=np.mean((seq_int[rear,:,0]-meanfx)*(seq_int[(rear-j)%count_index,:,1]-meanfy))

            rear = (rear+1)%count_index # 更新队列指针位置


    # corx=np.sum(time_corrx)*dt*save_rate
    # cory=np.sum(time_corry)*dt*save_rate

    return  colloid_x,colloid_y, colloid_s, time_corrx/record_number, time_corry/record_number




        


# In[42]:




#@nb.njit
def calculate_mean():
    '''
    Calculate the mean value and noise intensity of the system, return the mean value and time correlation function.
    '''
    
    colloid_s = np.random.random(N_colloids)*2*np.pi
    colloid_x = mod2(np.random.random(N_colloids)*(L-2*radius)+radius)
    colloid_y = mod2(np.random.random(N_colloids)*(L-2*radius)+radius)
    colloid_x, colloid_y, colloid_s =sweep_mean(N_eq, colloid_x, colloid_y, colloid_s,0.)[:3]

    meanfx= np.empty(len(tab_drift))
    meanfy= np.empty(len(tab_drift))
    # time_corx= np.empty((len(tab_drift),count_index))
    # time_cory= np.empty((len(tab_drift),count_index))

    for i,drift in enumerate(tab_drift):
        #print("first:",drift/3)
        colloid_x, colloid_y, colloid_s =sweep_mean(N_eq, colloid_x, colloid_y, colloid_s,drift)[:3]
        colloid_x, colloid_y, colloid_s,meanfx[i],meanfy[i] =sweep_mean(N_steps*3, colloid_x, colloid_y, colloid_s,drift)
    return meanfx,meanfy

def calculate_variance(meanfx):
    '''
    Calculate the noise intensity of the system.
    meanfx: mean interaction force under current parameters.
    '''
    
    colloid_s = np.random.random(N_colloids)*2*np.pi
    colloid_x = mod2(np.random.random(N_colloids)*(L-2*radius)+radius)
    colloid_y = mod2(np.random.random(N_colloids)*(L-2*radius)+radius)

    colloid_x, colloid_y, colloid_s =sweep_mean(N_eq, colloid_x, colloid_y, colloid_s,0.)[:3]


    variancex= np.empty(len(tab_drift))
    variancey= np.empty(len(tab_drift))
    

    for i,drift in enumerate(tab_drift):
        #print("second:",drift/3)
        colloid_x, colloid_y, colloid_s =sweep_mean(N_eq, colloid_x, colloid_y, colloid_s,drift)[:3]

        colloid_x, colloid_y, colloid_s, time_corx, time_cory=sweep_noise(N_steps, colloid_x, colloid_y, colloid_s,meanfx[i],drift)

        variancex[i]=integrate.simpson( time_corx, dx=dt*save_rate) 
        variancey[i]=integrate.simpson(time_cory, dx=dt*save_rate)


    return variancex,variancey
# In[43]:

tab_drift = np.arange(0,3,.1)


    # 用于存储结果
results = [None] * cpus

print("initializing")
with ProcessPoolExecutor(max_workers=cpus) as executor:
    # 提交所有任务
    futures = [executor.submit(calculate_mean) for mp in range(cpus)]
    # collect results of mean force from each CPU
    for i, future in enumerate(futures):
        results[i] = future.result()
# results中的每个元素都是对应不同drift值的一个元组，这些元组包含fx, fy

fx=np.zeros(len(tab_drift))
fy=np.zeros(len(tab_drift))
Bx=np.zeros(len(tab_drift))
By=np.zeros(len(tab_drift))

for res in results:
    fx += res[0]/len(results)
    fy += res[1]/len(results)

with ProcessPoolExecutor(max_workers=cpus) as executor:
    # 提交所有任务
    futures = [executor.submit(calculate_variance, fx) for mp in range(cpus)]
    # collect results of time-integrated variance from each CPU
    for i, future in enumerate(futures):
        results[i] = future.result()
# results中的每个元素都是对应不同drift值的一个元组，这些元组包含fx, fy

for res in results:
    Bx += res[0]/len(results)
    By += res[1]/len(results)


# In[44]: save results to file

with h5py.File("ABM_landscape_{}_{}.h5".format(int_force,tumble_rate),"w") as opt:
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
    opt["dt"]=dt
    opt["N_colloids"]=N_colloids
    opt["N_steps"]=N_steps


