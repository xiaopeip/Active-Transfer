# coding: utf-8

# In[1]:


import numpy as np
import numba as nb



import h5py

from concurrent.futures import ProcessPoolExecutor



# In[ ]:

cpus = 48

L = 20  # Box size

active_force = 3.  # strength of active force
tumble_rate = 3.0  # diffusion of the orientation
int_force= 1.95 #strength of interaction
medium_mobility=1.
N_colloids =  5  # number of particles
radius=.5

mass=40.


dt = 0.003  # time step size
save_rate=5000


N_steps = 800_000_000  # number of time steps to integrate
N_eq = 5000_000  # number of time steps to let the system equilibrate

passive=0.45 # left threshold of the bistable speed
peak=.95   # right threshold of the bistable speed


@nb.jit(nopython=True)
def mod2(x):
    # 使用 numpy.mod 将 x 限制在 [0, 2) 区间
    result = np.mod(x, L)
    return np.where(result > L/2, result - L, result)

@nb.jit(nopython=True)
def mod(x):
    # 使用 numpy.mod 将 x 限制在 [0, 2) 区间
    result = x%L
    if result >L/2:
        return result-L
    else:
        return result

@nb.jit(nopython=True)
def force2(colloid_x,colloid_y,probe_x,probe_y):
    #Lennard jones
    relative_x=mod2(colloid_x-probe_x)
    relative_y=mod2(colloid_y-probe_y)
    distance=np.sqrt(relative_x**2+relative_y**2) 
    return 169/42*((13/7)**(1/6))*((distance/radius)**(-13)-(distance/radius)**(-7))*int_force*relative_x/distance, 169/42*((13/7)**(1/6))*((distance/radius)**(-13)-(distance/radius)**(-7))*int_force*relative_y/distance



@nb.jit(nopython=True)
def force(colloid_x,colloid_y):
    #Lennard jones
    return force2(colloid_x,colloid_y,0.,0.)

@nb.jit(nopython=True)
def sweep_composite(N, colloid_x, colloid_y,colloid_s,probe_x,probe_y,probe_vx,probe_vy):
    '''
    Simulate the composite dynamics; record the probe velocity path; record the transition times between the two bistable states.
    Performs N time steps.
    Initial values (7 variables) should be given.
    '''

    transition_time_l=[]
    transition_time_r=[]
    probe_speed=np.sqrt(probe_vx**2+probe_vy**2)
    if probe_speed>peak:
        # previous=0/1 means probe velocity in left/right peak area
        previous=1
    else:
        previous=0
    
    path_v=np.empty((N//save_rate,2))

    for i in range(N):
        if i%save_rate==0:
            path_v[i//save_rate,0]=probe_vx
            path_v[i//save_rate,1]=probe_vy
            if i%1000000==0:
                print(i/N_steps)


        int_x, int_y = force2(colloid_x,colloid_y,probe_x,probe_y)

        d_colloid_x1= (active_force*np.cos(colloid_s)+ int_x ) * dt*medium_mobility
        d_colloid_y1 = (active_force*np.sin(colloid_s)+ int_y ) * dt*medium_mobility


        d_probex1 = probe_vx *dt
        d_probey1 = probe_vy *dt

        d_probevx1 = -np.sum(int_x)*dt/mass
        d_probevy1 = -np.sum(int_y)*dt/mass
        

        #path_v[i]=(((int_m + active_force * colloid_s-drift ) * dt+if_tumbles*(-2*dt+2*tumble_times)*colloid_s*active_force)/medium_friction)/dt
        colloid_s += np.random.normal(0,np.sqrt(2*tumble_rate*dt),N_colloids) 


        int_x2,int_y2=force2(colloid_x+d_colloid_x1,colloid_y+d_colloid_y1,probe_x+d_probex1,probe_y+d_probey1)

        d_colloid_x2 =  (active_force*np.cos(colloid_s)+ int_x2 ) * dt*medium_mobility
        d_colloid_y2 =   (active_force*np.sin(colloid_s)+ int_y2 ) * dt*medium_mobility

        d_probex2 = (probe_vx+d_probevx1)*dt
        d_probey2 = (probe_vy+d_probevy1)*dt

        d_probevx2 = -np.sum(int_x2)*dt/mass
        d_probevy2 = -np.sum(int_y2)*dt/mass

        colloid_x = mod2(colloid_x+(d_colloid_x1+d_colloid_x2)/2)
        colloid_y = mod2(colloid_y+(d_colloid_y1+d_colloid_y2)/2)

        probe_x = mod(probe_x+(d_probex1+d_probex2)/2)
        probe_y = mod(probe_y+(d_probey1+d_probey2)/2)

        probe_vx = probe_vx+(d_probevx1+d_probevx2)/2
        probe_vy = probe_vy+(d_probevy1+d_probevy2)/2
        #second order Ruge-Kutta method

        if d_probevx1>3 or d_probevx2>3 or d_probevy1>3 or d_probevy2>3:
            print(colloid_x,colloid_y)
            print(probe_x,probe_y)
            print(probe_vx,probe_vy)
            break

        probe_speed=np.sqrt(probe_vx**2+probe_vy**2)

        if probe_speed > peak:
            if previous!=1:
                previous=1
                transition_time_r.append(i*dt)
        if probe_speed < passive:
            if previous!=0:
                previous=0
                transition_time_l.append(i*dt)



    return colloid_x, colloid_y, colloid_s,probe_x,probe_y, probe_vx,probe_vy, transition_time_l,transition_time_r, path_v





# In[ ]:

def calculate(N_steps):

    colloid_s = np.random.random(N_colloids)*2*np.pi
    colloid_x = mod2(np.random.random(N_colloids)*(L-2*radius)+radius)
    colloid_y = mod2(np.random.random(N_colloids)*(L-2*radius)+radius)


    probe_x,probe_y,probe_vx,probe_vy=0.,0.,0.,0.



    colloid_x, colloid_y, colloid_s,probe_x,probe_y,probe_vx,probe_vy =sweep_composite(N_eq, colloid_x, colloid_y, colloid_s,probe_x,probe_y,probe_vx,probe_vy)[:7]
    transition_time_l, transition_time_r ,pathv =sweep_composite(N_steps, colloid_x, colloid_y, colloid_s,probe_x,probe_y,probe_vx,probe_vy)[7:]

    return pathv,transition_time_l,transition_time_r


results = [None] * cpus

with ProcessPoolExecutor(max_workers=cpus) as executor:
    # 提交所有任务
    futures = [executor.submit(calculate, N_steps) for k in range(cpus)]
    # 收集结果
    for i, future in enumerate(futures):
        results[i] = future.result()


pathvs=np.array([result[0] for result in results])

passage_l2r=[] # first passage time from left to right
passage_r2l=[] # first passage time from right to left

for result in results:
    print(result)
    if len(result[1])*len(result[0])!=0:
        if result[1][0]<result[2][0]:
            if len(result[1])==len(result[2]):
                passage_l2r.append(np.array(result[2])-np.array(result[1]))
                passage_r2l.append(np.array(result[1][1:])-np.array(result[2][:-1]))
            else:
                passage_l2r.append(np.array(result[2])-np.array(result[1][:-1]))
                passage_r2l.append(np.array(result[1][1:])-np.array(result[2]))
        else:
            if len(result[1])==len(result[2]):
                passage_r2l.append(np.array(result[1])-np.array(result[2]))
                passage_l2r.append(np.array(result[2][1:])-np.array(result[1][:-1]))
            else:
                passage_r2l.append(np.array(result[1])-np.array(result[2][:-1]))
                passage_l2r.append(np.array(result[2][1:])-np.array(result[1]))
# In[ ]:

passage_l2r=np.concatenate(passage_l2r)
passage_r2l=np.concatenate(passage_r2l)


with h5py.File("ABM_path_{}_{}_{}.h5".format(int_force,tumble_rate,mass), "w") as opt:
    opt["L"]=L
    opt["alpha"]=tumble_rate
    opt["mass"]=mass
    opt["N_medium"]=N_colloids
    opt["interaction"]=int_force
    opt["radius"]=radius
    opt["dt"]=dt
    opt["time_step"]=save_rate*dt
    opt["v0"]=active_force
    opt["passage_l2r"]=passage_l2r
    opt["passage_r2l"]=passage_r2l
    opt["path_v"]=pathvs


# In[ ]:

