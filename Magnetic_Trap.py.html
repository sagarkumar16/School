
# coding: utf-8

# In[229]:


import pandas as pd
import matplotlib.pyplot as plt
import statistics
from scipy.stats import maxwell
import numpy as np
from scipy.special import zeta


# In[2]:


from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


# ### All measurements and definitions in MKS

# ## Functions

# In[5]:


# Equations of Motion
def move(t):
    for t in range(T-1):
        # Lorentz Force
        # E, v, and B are all 3-vectors
        #F[t] = q*(np.cross(v[t], B[tuple(X[t])]))
        
        F[t] = q*(np.cross(v[t], B(tuple(X[t]))))
        
        X[t+1] = X[t] + v[t] + (1/2)*(F[t]/m)

        v[t+1] = X[t+1] - X[t]


# In[179]:


# Equations of Motion
def move_fn(time, charge, velocity, displacement, force, mag, m):
    for t in range(time-1):
        # Lorentz Force
        # v, and B are all 3-vectors
        
        force[t] = charge*(np.cross(velocity[t], mag(tuple(displacement[t]))))
        
        displacement[t+1] = displacement[t] + velocity[t] + (1/2)*(force[t]/m)

        velocity[t+1] = displacement[t+1] - displacement[t]


# In[189]:


def run(time, B, q, vx, vy, vz, xx, xy, xz, eq):
    
    m = 2*kb*temp/(np.pi*float(mean)**2)
    
    #Dependent

    # Initializing Dependent Matrices

    # v is a discretized velocity dim(v) = 3 x t
    vf = np.array([[0,0,0]]*(time), dtype = np.float)

    # This is the displacement matrix of dim(x) = 3 x t
    Xf = np.array([[0,0,0]]*(time), np.float)

    #F is a discretized Lorentz Force dim(F) = 3 x t
    Ff = np.array([[0,0,0]]*(time), np.float)
    
    
    #Initial Velocity, (0, e) where E = {0, 1, 2} corresponding to {x, y, z}
    vf[(0,0)] = vx
    vf[(0,1)] = vy
    vf[(0,2)] = vz
    
    #Initial Position
    Xf[(0,0)] = xx
    Xf[(0,1)] = xy
    Xf[(0,2)] = xz
    
    move_fn(time, q, vf, Xf, Ff, B, m)
    
    # plotting thre displacement, log-displacement, and phase space
    xsub = [i[0] for i in Xf]
    ysub = [i[1] for i in Xf]
    zsub = [i[2] for i in Xf]

    plt.figure(figsize = (20, 5))
    plt.suptitle('$\mathcal{O}(q * m * v)$ %s $\mathcal{O}(B)$' % (eq), fontsize = 16)


    plt.subplot(1,3,1)
    plt.plot(xsub, label='x')
    plt.plot(ysub, label = 'y')
    plt.plot(zsub, label = 'z')
    plt.ylabel('Displacement (m)')
    plt.xlabel('Time (steps)')
    #plt.fill_between(range(T), txmax, txmin, color = 'purple', alpha = 0.4)
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(xsub, label='x')
    plt.plot(ysub, label = 'y')
    plt.plot(zsub, label = 'z')
    plt.yscale('symlog')
    plt.ylabel('Displacement (m)')
    plt.xlabel('Time (steps)')
    #plt.yticks([10**])
    #plt.fill_between(range(T), txmax, txmin, color = 'purple', alpha = 0.4)
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(xsub, [vf[i][0] for i in range(time)], label = 'x')
    plt.plot(ysub, [vf[i][1] for i in range(time)], label = 'y')
    plt.plot(zsub, [vf[i][2] for i in range(time)], label = 'z')
    plt.legend()
    plt.xlabel('Displacement (m)')
    plt.ylabel('Velocity (m/s)')
    
    fig = plt.figure()
    fig.set_size_inches(10,10)
    ax = fig.gca(projection='3d')
    ax.plot(np.array(xsub), np.array(ysub), np.array(zsub))


# In[330]:


# Equations of Motion
def move_nonlin(t):
    for t in range(T-1):
        # Lorentz Force
        # E, v, and B are all 3-vectors
        #F[t] = q*(np.cross(v[t], B[tuple(X[t])]))
        
        F[t] = q*(np.cross(v(tuple(X[t])), B(tuple(X[t]))))
        
        X[t+1] = X[t] + v(X[t]) + (1/2)*(F[t]/m)


# # The Project

# In[6]:


# Set up a Magnetic Field
# Throw a 3D Maxwell Distribution of particles in
# Get the phase space diagrams


# In[7]:


# for a particle 


# In[8]:


temp = 400
kb = 1.380649e-23


# In[9]:


mean, var, skew, kurt = maxwell.stats(moments ='mvsk')
fig, ax = plt.subplots(1, 1)
x = np.linspace(maxwell.ppf(0.01),maxwell.ppf(0.99), 100)
ax.plot(x, maxwell.pdf(x), 'r-', lw=4, alpha=0.6, label='maxwell pdf')
plt.title('1D Maxwell-Boltzmann Distribution')
plt.xlabel('Speed (m/s)') # fix this
plt.ylabel('Probability Density (s/m)')


# In[10]:


m = 2*kb*temp/(np.pi*float(mean)**2)
print('At STP, \nmass of particle = %s kg' % (m))


# In[11]:


std = var**1/2
v_mean = float(mean)


# In[361]:


v_mean


# ### E[0,0,10^-21]

# In[190]:


B = lambda e: [0,0,1e-21]
run(70, B, 1, v_mean, v_mean, v_mean, 0, 0, 0, '=')


# In[362]:


B = lambda e: [0,0,1e-22]
run(70, B, 1, v_mean, v_mean, v_mean, 0, 0, 0, '- 1 =')


# ### B = [cos(x), sin(y), 0]

# In[203]:


B = lambda e: [1e-21*np.cos(e[0]), 1e-21*np.sin(e[1]), 0]
run(200, B, 1, v_mean, v_mean, v_mean, 0, 0, 0, '=')


# In[204]:


B = lambda e: [1e-20*np.cos(e[0]), 1e-20*np.sin(e[1]), 0]
run(200, B, 1, v_mean, v_mean, v_mean, 0, 0, 0, '=')


# In[207]:


B = lambda e: [1e-22*np.cos(e[0]), 1e-22*np.sin(e[1]), 0]
run(200, B, 1, v_mean, v_mean, v_mean, 0, 0, 0, '- 1 =')


# ### B = [1/x, 0, 0]

# In[214]:


B = lambda e: [1e-21*e[0]**-1, 0, 0]
run(200, B, 1, v_mean, v_mean, v_mean, 1, 1, 1, '- 1 =')


# In[239]:


B = lambda e: [1e-19*e[0]**-1, 0, 0]
run(200, B, 1, v_mean, v_mean, v_mean, 1, 1, 1, '=')


# ### B = [cos(x)^2, sin(y)^2, tan(z)^2]

# In[246]:


B = lambda e: [1e-21*np.sin(e[0])**2, 1e-21*np.sin(e[1])**2, 1e-21*np.tan(e[2])**2]
run(200, B, 1, v_mean, v_mean, v_mean, 0, 0, 0, '=')


# In[251]:


B = lambda e: [1e-22*np.sin(e[0])**2, 1e-22*np.sin(e[1])**2, 1e-22*np.tan(e[2])**2]
run(200, B, 1, v_mean, v_mean, v_mean, 0, 0, 0, '=')


# ### Initial V for each time step gets randomly chosen from the distribution

# In[346]:


T = 70
q = 1


# In[347]:


#Dependent

# Initializing Dependent Matrices

# v is a discretized velocity dim(v) = 3 x t
v = np.array([[0,0,0]]*T, dtype = np.float)

# This is the displacement matrix of dim(x) = 3 x t
X = np.array([[0,0,0]]*T, np.float)

#F is a discretized Lorentz Force dim(F) = 3 x t
F = np.array([[0,0,0]]*T, np.float)


# In[348]:


for e in range(3):
    val = maxwell.rvs(size = T)
    for n in range(T):
        v[n][e] = val[n]


# In[349]:


#Initial Position
X[(0,0)] = 0
X[(0,1)] = 0
X[(0,2)] = 0


# In[350]:


B = lambda e: [0, 0, 1e-21]


# In[351]:


move(T)


# In[352]:


# plotting thre displacement, log-displacement, and phase space
xsub = [i[0] for i in X]
ysub = [i[1] for i in X]
zsub = [i[2] for i in X]

plt.figure(figsize = (20, 5))
plt.suptitle('$\mathcal{O}(q * m)$ = $\mathcal{O}(B)$', fontsize = 16)



plt.subplot(1,3, 1)
plt.plot(xsub, label='x')
plt.plot(ysub, label = 'y')
plt.plot(zsub, label = 'z')
plt.ylabel('Displacement (m)')
plt.xlabel('Time (steps)')
#plt.fill_between(range(T), txmax, txmin, color = 'purple', alpha = 0.4)
plt.legend()

plt.subplot(1,3, 2)
plt.plot(xsub, label='x')
plt.plot(ysub, label = 'y')
plt.plot(zsub, label = 'z')
plt.yscale('symlog')
plt.ylabel('Displacement (m)')
plt.xlabel('Time (steps)')
#plt.yticks([10**])
#plt.fill_between(range(T), txmax, txmin, color = 'purple', alpha = 0.4)
plt.legend()

plt.subplot(1,3,3)
plt.plot(xsub, [v[i][0] for i in range(T)], label = 'x')
plt.plot(ysub, [v[i][1] for i in range(T)], label = 'y')
plt.plot(zsub, [v[i][2] for i in range(T)], label = 'z')
plt.legend()
plt.xlabel('Displacement (m)')
plt.ylabel('Velocity (m/s)')

fig = plt.figure()
fig.set_size_inches(10,10)
ax = fig.gca(projection='3d')
ax.plot(np.array(xsub), np.array(ysub), np.array(zsub))

### Functional Velocity
# In[354]:


T = 70
q = 1


# In[355]:


#Dependent

# Initializing Dependent Matrices

# v is a discretized velocity dim(v) = 3 x t
#v = np.array([[0,0,0]]*T, dtype = np.float)

# This is the displacement matrix of dim(x) = 3 x t
X = np.array([[0,0,0]]*T, np.float)

#F is a discretized Lorentz Force dim(F) = 3 x t
F = np.array([[0,0,0]]*T, np.float)


# In[356]:


v = lambda e: [np.cos(e[0]), np.sin(e[1]), 0]


# In[357]:


#Initial Position
X[(0,0)] = 0
X[(0,1)] = 0
X[(0,2)] = 0


# In[358]:


B = lambda e: [0, 0, 1e-20]


# In[359]:


move_nonlin(T)


# In[360]:


# plotting thre displacement, log-displacement, and phase space
xsub = [i[0] for i in X]
ysub = [i[1] for i in X]
zsub = [i[2] for i in X]

plt.figure(figsize = (20, 5))
plt.suptitle('$\mathcal{O}(q * m)$ = $\mathcal{O}(B)$', fontsize = 16)



plt.subplot(1,3, 1)
plt.plot(xsub, label='x')
plt.plot(ysub, label = 'y')
plt.plot(zsub, label = 'z')
plt.ylabel('Displacement (m)')
plt.xlabel('Time (steps)')
#plt.fill_between(range(T), txmax, txmin, color = 'purple', alpha = 0.4)
plt.legend()

plt.subplot(1,3, 2)
plt.plot(xsub, label='x')
plt.plot(ysub, label = 'y')
plt.plot(zsub, label = 'z')
plt.yscale('symlog')
plt.ylabel('Displacement (m)')
plt.xlabel('Time (steps)')
#plt.yticks([10**])
#plt.fill_between(range(T), txmax, txmin, color = 'purple', alpha = 0.4)
plt.legend()

plt.subplot(1,3,3)
plt.plot(xsub, [v(X[i])[0] for i in range(T)], label = 'x')
plt.plot(ysub, [v(X[i])[1] for i in range(T)], label = 'y')
plt.plot(zsub, [v(X[i])[2] for i in range(T)], label = 'z')
plt.legend()
plt.xlabel('Displacement (m)')
plt.ylabel('Velocity (m/s)')

fig = plt.figure()
fig.set_size_inches(10,10)
ax = fig.gca(projection='3d')
ax.plot(np.array(xsub), np.array(ysub), np.array(zsub))

