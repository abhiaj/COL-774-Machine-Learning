#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
import plotly
import plotly.graph_objects as go
import time
import math
from mpl_toolkits import mplot3d


x1 = np.random.normal(3,2,1000000)
#plt.hist(x1, 30, density=True)

x2 = np.random.normal(-1,2,1000000)
print(x2.shape)
#plt.hist(x2, 30, density=True)

er = np.random.normal(0,math.sqrt(2),1000000)
#plt.hist(er, 30, density=True)

theta = np.array([3, 1, 2], dtype=float)
print(theta)
theta0 = np.full(1000000, theta[0])
print(theta0.shape)

y = theta0 + np.dot(theta[1], x1) + np.dot(theta[2], x2) + er

plt.hist(y, 30, density=True)
plt.show()


# In[2]:


print(x1.shape)


# In[3]:


x1[0]


# In[4]:


x0 = np.full(1000000, 1)
data = np.column_stack((x0,x1,x2,er))


np.random.shuffle(data)


y = theta0 + np.dot(theta[1], x1) + np.dot(theta[2], x2) + er
plt.hist(y, 30, density=True)

y = np.dot(theta[0], data[:,0]) + np.dot(theta[1], data[:,1]) + np.dot(theta[2], data[:,2]) + data[:,3]
plt.hist(y, 30, density=True)
plt.show()
print(theta)
print(theta.T)


# In[9]:


def calc_cost(data, y, theta, batch_num, r):
    cost=0
    index=batch_num*r
    for i in range (0,r):
        hx = np.dot(theta,data[index,0:3])
        cost+=( ( hx - y[index] )**2 )
        index+=1
    return cost/(2*r)


# In[10]:


print(theta.T*data[0,0:3])
np.dot(theta,data[0,0:3])


# In[15]:


theta = np.array([0,0,0], dtype=float)
m = 1000000
r = 100
num_batch = int(m/r)
rate = 0.001
temp = np.array(theta)
cost_old = cost_new = -1
array_j=[]
array_theta=np.array([])
array_theta= np.append(array_theta,theta,axis=0)
print(theta)

for i in range(0, num_batch):
    for j in range (0, r):
        index = i*r+j
        hx = np.dot(theta,data[index,0:3])
        #print(hx)
        for k in range (0, theta.shape[0]):
            temp[k]+= rate*( y[index] - hx )*data[index,k]
        
        theta = temp
    
    cost_new = calc_cost(data,y,theta,i,r)
    diff=abs(cost_new-cost_old)
    array_j.append(cost_new)
    array_theta= np.append(array_theta,theta,axis=0)
    
    cost_old=cost_new
    if(diff<0.000003):
        break
    #print(theta)

plt.plot(range(0,len(array_j)),array_j)
plt.show()
print(theta)


# In[16]:


print(array_theta.shape)

num_rows_theta_arr=array_theta.shape[0]//3
t=np.resize(array_theta,(num_rows_theta_arr,3))

print(t.shape)

fig=plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim(0,3.1)
ax.set_ylim(0,1.1)
ax.set_zlim(0, 2.1)

for i in range(t.shape[0]):
    ax.plot3D(t[0:i,0],t[0:i,1],t[0:i,2],'gray')
    plt.pause(0.2)
    
# print(t[:,0].shape)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




