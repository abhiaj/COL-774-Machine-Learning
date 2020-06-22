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

theta_arr=np.empty((0,2))
array_j=[]
mod_x=np.empty((0,2))
y=np.array([])

def read_file():

    f = open("ass1/ass1_data/data/q1/linearX.csv", "r")
    x = np.empty((0,2))
    for i in f:
        #print(x.strip())
        ''' a is matrix of m*2 size, where the first column is 1 '''
        x=np.append(x, np.array( [[1,float(i.strip())]] ), axis=0)

    f.close()

    f=open("ass1/ass1_data/data/q1/linearY.csv","r")
    y=np.array([])
    for i in f:
        #print(i.strip())
        #y is vector of single dimension
        y=np.append(y,float(i.strip()))
    f.close()
    #print(a,y)
    return x,y

def calc_cost(x, y, theta):
    cost=0
#     for i in range (0,x.shape[0]):
    cost=np.dot( ( theta.dot(x.T) - y ), ( theta.dot(x.T) - y ).T )
#     print(cost.shape)
#     print(cost.shape)
#     print(cost[0].shape)
    return cost[0,0]/2

def normalize_data(x):
    y=x[:,1]
    m=np.mean(y)
    v=np.std(y)
    #print(m,v)
    y=x
    for i in range(0, x.shape[0]):
        y[i,1]=(y[i,1]-m)/v

    return y

def lin_reg():
    global mod_x,y
    x,y=read_file()

    mod_x=normalize_data(x)

    theta = np.zeros((1,2), dtype=np.float64)
    #print(theta)
    cost = calc_cost(mod_x,y,theta)
#     print(cost)
    diff=1
    temp = np.zeros([2,], dtype=np.float64)
    global theta_arr

    rate=0.0001
    stop=0.000001

    global array_j

    while(diff>stop):
        for i in range(0, temp.shape[0]):
            for j in range (0,mod_x.shape[0]):
                temp[i]+= rate*( y[j] - (theta.dot(mod_x[j])) )*mod_x[j,i]
        #print(temp)
        cost_old=calc_cost(mod_x,y,theta)
        #print(cost_old)
        for i in range(0, temp.shape[0]):
            theta[0,i]=temp[i]
        cost_new=calc_cost(mod_x,y,theta)
#         print(type(cost_new))
        diff=abs(cost_new-cost_old)

        array_j.append(cost_new)
        theta_arr=np.append(theta_arr,theta,axis=0)
        #fig = go.Figure(data=[go.Mesh3d(x=theta[0,0], y=theta[0,1], z=j, color='lightpink', opacity=0.50)])

    print("theta parameters:",theta)
    print("rate:",rate," stopping condition:",stop)
    print("final cost:",calc_cost(mod_x,y,theta))


    plt.plot(range(0,len(array_j)),array_j)
    plt.show()

#show_plots(mod_x,y,theta)


def show_plots(x,y,theta):
    plt.plot(x[:,1],y, '.', color='black')
    hy = []
    for j in range (0,x.shape[0]):
        hy.append(theta.dot(x[j]))
        #print(hy[j])

    plt.plot(x[:,1],hy, '.', color='blue')
    plt.show()


# In[2]:


lin_reg()








# In[ ]:


fig=plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim(0,1)
ax.set_ylim(0,0.0015)
ax.set_zlim(0, 60)
# ax.plot3D(theta_arr[:,0], theta_arr[:,1],array_j,'gray')
# plt.pause(1)
# plt.show()

for i in range (len(array_j)):
    ax.plot3D(theta_arr[0:i,0], theta_arr[0:i,1],array_j[0:i],'gray')
    
    print("A")  
    plt.pause(0.02)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




