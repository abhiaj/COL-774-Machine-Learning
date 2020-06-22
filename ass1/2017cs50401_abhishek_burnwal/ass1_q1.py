#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'tk')
import numpy as np 
import matplotlib.pyplot as plt 
import plotly
import plotly.graph_objects as go
import time
import math
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.patches as mpatches


# In[2]:


def calc_cost(X, Y, theta):
    cost=np.dot(  theta.T.dot(X) - Y , ( theta.T.dot(X) - Y ).T )
    return cost[0,0]/(2*X.shape[1])


# In[3]:


def show_plots(x,y,theta):
    plt.plot(x[1,:],y.T, '.', color='black',label='Data points')
    hy = []
    hy = np.matmul(theta.T, x)
        #print(hy[j])
    
    plt.plot(x[1,:],hy.T, '.', color='blue',label='Hypothesis')
    plt.xlabel('x, acidity of wine -->')
    plt.ylabel('y, density -->')
    plt.title('Acidity vs density plot')
    plt.legend()
    plt.show()


# In[4]:


theta_arr = np.empty((0,2))
array_j = []
x1 = []
y = []
theta = []
m = 0

f = open("ass1/ass1_data/data/q1/linearX.csv", "r")

for i in f:
    #print(x.strip())
    ''' a is matrix of m*2 size, where the first column is 1 '''
    x1.append(i.strip())
f.close()

f=open("ass1/ass1_data/data/q1/linearY.csv","r")

for i in f:
    #print(i.strip())
    #y is vector of single dimension
    y.append(i.strip())
f.close()

m = len(y)

x1 = np.array(x1).astype(float)
x1 = (x1 - np.mean(x1)) / (np.std(x1))
x0 = np.full(m, 1)
# print(x0)

y=np.array(y).astype(float)


# In[5]:


X=np.column_stack((x0,x1)).T
Y=y.reshape(100,1).T

print(X.shape, Y.shape)


# In[6]:


theta = np.zeros((2,1))
print(theta)


# In[7]:


rate=0.005
stop=0.0000000001
cost_new = calc_cost(X, Y,theta)
array_j.append(cost_new)
theta_arr=np.append(theta_arr,theta.T,axis=0)
cost_old = 0
diff = 1

while(diff>stop):
    gd = rate * np.matmul(np.subtract(Y, np.matmul(theta.T, X)), X.T)
    gd /= m
#     print(gd)
    theta = np.add(theta, gd.T)
    cost_old = cost_new
    cost_new = calc_cost(X, Y, theta)
    diff = abs(cost_old-cost_new)
    array_j.append(cost_new)
    theta_arr=np.append(theta_arr,theta.T,axis=0)

print("theta parameters:",theta)
print("rate:",rate," stopping condition:",stop)
print("final cost:",calc_cost(X,Y,theta))
print("Number of iterations:",len(array_j))

#     print(array_j)
# plt.plot(range(len(array_j)),array_j)
# plt.show()


# In[8]:


show_plots(X,Y,theta)


# In[ ]:





# In[9]:


print('J(theta_0, theta_1) ...\n')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ms = np.linspace(-0.1, 2.0, 200)
bs = np.linspace(-1, 1, 200)
M, B = np.meshgrid(ms, bs)
zs = np.array([
    calc_cost(X, Y, np.array([[mp], [bp]]).reshape(2,1))
    for mp, bp in zip(np.ravel(M), np.ravel(B))
])
Z = zs.reshape(M.shape)
surf = ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.4, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.set_xlabel('theta(0)')
ax.set_ylabel('theta(1)')
ax.set_zlabel('J(theta)')
ax.plot([theta[0,0]], [theta[1,0]], calc_cost(X,Y,theta), color='r', marker='x', label='Optimal Value')

for i in range (len(array_j)):
    ax.plot3D(theta_arr[0:i,0], theta_arr[0:i,1],array_j[0:i],'red')
#     print("A")
    plt.pause(0.002)


# In[10]:


print(theta.shape)
print('\nPlotting the contour ... \n')
fig = plt.figure()

CS = plt.contour(M, B, Z, 25)
plt.clabel(CS, inline=1, fontsize=8)
plt.plot([theta[0,0]], [theta[1,0]], color='r', marker='x', label='Optimal Value')
for i in range(len(array_j)):
    plt.plot(theta_arr[0:i,0], theta_arr[0:i,1], color='r')
    plt.draw()
    plt.pause(0.02)
plt.legend()
plt.xlabel('Theta0 -->')
plt.ylabel('Theta1 -->')
plt.title('Contour plot')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




