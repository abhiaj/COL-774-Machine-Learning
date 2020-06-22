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

tx1 = []
tx2 = []
ty = []
tm = 0

f = open("ass1/ass1_data/data/q2/q2test.csv", "r")

for i in f:
    #print(x.strip())
    ''' a is matrix of m*2 size, where the first column is 1 '''
    i = i.strip()
    data = i.split(",")
    if(data[2]=="Y"):
        continue
    tx1.append(data[0])
    tx2.append(data[1])
    ty.append(data[2])
f.close()


# In[2]:


m = 1000000
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

# plt.hist(y, 30, density=True)
# plt.show()


# In[3]:


print(x1.shape)


# In[4]:


x1[0]


# In[5]:


x0 = np.full(1000000, 1)
data = np.column_stack((x0,x1,x2,er))


# In[6]:


data


# In[7]:


np.random.shuffle(data)


# In[8]:


data.shape


# In[9]:


y = theta0 + np.dot(theta[1], x1) + np.dot(theta[2], x2) + er
plt.hist(y, 30, density=True)

y = np.dot(theta[0], data[:,0]) + np.dot(theta[1], data[:,1]) + np.dot(theta[2], data[:,2]) + data[:,3]
# plt.hist(y, 30, density=True)
# plt.show()
print(theta.shape)
print(y.shape)

X = data[:,0:3].T
print(X.shape)
Y = y.reshape(1,m).astype(float)
print(Y.shape)
theta = np.zeros((3,1))
print(theta.shape)


# In[10]:


def calc_cost(X, Y, theta, i, r):
    s_i = i*r
    f_i = i*r + r - 1
    cost=np.dot(  theta.T.dot(X[:,s_i:f_i+1]) - Y[:,s_i:f_i+1] , ( theta.T.dot(X[:,s_i:f_i+1]) - Y[:,s_i:f_i+1] ).T )
    return cost[0,0]/(2*r)
def calc_cost_batch(X, Y, theta):
    cost=np.dot(  theta.T.dot(X) - Y , ( theta.T.dot(X) - Y ).T )
    return cost[0,0]/(2*X.shape[1])


# In[ ]:





# In[11]:


# theta = np.array([0,0,0], dtype=float)
m = 1000000
r = 1000000
num_batch = int(m/r)
rate = 0.001
stop = 0.00000000001
temp = np.array(theta)
cost_old = cost_new = -1
array_j=[]
array_theta=np.empty((0,3))
array_theta= np.append(array_theta,theta.T,axis=0)
theta_old=theta
error = 1
num_epoch = 0
num = 0
print(theta)
while(1):
    av_gd = 0
    for i in range(0, num_batch):
        
        
        s_i = i*r
        f_i = i*r + r - 1
        gd = rate * np.matmul(np.subtract(Y[:,s_i:f_i+1], np.matmul(theta.T, X[:,s_i:f_i+1])), X[:,s_i:f_i+1].T)
#         for j in range (0, r):
#             index = i*r+j
#             hx = np.dot(theta,data[index,0:3])
#             print(hx)
#             for k in range (0, theta.shape[0]):
#                 temp[k]+= rate*( y[index] - hx )*data[index,k]
        gd = gd/r
#         print(gd)
#             print(temp)
        theta_old = theta
        theta = np.add(theta, gd.T)
        
#         theta = theta + temp
        
        cost_old=cost_new
        cost_new = calc_cost(X,Y,theta,i,r)
        diff=abs(cost_new-cost_old)
        error=np.max(np.abs(theta - theta_old))
        errorp = error/np.max(theta)*100
        array_j.append(cost_new)
        array_theta= np.append(array_theta,theta.T,axis=0)
        av_gd += gd
#         print(len(array_theta))
#         print(cost_old,cost_new)
#         print(theta, np.max(gd))
    num_epoch += 1
    av_gd /= r
    av_gd = np.max(av_gd)
    print("num_epoch:",num_epoch)
    if(av_gd<stop):
        break
    
        #print(theta)

print("theta parameters:",theta)
print("rate:",rate," stopping condition:",stop)
print("Number of epochs:",num_epoch)
# plt.plot(range(0,len(array_j)),array_j)
# plt.show()
# print(theta)
# print(cost_new)


# In[12]:


print(theta)


# In[13]:


tm = len(ty)

tx1 = np.array(tx1).astype(float)
# tx1 = (tx1 - np.mean(tx1)) / (np.std(tx1))

tx2 = np.array(tx2).astype(float)
# tx2 = (tx2 - np.mean(tx2)) / (np.std(tx2))
tx0 = np.full(tm, 1)
# print(x0)

ty=np.array(ty).astype(float)


# In[14]:


tX=np.column_stack((tx0,tx1,tx2)).T
tY=ty.reshape(10000,1).T

print(tX.shape, tY.shape)


# In[15]:


theta_orig = np.array([3,1,2]).reshape(3,1).astype(float)
print(theta_orig,theta_orig.shape)


# In[16]:


cost_lern_para = calc_cost_batch(tX,tY,theta)
cost_orig_para = calc_cost_batch(tX,tY,theta_orig)
print("rate:",rate)
print("stopping condition:",stop)
print("cost with learnt parameters:",cost_lern_para)
print("cost with original parameters:",cost_orig_para)


# In[17]:


# print(calc_cost(tX,tY,theta,0,tm))


# In[ ]:


print(array_theta.shape)
print(np.max(array_theta[:,1]))
# num_rows_theta_arr=array_theta.shape[0]//3
# t=np.resize(array_theta,(num_rows_theta_arr,3))
print(array_theta)
# print(t.shape)
t = array_theta
fig=plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('theta(0)')
ax.set_ylabel('theta(1)')
ax.set_zlabel('theta(2)')
ax.set_xlim(0,np.max(array_theta[:,0]))
ax.set_ylim(0,np.max(array_theta[:,1]))
ax.set_zlim(0,np.max(array_theta[:,2]))

for i in range(0, t.shape[0], 20):
    ax.plot3D(t[0:i,0],t[0:i,1],t[0:i,2],'gray')
    plt.pause(0.02)
    
# print(t[:,0].shape)

plt.show()


# In[ ]:




