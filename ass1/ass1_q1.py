#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %matplotlib notebook
# %matplotlib tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches

theta_arr = np.empty((0, 2))
array_j = []
mod_x = np.empty((0, 2))
y = np.array([])


def read_file():

    f = open("ass1/ass1_data/data/q1/linearX.csv", "r")
    x = np.empty((0, 2))
    for i in f:
        # print(x.strip())
        ''' a is matrix of m*2 size, where the first column is 1 '''
        x = np.append(x, np.array([[1, float(i.strip())]]), axis=0)

    f.close()

    f = open("ass1/ass1_data/data/q1/linearY.csv", "r")
    y = np.array([])
    for i in f:
        # print(i.strip())
        # y is vector of single dimension
        y = np.append(y, float(i.strip()))
    f.close()
    # print(a,y)
    return x, y


def calc_cost(x, y, theta):
    cost = 0
#     for i in range (0,x.shape[0]):
    cost = np.dot((theta.dot(x.T) - y), (theta.dot(x.T) - y).T)
#     print(cost.shape)
#     print(cost.shape)
#     print(cost[0].shape)
    return cost[0, 0] / 2


def normalize_data(x):
    y = x[:, 1]
    m = np.mean(y)
    v = np.std(y)
    # print(m,v)
    y = x
    for i in range(0, x.shape[0]):
        y[i, 1] = (y[i, 1] - m) / v

    return y


def lin_reg():
    global mod_x, y
    x, y = read_file()

    mod_x = normalize_data(x)

    theta = np.zeros((1, 2), dtype=np.float64)
    # print(theta)
    # cost = calc_cost(mod_x, y, theta)
#     print(cost)
    diff = 1
    temp = np.zeros([2, ], dtype=np.float64)
    global theta_arr

    rate = 0.0001
    stop = 0.000001

    global array_j

    while(diff > stop):
        for i in range(0, temp.shape[0]):
            for j in range(0, mod_x.shape[0]):
                temp[i] += rate * (y[j] - (theta.dot(mod_x[j]))) * mod_x[j, i]
        # print(temp)
        cost_old = calc_cost(mod_x, y, theta)
        # print(cost_old)
        for i in range(0, temp.shape[0]):
            theta[0, i] = temp[i]
        cost_new = calc_cost(mod_x, y, theta)
#         print(type(cost_new))
        diff = abs(cost_new - cost_old)

        array_j.append(cost_new)
        theta_arr = np.append(theta_arr, theta, axis=0)
        #fig = go.Figure(data=[go.Mesh3d(x=theta[0,0], y=theta[0,1], z=j, color='lightpink', opacity=0.50)])

    print("theta parameters:", theta)
    print("rate:", rate, " stopping condition:", stop)
    print("final cost:", calc_cost(mod_x, y, theta))

    plt.plot(range(0, len(array_j)), array_j)
    plt.show()

# show_plots(mod_x,y,theta)


def show_plots(x, y, theta):
    plt.plot(x[:, 1], y, '.', color='black')
    hy = []
    for j in range(0, x.shape[0]):
        hy.append(theta.dot(x[j]))
        # print(hy[j])

    plt.plot(x[:, 1], hy, '.', color='blue')
    plt.show()



# In[2]:


lin_reg()


# In[ ]:


# In[3]:


# import numpy as np
# import matplotlib.pyplot as plot
# import pylab

# # List of points in x axis
# XPoints     = []

# # List of points in y axis
# YPoints     = []

# # X and Y points are from -6 to +6 varying in steps of 2
# for val in range(-6, 8, 2):
#     XPoints.append(val)
#     YPoints.append(val)

# # Z values as a matrix
# ZPoints     = np.ndarray((7,7))

# # Populate Z Values (a 7x7 matrix) - For a circle x^2+y^2=z
# for x in range(0, len(XPoints)):
#     for y in range(0, len(YPoints)):
#         ZPoints[x][y] = (XPoints[x]* XPoints[x]) + (YPoints[y]*YPoints[y])

# # Print x,y and z values
# print(XPoints)
# print(YPoints)
# print(ZPoints)

# # Set the x axis and y axis limits
# pylab.xlim([-10,10])
# pylab.ylim([-10,10])

# # Provide a title for the contour plot
# plot.title('Contour plot')

# # Set x axis label for the contour plot
# plot.xlabel('X')

# # Set y axis label for the contour plot
# plot.ylabel('Y')

# # Create contour lines or level curves using matplotlib.pyplot module
# contours = plot.contour(XPoints, YPoints, ZPoints)

# # Display z values on contour lines
# plot.clabel(contours, inline=1, fontsize=10)

# # Display the contour plot
# plot.show()


# In[4]:


# fig=plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_xlim(0,1)
# ax.set_ylim(0,0.0015)
# ax.set_zlim(0, 60)
# # ax.plot3D(theta_arr[:,0], theta_arr[:,1],array_j,'gray')
# # plt.pause(1)
# # plt.show()

# for i in range (len(array_j)):
#     ax.plot3D(theta_arr[0:i,0], theta_arr[0:i,1],array_j[0:i],'gray')
#     plt.pause(0.02)

# plt.show()


# In[7]:


# Visualizing J(theta_0, theta_1) as a 3D-mesh followed from one of the quesions on stackoverflow
print('Visualizing J(theta_0, theta_1) ...\n')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ms = np.linspace(-0.2, 2.0, 100)
bs = np.linspace(-1, 1, 100)
M, B = np.meshgrid(ms, bs)
zs = np.array([
    calc_cost(mod_x, y, np.array([[mp], [bp]]).reshape(1, 2))
    for mp, bp in zip(np.ravel(M), np.ravel(B))
])
Z = zs.reshape(M.shape)
surf = ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b',
                       alpha=0.5, cmap=cm.coolwarm, linewidth=0, antialiased=False)
blue_patch = mpatches.Patch(color='blue', label='Cost Function')
red_patch = mpatches.Patch(color='red', label='Path taken by GD')
ax.plot3D(theta_arr[:, 0], theta_arr[:, 1], array_j, color='green', alpha=0.5)
ax.set_xlabel('theta(0)')
ax.set_ylabel('theta(1)')
ax.set_zlabel('J(theta)')
plt.legend(loc='upper left', handles=[blue_patch, red_patch])
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title('J(theta) vs theta')
# plt.show(block=False)
plt.show()
