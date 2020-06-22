import numpy as np 
import matplotlib.pyplot as plt 
import plotly
import plotly.graph_objects as go
import time
import math


def read_file():
	f = open("ass1_data/data/q1/linearX.csv", "r")
	x = np.empty((0,2))
	for i in f:
		#print(x.strip())
		''' a is matrix of m*2 size, where the first column is 1 '''
		x=np.append(x, np.array( [[1,float(i.strip())]] ), axis=0)

	f.close()

	f=open("ass1_data/data/q1/linearY.csv","r")
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
	for i in range (0,x.shape[0]):
		cost+=( ( theta.dot(x[i]) - y[i] )**2 )
	return cost[0]/2

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
	x,y=read_file()

	mod_x=normalize_data(x)

	theta = np.zeros((1,2), dtype=np.float64)
	#print(theta)
	cost = calc_cost(mod_x,y,theta)
	#print(cost)
	diff=1
	temp = np.zeros([2,], dtype=np.float64)
	
	rate=0.0001
	stop=0.000001

	array_j=[]

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
		diff=abs(cost_new-cost_old)

		array_j.append(cost_new)
		fig = go.Figure(data=[go.Mesh3d(x=theta[0,0], y=theta[0,1], z=j, color='lightpink', opacity=0.50)])
	#print(calc_cost(x,y,theta))
	fig.show()
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
	time.sleep(1)
	plt.close()

def show_3dmesh(j,theta):
	
	fig.show()

'''q2'''
def generate_sample():

	x1 = np.random.normal(3,2,1000000)
	#plt.hist(x1, 30, density=True)

	x2 = np.random.normal(-1,2,1000000)
	print(x2.shape)
	#plt.hist(x2, 30, density=True)

	er = np.random.normal(0,math.sqrt(2),1000000)
	#plt.hist(er, 30, density=True)

	theta = np.array([3, 1, 2])

	theta0 = np.full(1000000, theta[0])
	print(theta0.shape)

	y = theta0 + np.dot(theta[1], x1) + np.dot(theta[2], x2) + er

	plt.hist(y, 30, density=True)
	plt.show()


#lin_reg()
#read_file()
generate_sample()