#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
#%%
#x=K, y=r
m=2
d=1
x=np.linspace(1,20,100)
y=np.linspace(1,0.01,100)

def g(x):
    return (1/np.sqrt(2*np.pi*d**2))*(np.exp(-(x**2)/(2*d**2)))

def protocol(x,y):
    def f1(u):
        return (np.cos(u))**2 * g(x*y*np.sin(u))
    def f2(u): 
        return g(u)/u**2
    omega_p=(4/np.pi)*np.sqrt(x*y/m)
    theta_p=np.arcsin(omega_p/(x*y))
    int1= (-x*y/m)*integrate.quad(f2, x*y, np.inf)[0] + x*y*integrate.quad(f1,-np.pi/2, np.pi/2)[0]
    int2=  x*y*integrate.quad(f1,-theta_p, theta_p)[0] + (-x*y/m)*integrate.quad(f2, omega_p, np.inf)[0]
    return y-int1, y-int2

X, Y = np.meshgrid(y,x, indexing='ij')
coords = np.stack((X,Y), axis=-1) 
print(coords[2,2])

val1=np.zeros((len(x), len(y)))
val2=np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        val1[i,j],val2[i,j] = protocol(coords[i,j][1], coords[i,j][0])  
    
print(val2[1,2])
#%%
stand=0

def marching(val,coords):
    x_out=[]
    y_out=[]
    a_x= coords[1,2][1]-coords[1,1][1]
    a_y= coords[2,1][0]-coords[1,1][0]
    for i in range(val.shape[0]-1):
        for j in range(val.shape[1]-1):
            if (val[i,j] < stand and val[i+1,j] > stand) or (val[i,j] > stand and val[i+1,j] < stand):
                tempy= coords[i,j][0] + a_y * (abs(val[i,j]) / (abs(val[i,j]) + abs(val[i+1,j])))
                tempx = coords[i,j][1]
                y_out.append(tempy)
                x_out.append(tempx)
            if (val[i,j] < stand and val[i,j+1] > stand) or (val[i,j] > stand and val[i,j+1] < stand):
                tempy = coords[i,j][0]
                tempx = coords[i,j][1] + a_x * (abs(val[i,j]) / (abs(val[i,j]) + abs(val[i,j+1])))
                y_out.append(tempy)
                x_out.append(tempx)

    return x_out, y_out

x1,y1= marching(val1, coords)
x2,y2= marching(val2, coords)

plt.plot(x1, y1)
plt.plot(x2, y2)
plt.show()


# %%
