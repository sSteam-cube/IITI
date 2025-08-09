#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*Casting complex values to real discards the imaginary part.*")
#%%

a=1.1     # Coupling strength
b=1.2
t_fin=100    # Lorentzian width
t_span = [0, t_fin]
t_eval = np.linspace(*t_span, 500)

def diff(t,theta_thetadot_list):
    out=[]
    theta= theta_thetadot_list[0]
    thetadot= theta_thetadot_list[1]
    alpha=a  
    beta = b

    out.append(thetadot)
    out.append(-alpha * thetadot + beta - np.sin(theta))
    return out

def mod(theta):
    if -np.pi <= theta <= np.pi:
        return theta
    elif theta > np.pi:
        temp=theta%(2*np.pi)
        return temp - np.pi
    elif theta < -np.pi:
        temp=theta%(2*np.pi)
        return temp + np.pi
    
theta_init = np.random.uniform(0, 2*np.pi,8)
thetadot_init = np.random.uniform(-3, 3 ,8)
init_list= [[i,j] for i,j in zip(theta_init, thetadot_init)]
# init_list = np.array(init_list).flatten()

def solution(init_list):
    for i in init_list:
        sol = solve_ivp(diff, t_span, i, t_eval=t_eval, method='RK45')
        x= sol.y[0]
        y= sol.y[1]
        x_new=[np.mod(i, 2*np.pi) for i in x]
        plt.scatter(x_new, y, s=5)
    plt.xlabel('Theta')
    plt.ylabel('Theta dot')
    # plt.legend()
    plt.xlim(0,  2*np.pi)
    plt.ylim(-3, 3)
    plt.show()
    return sol
solution(init_list)

# Visualize the flow in phase space using a vector field (quiver plot)
theta_vals = np.linspace(-np.pi, np.pi, 20)
thetadot_vals = np.linspace(-5, 5, 20)
Theta, Thetadot = np.meshgrid(theta_vals, thetadot_vals)

dTheta = Thetadot
dThetadot = -a * Thetadot + b - np.sin(Theta)

plt.figure(figsize=(8, 6))
plt.quiver(Theta, Thetadot, dTheta, dThetadot, color='r', angles='xy')
plt.xlabel('Theta')
plt.ylabel('Theta dot')
plt.title('Phase Space Flow')
plt.xlim(-np.pi, np.pi)
# plt.ylim(-2, 2)
plt.grid(True)
plt.show()