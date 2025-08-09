import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pandas as pd
import warnings
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings("ignore", message=".*Casting complex values to real discards the imaginary part.*")
#%%
n = 2000     # Number of oscillators
x0 = 0
t_fin=50    
t_span = [0, t_fin]
t_eval = np.linspace(*t_span, 100)

def lorentzian_pdf(x, x0, gamma):
    # return (1/np.pi) * (gamma / ((x - x0)**2 + gamma**2))
    return (1/np.sqrt(2*np.pi*gamma**2))*np.exp(-(x-x0)**2 / (2*gamma**2))


def sample_lorentzian(n, x0=0, gamma=1, x_range=(-10, 10)):
    samples = []
    max_pdf = lorentzian_pdf(x0, x0, gamma) + lorentzian_pdf(-x0, x0, gamma)  # Estimate max for rejection

    while len(samples) < n:
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(0, max_pdf)
        if y < lorentzian_pdf(x, x0, gamma):
            samples.append(x)

    return np.array(samples)

omega_list = sample_lorentzian(n, x0=x0, gamma=1.0)


#%%
def z1(theta_array):
    r = np.mean(np.exp(1j * theta_array))
    return np.abs(r)

def diff(t,theta_thetadot_list):
    thetaddot_out=[]
    theta_list= theta_thetadot_list[:n]
    thetadot_list= theta_thetadot_list[n:]
    r=z1(theta_list)
    for i in range(n):
        omega_i = omega_list[i]
        theta_i = theta_list[i]
        thetadot_i = thetadot_list[i]
        thetaddot_out.append((-thetadot_i+ omega_i - K1*r*np.sin(theta_i))/m) 

    return np.concatenate([thetadot_list, thetaddot_out])

 

def solution(k1,M,init):
    global K1
    global m 
    m=M
    K1= k1
    sol = solve_ivp(diff, t_span, init, t_eval=t_eval,method='RK45')
    order_par = [z1(sol.y.T[i][:n]) for i in range(len(sol.y.T))]
    mean_r = np.mean(order_par[5*len(order_par)//6:])
    return [t_eval,order_par,mean_r]

def protocol(k,omega_s,mass):
    rand= np.random.uniform(0, 2 * np.pi, n)
    theta=[]
    thetadot=[]
    for i in range(len(omega_list)):
        if omega_list[i] < omega_s:
            theta.append(0)
            thetadot.append(0)
        else:
            theta.append(rand[i])
            thetadot.append(omega_list[i])
    init = np.concatenate([theta, thetadot])
    sol = solution(k, mass, init)
    return sol[2]

k_list=np.linspace(2, 2.6, 50)
omegas=np.arange(0, 3,0.005)

# z=[]
# for i in tqdm(k_list):
#     for j in omegas:
#         temp=protocol(i, j, 6)
#         z.append(temp)


def run_protocols(k_list, omegas, mass):
    output = []
    k_output = []
    for k in tqdm(k_list):
        check=1
        for omega_s in omegas:
            if check==1:
                temp=protocol(k, omega_s, mass)
                if temp>=0.2:
                    output.append(omega_s)
                    k_output.append(k)
                    check=0

    return k_output, output


x,y=run_protocols(k_list, omegas, 6)
#%%
plt.plot(x, y,marker='*',linestyle='--')
plt.xlabel('k')
plt.ylabel('Omega_s')
# plt.title('Protocol Results')
# plt.xlim(0, 3)
plt.grid()
plt.show()

#%% 

# Example: using k_list, omegas, and z for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Prepare meshgrid for k_list and omegas
K, O = np.meshgrid(k_list, omegas)
Z = np.array(z).reshape(len(k_list), len(omegas)).T  # shape: (len(omegas), len(k_list))

# Plot the surface
surf = ax.plot_surface(K, O, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('k')
ax.set_ylabel('omega_s')
ax.set_zlabel('mean_r')
ax.set_title('3D Surface Plot')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
# %%
