import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*Casting complex values to real discards the imaginary part.*")
#%%
n = 500     # Number of oscillators
K1 =0     # Coupling strength
x0 = 0       # Lorentzian center
m=1
t_fin=50    # Lorentzian width
t_span = [0, t_fin]
t_eval = np.linspace(*t_span, 100)
def lorentzian_pdf(x, x0, gamma):
    return (1/np.sqrt(2*np.pi*gamma**2))*np.exp(-(x-x0)**2 / (2*gamma**2))

def sample_lorentzian(n, x0=0, gamma=1, x_range=(-50, 50)):
    samples = []
    max_pdf = lorentzian_pdf(x0, x0, gamma) + lorentzian_pdf(-x0, x0, gamma)  # Estimate max for rejection

    while len(samples) < n:
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(0, max_pdf)
        if y < lorentzian_pdf(x, x0, gamma):
            samples.append(x)

    return np.array(samples)

omega_list = sample_lorentzian(n, x0=x0, gamma=1.0)

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
    next_init= np.concatenate([sol.y.T[-1][:n], sol.y.T[-1][n:]])
    return [t_eval,order_par,mean_r,next_init]

def find_omega_max(state,omega_list,threshold=0.1):
    thetadot_list = state[n:]
    temp=0
    for i in range(n):
        if abs(thetadot_list[i])<= threshold and abs(omega_list[i])>=temp:
            temp=abs(omega_list[i])
    return temp

rand_init = np.concatenate([np.random.uniform(0, 2 * np.pi, n), omega_list])

def skibidi(K_start, K_end, deltaK=0.1, mass=2):
    
    K_values = np.arange(K_start, K_end + deltaK, deltaK)
    results = []
    rev_results = []
    check = 0
    for K in tqdm(K_values):
        if check==0:
            initial = rand_init
            check = 1
        else:
            initial = results[-1][5]  # Use the last solution's final state as the next initial condition

        order_par_list,mean_r,start = solution(K, mass, init=initial)[1:4]
        omega_p=(4/np.pi)*np.sqrt(K*mean_r/(m))
        omega_max=find_omega_max(start,omega_list)
        results.append([K, mean_r,omega_max,omega_p, order_par_list,start])

    k_rev = K_values[::-1]
    for K in tqdm(k_rev):
        if check==1:
            initial = results[-1][5]  # Use the last solution's final state as the next initial condition
            check = 0
        else:
            initial = rev_results[-1][5]
        order_par_list,mean_r,start = solution(K, mass, init=initial)[1:4]
        omega_d=K*mean_r
        omega_max2=find_omega_max(start,omega_list)
        rev_results.append([K, mean_r,omega_max2,omega_d, order_par_list,start])
    return results, rev_results

results = skibidi(1,10, deltaK=0.2 , mass=2)
#%%
def plot_results(results):
    res, rev = results
    k_f=[]
    k_b=[]
    r_f=[]
    r_b=[]
    omega_m_f=[]
    omega_m_b=[]
    omega_p_l=[]
    omega_d_l=[]
    for i in range(len(res)):
        k_for, r_for = res[i][0], res[i][1]
        omega_m_for, omega_p = res[i][2], res[i][3]
        k_rev, r_rev = rev[i][0], rev[i][1]
        omega_m_back, omega_d = rev[i][2], rev[i][3]
        k_f.append(k_for)
        k_b.append(k_rev)
        r_f.append(r_for)   
        r_b.append(r_rev)
        omega_m_f.append(omega_m_for)
        omega_m_b.append(omega_m_back)
        omega_p_l.append(omega_p)
        omega_d_l.append(omega_d)
        # plt.scatter(k_for, r_for, color='darkgreen', s=10)
        # plt.scatter(k_rev, r_rev, color='darkred', s=10)
        # plt.scatter(k_for, omega_m_for, color='blue', s=10)
        # plt.scatter(k_rev, omega_m_back, color='orange', s=10)
        # plt.scatter(k_for, omega_p, color='purple', s=10)
        # plt.scatter(k_rev, omega_d, color='brown', s=10)
    plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    plt.plot(k_f, r_f, marker='x', linestyle='--', color='darkgreen', label='Forward Order Parameter')
    plt.plot(k_b, r_b, marker='x', linestyle='--', color='darkred', label='Backward Order Parameter')
    plt.plot(k_f, omega_m_f, marker='^', linestyle='--', color='blue', label='Forward Max Omega')
    plt.plot(k_b, omega_m_b, marker='v', linestyle='--', color='orange', label='Backward Max Omega')
    plt.plot(k_f, omega_p_l, linestyle='-', color='purple', label='Omega_p')
    plt.plot(k_b, omega_d_l, linestyle='-', color='brown', label='Omega_d')
    plt.legend()
    plt.xlabel('Coupling Strength (K)')
    plt.grid(True)
    plt.ylim(0,3.5)
    plt.show()
plot_results(results)
# %%
