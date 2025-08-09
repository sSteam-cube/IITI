import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm 
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*Casting complex values to real discards the imaginary part.*")

def main(n,m,x0=0,t_fin=100, std=1): 

    t_span = [0, t_fin]
    t_eval = np.linspace(*t_span, 200)
    # Initial phases uniformly from [0, 2π)
    init = np.random.uniform(0, 2 * np.pi, n)

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

    omega_list = sample_lorentzian(n, x0=x0, gamma=std)

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
                initial = results[-1][3]  # Use the last solution's final state as the next initial condition

            order_par_list,mean_r,start = solution(K, mass, init=initial)[1:4]
            results.append([K, mean_r, order_par_list,start])

        k_rev = K_values[::-1]
        for K in tqdm(k_rev):
            if check==1:
                initial = results[-1][3]  # Use the last solution's final state as the next initial condition
                check = 0
            else:
                initial = rev_results[-1][3]
            order_par_list,mean_r,start = solution(K, mass, init=initial)[1:4]
            rev_results.append([K, mean_r, order_par_list,start])
            return results, rev_results