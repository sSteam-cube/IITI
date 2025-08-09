#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*Casting complex values to real discards the imaginary part.*")
#%%

n = 1000     # Number of oscillators
K1 =0     # Coupling strength
x0 = 2.5        # Lorentzian center

t_fin=50    # Lorentzian width
t_span = [0, t_fin]
t_eval = np.linspace(*t_span, 500)


# Initial phases uniformly from [0, 2π)
init = np.random.uniform(0, 2 * np.pi, n)

# Sample natural frequencies from the Lorentzian distribution
# Using inverse transform sampling
def lorentzian_pdf(x, x0, gamma):
    return (gamma / (2*np.pi)) * (1 / ((x - x0)**2 + gamma**2) + 1 / ((x + x0)**2 + gamma**2))

def sample_lorentzian(n, x0=1.0, gamma=0.5, x_range=(-10, 10)):
    samples = []
    max_pdf = lorentzian_pdf(x0, x0, gamma) + lorentzian_pdf(-x0, x0, gamma)  # Estimate max for rejection

    while len(samples) < n:
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(0, max_pdf)
        if y < lorentzian_pdf(x, x0, gamma):
            samples.append(x)

    return np.array(samples)




def z1(theta_array):
    r = np.mean(np.exp(1j * theta_array))
    return np.abs(r)


def H(theta_array):
    z1 = np.mean(np.exp(1j * theta_array))
    return K1*z1   

def theta_dot(t,theta_list):
    out = []
    h=H(theta_list)
    for i in range(n):
        omega_i = omega_list[i]
        theta_i = theta_list[i]
        temp = omega_i + np.complex(h*np.exp(-1j*theta_i)) #omega_i + (1/(2*1j)) * (h*np.exp(-1j*theta_i) - np.conj(h)*np.exp(1j*theta_i))
        out.append(temp)
    # progress_bar.update(1)    

    return out


def solution(k1):
    global K1
    K1= k1
    sol = solve_ivp(theta_dot, t_span, init, t_eval=t_eval,method='RK45')
    order_par = [z1(sol.y.T[i]) for i in range(len(sol.y.T))]
    mean_r = np.mean(order_par[len(order_par)//4:])
    sol2 = solve_ivp(theta_dot, t_span, [0]*n, t_eval=t_eval,method='RK45')
    order_par2 = [z1(sol2.y.T[i]) for i in range(len(sol2.y.T))]
    mean_r2 = np.mean(order_par2[len(order_par2)//4:])

    plt.plot(t_eval, order_par, label=f'K1={k1}')
    plt.plot(t_eval, order_par2, linestyle='--', label=f'K1={k1}')
    plt.xlabel("Time")
    plt.ylabel("Order Parameter ⟨r⟩")
    plt.title("Order Parameter Over Time")
    plt.grid(True)
    # print(mean_r)
    return mean_r,mean_r2,order_par,order_par2


k_list = [4]
gamma = 1 
x_0_list = [1.2]  # List of omega_0 values
progress_bar = tqdm(total=len(k_list)*len(x_0_list), desc="Calculating Order Parameters", unit="step")
 # List of omega_0 values
for j in x_0_list:

    x0 = j
    omega_list = sample_lorentzian(n, x0=x0, gamma=1.0)
    sd_for=[]
    sd_back = []
    for i in k_list:
        sd1=solution(i)[2]#,solution(i)[3]
        sd_for.append(np.std(sd1[len(sd1)//2:]))
        # sd_back.append(np.std(sd2[len(sd2)//2:]))
        progress_bar.update(1)
    # plt.figure(figsize=(10, 6))
    # plt.plot(k_list, sd_for, label=f'Forward Order Parameter Std Dev for omega_0={x0}')
# plt.plot(k_list, sd_back, label='Backward Order Parameter Std Dev', color='red')
# plt.xlabel('Coupling Strength K1')
# plt.ylabel('Standard Deviation of Order Parameter')
# plt.title('Standard Deviation of Order Parameter vs Coupling Strength')
plt.legend()
plt.show()
progress_bar.close()
# %%
