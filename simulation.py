#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*Casting complex values to real discards the imaginary part.*")
#%%

n = 2000     # Number of oscillators
K1 =0     # Coupling strength
K2= 0
K3 = 0

x0 = 0        # Lorentzian center
gamma = 1 
t_fin=200    # Lorentzian width
t_span = [0, t_fin]
t_eval = np.linspace(*t_span, 400)


# Initial phases uniformly from [0, 2π)
init = np.random.uniform(0, 2 * np.pi, n)

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

def z1(theta_array):
    r = np.mean(np.exp(1j * theta_array))
    return np.abs(r)

def stable_vals(k1, k23):
    disc = (k1 + k23)**2 - 8*k23
    if disc < 0 or k23 == 0:
        return np.nan
    return np.sqrt((k23 - k1 + np.sqrt(disc)) / (2 * k23))

def unstable_vals(k1, k23):
    disc = (k1 + k23)**2 - 8*k23
    if disc < 0 or k23 == 0:
        return np.nan
    return np.sqrt((k23 - k1 - np.sqrt(disc)) / (2 * k23))


def z2(theta_array):
    r = np.mean(np.exp(2j * theta_array))
    return np.abs(r)

def H(theta_array):
    z1 = np.mean(np.exp(1j * theta_array))
    z2 = np.mean(np.exp(2j * theta_array))
    return K1*z1 + K2*z2*np.conj(z1) + K3*z1**2*np.conj(z1)  

def theta_dot(t,theta_list):
    out = []
    h=H(theta_list)
    for i in range(n):
        omega_i = omega_list[i]
        theta_i = theta_list[i]
        temp = omega_i + (1/(2*1j)) * (h*np.exp(-1j*theta_i) - np.conj(h)*np.exp(1j*theta_i))
        out.append(temp)
    # progress_bar.update(1)    

    return out


def solution(k1, k23):
    global K1
    global K2
    global K3
    K1= k1
    K2= k23/2
    K3= k23/2
    sol = solve_ivp(theta_dot, t_span, init, t_eval=t_eval,method='RK45')
    order_par = [z1(sol.y.T[i]) for i in range(len(sol.y.T))]
    mean_r = np.mean(order_par[len(order_par)//4:])
    sol2 = solve_ivp(theta_dot, t_span, [0]*n, t_eval=t_eval,method='RK45')
    order_par2 = [z1(sol2.y.T[i]) for i in range(len(sol2.y.T))]
    mean_r2 = np.mean(order_par2[len(order_par2)//4:])
    plt.plot(t_eval, order_par)
    # plt.xlabel("Time")
    # plt.ylabel("Order Parameter ⟨r⟩")
    # plt.title("Order Parameter Over Time")
    # plt.grid(True)
    plt.show()
    # print(mean_r)
    return mean_r,mean_r2

# progress_bar = tqdm(total=n, desc="Solving", unit="step")


# progress_bar.close()
k1=[2]#[2.2,2,1.8,1,-0.5]
k23= [0]#np.linspace(0, 12,5)
  
df = pd.DataFrame({'K23': k23})

for i in tqdm(k1, desc="Calculating Mean Order Parameter for K1"):
    mean_r_vals = []
    mean_r2_vals = []
    df[f'K1={i}'] = np.nan  # Initialize with NaN
    df[f'K1={i}(backward)'] = np.nan  # Initialize with NaN
    for k23_val in tqdm(k23, desc="Calculating Mean Order Parameter for K23"):
        mean_r, mean_r2 = solution(i,k23_val)
        df.loc[df['K23'] == k23_val, f'K1={i}'] = mean_r
        df.loc[df['K23'] == k23_val, f'K1={i}(backward)'] = mean_r2
        mean_r_vals.append(mean_r)
        mean_r2_vals.append(mean_r2)
# df.to_csv("C:\\Users\\ACER.DESKTOP-AET6VDV\\Desktop\\IIT Indore\\plots\\bruh.csv", index=False)

 

#%%
# Plotting the results
# plt.figure(figsize=(10, 6)) 
k232=np.linspace(0, 12,1000)
# Plot mean_r_vals for each k23
colors = plt.cm.viridis(np.linspace(0, 1, len(k1)))
for idx, k1_val in enumerate(k1):
    color = colors[idx]
    # Drop NaN values for plotting
    mask_forward = ~df[f'K1={k1_val}'].isna()
    mask_backward = ~df[f'K1={k1_val}(backward)'].isna()
    plt.scatter(df['K23'][mask_forward], df[f'K1={k1_val}'][mask_forward], marker='o', color=color, label=f'K1={k1_val}')
    plt.scatter(df['K23'][mask_backward], df[f'K1={k1_val}(backward)'][mask_backward], marker='x', color=color)
    y1 = [stable_vals(k1_val, i) for i in k232]
    y2 = [unstable_vals(k1_val, i) for i in k232]
    plt.plot(k232, y1, color=color, alpha=0.7)
    plt.plot(k232, y2, color=color, linestyle='--', alpha=0.7)




# plt.scatter(k1, mean_r_vals, marker='o')
# Save mean_r_vals as CSV on Desktop
# desktop_path = os.path.join(os.path.expanduser("C:\Users\ACER.DESKTOP-AET6VDV\Desktop"), "IIT Indore", "mean_r_vals.csv")
# df = pd.DataFrame({'K1': k1, 'mean_r_vals': mean_r_vals})
# df.to_csv("C:\\Users\\ACER.DESKTOP-AET6VDV\\Desktop\\IIT Indore\\plots\\k23=8 n=1000.csv", index=False)

# plt.scatter(k1, mean_r2_vals, marker='x',  color='orange')

plt.legend()
plt.xlabel("K23")    
plt.ylabel("r")
plt.grid(True)
# Draw a horizontal line at y=0 from x=-1 to x=2
plt.hlines(y=0, xmin=-2, xmax=2, color='green')
plt.xlim(-1.5, 13)
plt.ylim(-0.1, 1.1)
plt.show()


# %%
