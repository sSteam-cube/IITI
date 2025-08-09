
#%%
import numpy as np
from scipy.integrate import solve_ivp   
import matplotlib.pyplot as plt
import pandas as pd

def stable_vals(k1, k23):
    disc = (k1 + k23)**2 - 8*k23
    if disc < 0 :
        return np.nan
    
    if k23 == 0:
        return np.sqrt(1-(2/k1)) if k1 >= 2 else np.nan
    return np.sqrt((k23 - k1 + np.sqrt(disc)) / (2 * k23))

def unstable_vals(k1, k23):
    disc = (k1 + k23)**2 - 8*k23
    if disc < 0:
        return np.nan
    if k23 == 0:
        return np.sqrt(1-(2/k1)) if k1 >= 0 else np.nan
    return np.sqrt((k23 - k1 - np.sqrt(disc)) / (2 * k23))

df = pd.read_csv('C:\\Users\\ACER.DESKTOP-AET6VDV\\Desktop\\IIT Indore\\plots\\simulations\\r vs k1(n=1000).csv')
df_k23 = pd.read_csv('C:\\Users\\ACER.DESKTOP-AET6VDV\\Desktop\\IIT Indore\\plots\\simulations\\r vs k23(n=1000).csv')

k1=[2.2,2,1.8,1,-0.5]
k23=[0,2,5,8,10]

k232=np.linspace(0, 12,1000)
k12=np.linspace(-1,4,1000)

#plot 1
colors = plt.cm.viridis(np.linspace(0, 1, len(k23)))
for idx, k23_val in enumerate(k23):
    color = colors[idx]
    # Drop NaN values for plotting
    mask_forward = ~df[f'K23={k23_val}'].isna()
    mask_backward = ~df[f'K23={k23_val}(backward)'].isna()
    plt.scatter(df['K1'][mask_forward], df[f'K23={k23_val}'][mask_forward], marker='o', color=color, label=f'K23={k23_val}')
    plt.scatter(df['K1'][mask_backward], df[f'K23={k23_val}(backward)'][mask_backward], marker='x', color=color)
    y1 = [stable_vals(i,k23_val) for i in k12]
    y2 = [unstable_vals(i,k23_val) for i in k12]
    plt.plot(k12, y1, color=color, alpha=0.7)
    plt.plot(k12, y2, color=color, linestyle='--', alpha=0.7)
plt.legend()
plt.xlabel("K1")    
plt.ylabel("r")
plt.grid(True)
# Draw a horizontal line at y=0 from x=-1 to x=2
plt.hlines(y=0, xmin=-2, xmax=2, color='green')
plt.xlim(-1.5, 4.1)
plt.ylim(-0.1, 1.1)
plt.show()
#%%
#plot 2
colors = plt.cm.viridis(np.linspace(0, 1, len(k1)))
for idx, k1_val in enumerate(k1):
    color = colors[idx]
    # Drop NaN values for plotting
    mask_forward = ~df_k23[f'K1={k1_val}'].isna()
    mask_backward = ~df_k23[f'K1={k1_val}(backward)'].isna()
    plt.scatter(df_k23['K23'][mask_forward], df_k23[f'K1={k1_val}'][mask_forward], marker='o', color=color, label=f'K1={k1_val}')
    plt.scatter(df_k23['K23'][mask_backward], df_k23[f'K1={k1_val}(backward)'][mask_backward], marker='x', color=color)
    y1 = [stable_vals(k1_val, i) for i in k232]
    y2 = [unstable_vals(k1_val, i) for i in k232]
    plt.plot(k232, y1, color=color, alpha=0.7)
    plt.plot(k232, y2, color=color, linestyle='--', alpha=0.7)

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
