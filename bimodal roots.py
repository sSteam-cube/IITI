import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
global delta
global omega_0
global K

K = 1
delta = 1
omega_0 = 2
def root_finder(f,interval,accur):
    roots=[]
    x=np.arange(interval[0], interval[1], accur)
    for i in range(1,len(x)-1):
        if f(x[i-1])*f(x[i+1]) <=0 :
            roots.append(x[i] )
        elif abs(f(x[i]))<= accur : # and abs(abs(f(x[i-1]))-abs(f(x[i])))>=accur and abs(abs(f(x[i+1]))-abs(f(x[i]))) >= accur :
            roots.append(x[i])
    refined_roots = []
    for r in roots:
        if not any(abs(r - existing) < accur for existing in refined_roots):
            refined_roots.append(r)
    for i in range(len(refined_roots)):
        refined_roots[i] = np.round(refined_roots[i], 3)
    return refined_roots

def stability(f,x0,h=0.00001):
    if (f(x0+h)-f(x0-h))/(2*h) <= 0:
        return 1
    return 0 


def f(r):
    if abs(r)==0:
        return 0
    return r*((K/4)*r*(1 - 4*(delta/K) - r**2 + ((1-r**2)/(K*(1 + r**2)))*np.sqrt((K**2) * (1 + r**2)**2 - 16*(omega_0**2))))

# print(root_finder(f, [0, 1], 0.0001))

progress_bar = tqdm(total=5, desc="Finding Roots", unit="step")

omega_list = [0.6]
for l in omega_list:
    omega_0=l
    k_list = np.arange(-0.2, 10, 0.1)
    r_stable = []
    r_unstable = []
    r_stable_k = []
    r_unstable_k = []
    for k in k_list:
        K = k
        temp = root_finder(f, [0, 1], 0.0001)
        for r in temp:
            if stability(f, r):
                r_stable.append(r)
                r_stable_k.append(k)
            else:
                r_unstable.append(r)
                r_unstable_k.append(k)
    # plt.figure(figsize=(10, 6))
    plt.scatter(r_stable_k, r_stable, color='blue', label='Stable Roots', s=1) 
    plt.scatter(r_unstable_k, r_unstable, color='red', label='Unstable Roots', s=1)
plt.xlabel('K')
plt.ylabel('r')
plt.title('Roots of the Function f(r) vs K')
# plt.axhline(0, color='green', linestyle='--', linewidth=0.8)
# plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.grid(True)
plt.legend()
plt.xlim(0, 10)
plt.ylim(-0.1, 1.1)
progress_bar.update(1)
plt.tight_layout()
plt.show()
progress_bar.close()