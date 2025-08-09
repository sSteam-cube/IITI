#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm 
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*Casting complex values to real discards the imaginary part.*")
#%%
n = 500     # Number of oscillators
K1 =0     # Coupling strength
x0 = 0       # Lorentzian center
m=6
t_fin=100    # Lorentzian width
t_span = [0, t_fin]
t_eval = np.linspace(*t_span, 200)
# Initial phases uniformly from [0, 2π)
init = np.random.uniform(0, 2 * np.pi, n)

# Sample natural frequencies from the Lorentzian distribution
# Using inverse transform sampling
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

def a(k,r,m):
    return 1/np.sqrt(k*r*m)

def b(k,r,omega):
    return omega/(k*r)


def H(theta_array):
    z1 = np.mean(np.exp(1j * theta_array))
    return K1*z1   

def diff(t,theta_thetadot_list):
    thetaddot_out=[]
    theta_list= theta_thetadot_list[:n]
    thetadot_list= theta_thetadot_list[n:]
    r=z1(theta_list)
    alpha=a(K1, r, m)  
    for i in range(n):
        omega_i = omega_list[i]
        theta_i = theta_list[i]
        thetadot_i = thetadot_list[i]
        beta = b(K1, r, omega_i)
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
    # sol2 = solve_ivp(diff, t_span, np.concatenate([[0]*n, omega_list]), t_eval=t_eval,method='RK45')
    # order_par2 = [z1(sol2.y.T[i][:n]) for i in range(len(sol2.y.T))]
    # mean_r2 = np.mean(order_par2[len(order_par2)//2:])

    # plt.plot(t_eval, order_par, label=f'K1={k1}')
    # # plt.plot(t_eval, order_par2, linestyle='--', label=f'K1={k1}')
    
    # plt.xlabel("Time")
    # plt.ylabel("Order Parameter ⟨r⟩")
    # plt.title("Order Parameter Over Time")
    # plt.grid(True)
    # plt.show()
    # print(mean_r)
    next_init= np.concatenate([sol.y.T[-1][:n], sol.y.T[-1][n:]])
    return [t_eval,order_par,mean_r,next_init]

rand_init = np.concatenate([np.random.uniform(0, 2 * np.pi, n), omega_list])
# solution(50,2,init=rand_init)

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
    rev_results2=[]
    k_rev2 = k_rev[5*len(results)//7:]
    for K in tqdm(k_rev2):
        if check==0:
            initial = results[2*len(results)//7][3]  # Use the last solution's final state as the next initial condition
            check = 1
        else:
            initial = rev_results2[-1][3]
        order_par_list,mean_r,start = solution(K, mass, init=initial)[1:4]
        rev_results2.append([K, mean_r, order_par_list,start])
    return results, rev_results,rev_results2




def plot_results(results):
    res, rev, rev2 = results

    # Plot and collect data
    data = []
    for i in range(len(res)):
        k_for, r_for = res[i][0], res[i][1]
        k_rev, r_rev = rev[i][0], rev[i][1]
        data.append({'K_for': k_for, 'r_for': r_for, 'K_rev': k_rev, 'r_rev': r_rev})

        plt.scatter(k_for, r_for, color='darkgreen', s=10, marker='^')
        plt.scatter(k_rev, r_rev, color='darkred', s=10, marker='v')

    for i in range(len(rev2)):
        k_rev2, r_rev2 = rev2[i][0], rev2[i][1]
        plt.scatter(k_rev2, r_rev2, color='darkblue', s=10, marker='v')
        # Bring scatter points to the front
        for coll in plt.gca().collections:
            coll.set_zorder(10)
    # Create and save DataFrame
    df = pd.DataFrame.from_records(data)
    # df.to_excel("C:\\Users\\ACER.DESKTOP-AET6VDV\\Desktop\\IIT Indore\\order_parameter_results(m=6).xlsx", index=False)
    plt.tight_layout()
    plt.xlabel("Coupling Strength K")
    plt.ylabel("Mean Order Parameter ⟨r⟩")
    plt.title(f"Mean Order Parameter vs Coupling Strength(m={m},n=500)")
    plt.grid(True)
    # plt.legend() 
    # plt.show()
    # print(len(rev), len(res))


 # %%
d=1
m=2
x=np.linspace(1,20,100)
y=np.linspace(1,0.01,100)

def g(x):
    return (1 /np.sqrt(2*np.pi*d**2))*(np.exp(-(x**2)/(2*d**2)))

def protocol(x,y):
    def f1(u):
        return (np.cos(u))**2 * g(x*y*np.sin(u))
    def f2(u): 
        return g(u)/u**2
    def h2(u):
        return g(u)*b*(u/a)*(-b*(u/a) + np.sqrt((b**2 * u**2) /a**2 - a**2 / (a**4 + b**2 * u**2)))
    a=1/np.sqrt(x*y*m)
    b=1/(x*y)
    omega_p=(4/np.pi)*np.sqrt(x*y/m)
    theta_p=np.arcsin(omega_p/(x*y))
    int1= (-x*y/m)*integrate.quad(f2, x*y, np.inf)[0] + x*y*integrate.quad(f1,-np.pi/2, np.pi/2)[0]
    int2=  x*y*integrate.quad(f1,-theta_p, theta_p)[0] + (-x*y/m)*integrate.quad(f2, omega_p, np.inf)[0]
    int1_better = x*y*integrate.quad(f1,-np.pi/2, np.pi/2)[0] + 2*integrate.quad(h2, x*y, np.inf)[0]
    int2_better= x*y*integrate.quad(f1,-theta_p, theta_p)[0] + 2*integrate.quad(h2, omega_p, np.inf)[0]
    return y-int1, y-int2, y-int1_better, y-int2_better

X, Y = np.meshgrid(y,x, indexing='ij')
coords = np.stack((X,Y), axis=-1) 
print(coords[2,2])

val1=np.zeros((len(x), len(y)))
val2=np.zeros((len(x), len(y)))
val1_better=np.zeros((len(x), len(y)))
val2_better=np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        val1[i,j],val2[i,j],val1_better[i,j],val2_better[i,j] = protocol(coords[i,j][1], coords[i,j][0])  

# print(val2[1,2])
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
#%%
results = skibidi(1,12, deltaK=0.2 , mass=2)
#%%
x1,y1= marching(val1, coords)
x2,y2= marching(val2, coords)
x1_better,y1_better= marching(val1_better, coords)
x2_better,y2_better= marching(val2_better, coords)

plt.plot(x1, y1,linestyle='--', color="black")
plt.plot(x2, y2,linestyle='--', color="black")
plt.plot(x1_better, y1_better, color="orange")
plt.plot(x2_better, y2_better, color="orange")
plot_results(results)
plt.show()


# %%
