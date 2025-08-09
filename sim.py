import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

n=100
K=1

def order_param(sol,i):
    lst = sol[i-1]
    r = (np.sum([np.exp(1j*k) for k in lst]))/len(lst)
    return r

def lorentzian(x, x0, gamma):
    return (1 / np.pi) * (gamma / ((x - x0)**2 + gamma**2))

# Parameters
x0 = 0       # Mean (location of the peak)
gamma = 1    # Width (HWHM)

# Domain
x = np.linspace(-10, 10, n)

omega_list=lorentzian(x, x0, gamma)

def theta_dot(t,theta_list):
    out=[]
    for i in range(0,len(theta_list)):
        omega_i = omega_list[i]
        theta_i = theta_list[i]
        temp = np.sum([np.sin(j-theta_i) for j in theta_list])
        temp2 = omega_i + (K/n)*temp
        out.append(temp2)
    return out
init = np.random.uniform(0, 2 * math.pi, n)

t_span = [0, 100]
t_eval  = np.linspace(0, 100, 1000)

sol = solve_ivp(theta_dot, t_span, init, t_eval = t_eval )
y_values = sol.y[1]
x_values = sol.y[0]

order_par=[]

for i in (0,len(sol.y)):
    order_par.append(order_param(sol.y,i))

plt.plot(t_eval, order_par)
plt.show()



