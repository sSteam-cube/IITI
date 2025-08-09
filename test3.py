import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Define g(Ω): Replace this with your actual function
def g(omega):
    return np.exp(-omega**2)

# Constants
Omega_0 = 1.0
m = 1.0

# Precompute second integral (independent of r and K)
def second_integral():
    integrand = lambda omega: g(omega) / (m * omega)**3
    return m * quad(integrand, Omega_0, np.inf, limit=100)[0]

I2 = second_integral()

# Function to solve: F(r, K) = 0
def F(r, K):
    if K * r <= Omega_0:
        return np.inf  # Invalid region, skip solving
    theta_0 = np.arcsin(Omega_0 / (K * r))
    integrand = lambda theta: (np.cos(theta))**2 * g(K * r * np.sin(theta))
    I1 = quad(integrand, -theta_0, theta_0, limit=100)[0]
    return I1 - I2 - 1 / K

# K range and solving for corresponding r
K_vals = np.linspace(0.1, 5, 100)
r_vals = []

for K in K_vals:
    try:
        # Minimum r to satisfy Kr ≥ Ω0
        r_min = Omega_0 / K + 1e-6
        r_max = 10  # You may adjust this upper bound
        sol = root_scalar(F, args=(K,), bracket=[r_min, r_max], method='bisect')
        r_vals.append(sol.root if sol.converged else np.nan)
    except:
        r_vals.append(np.nan)

# Plot r vs K
plt.plot(K_vals, r_vals)
plt.xlabel('K')
plt.ylabel('r')
plt.title('r vs K')
plt.grid(True)
plt.show()
