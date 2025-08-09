import numpy as np
import matplotlib.pyplot as plt

def lorentzian_pdf(x, x0, gamma):
    return (gamma / np.pi) * (1 / ((x - x0)**2 + gamma**2) + 1 / ((x + x0)**2 + gamma**2))

def sample_lorentzian(n, x0=1.0, gamma=0.5, x_range=(-10, 10)):
    samples = []
    max_pdf = lorentzian_pdf(x0, x0, gamma) + lorentzian_pdf(-x0, x0, gamma)  # Estimate max for rejection

    while len(samples) < n:
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(0, max_pdf)
        if y < lorentzian_pdf(x, x0, gamma):
            samples.append(x)

    return np.array(samples)

# Example usage
n = 10000
samples = sample_lorentzian(n, x0=2.0, gamma=1.0)
# Plotting
x_vals = np.linspace(-10, 10, 1000)
pdf_vals = lorentzian_pdf(x_vals, x0=2.0, gamma=1.0)
plt.hist(samples, bins=100, density=True, alpha=0.5, label='Sampled')
plt.plot(x_vals, pdf_vals / np.trapz(pdf_vals, x_vals), 'r-', label='Target PDF (normalized)')
plt.legend()
plt.xlabel('x')
plt.ylabel('Probability density')
plt.title('Sampling from double-peaked Lorentzian')
plt.show()

