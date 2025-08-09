#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous
#%%
class my_distribution(rv_continuous):
    def _pdf(self, x):
        return 1 / (np.pi * (1 + x**2))
    # 1 / (2*np.pi) * (1 / ((x - 0)**2 + 1) + 1 / ((x + 0)**2 + 1))
    #np.exp(-0.5 * (x**2)) / np.sqrt(2 * np.pi)
    #1 / (2*np.pi) * (1 / ((x - 0)**2 + 1) + 1 / ((x + 0)**2 + 1))
    def _cdf(self, x):
        # Adding CDF for better numerical stability
        return 0.5 + np.arctan(x) / np.pi
    
    def _ppf(self, q):
        # Adding PPF (inverse CDF) for better sampling
        return np.tan(np.pi * (q - 0.5))

# Create an instance of the distribution
#kmrj = my_distribution(a=-10, b=10)  # Optional: define bounds
kmrj = my_distribution(a=-10,b=10)
sample = kmrj.rvs(size=200)

# Only for plotting, not truncation
trimmed = sample[(sample > -10) & (sample < 10)]

import matplotlib.pyplot as plt

plt.hist(trimmed, bins=100, density=True, alpha=0.7, color='skyblue',
         edgecolor='black', label='Sample')
plt.title("Unbounded Cauchy Distribution (Trimmed View)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
#%%
# Generate random samples
kmrj = my_distribution(a=-100,b=100)
sample = kmrj.rvs(loc=0,scale=1,size=1000)#optional: loc and scale parameters
# loc is the mean and scale is the standard deviation
# Generate 10 random samples from the distribution(size=10)
# Note: loc and scale are optional parameters, default is loc=0 and scale=1
# If you want to specify bounds, you can do so by passing a and b parameters when creating the instance
# For example: kmrj = my_distribution(a=-10, b=10)
# This will generate samples from the distribution with bounds -10 and 10



plt.hist(sample, bins=100, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Samples')
plt.show()
