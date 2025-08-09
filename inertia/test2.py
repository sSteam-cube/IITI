import numpy as np
import matplotlib.pyplot as plt

# Define the equation
def curve(x):
    return 1.2732 * x - 0.3056 * x**3

# x range
x = np.linspace(0, 1.2, 300)
y = curve(x)

# Setup the plot
fig, ax = plt.subplots(figsize=(6, 5))

# Fill bistable region
ax.fill_between(x, y, 1, where=(y < 1), color='lightgray')

# Plot the curve
ax.plot(x, y, color='black', linewidth=1)

# Dashed horizontal line at y = 1
ax.axhline(y=1, color='black', linestyle='--', linewidth=1)

# Axes limits
ax.set_xlim(0, 2)
ax.set_ylim(0, 1.5)

# Labeling axes
ax.set_xlabel(r'$\alpha$', fontsize=14)
ax.set_ylabel(r'$\beta$', fontsize=14)

# Add text labels
ax.text(0.2, 0.6, 'Bistable', fontsize=12)
ax.text(1.3, 0.5, 'Stable Fixed Point', fontsize=12)
ax.text(1.0, 1.2, 'Stable Limit Cycle', fontsize=12)

# Remove top and right spines for a cleaner look
ax.spines['top'].set_visible(1)
ax.spines['right'].set_visible(1)

plt.tight_layout()
plt.show()
# Add a box around the whole plot area