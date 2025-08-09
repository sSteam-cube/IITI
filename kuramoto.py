import matplotlib.pyplot as plt
import numpy as np

def r_stable(k1, k23):
    disc = (k1 + k23)**2 - 8*k23
    if disc < 0 or k23 == 0:
        return np.nan
    return np.sqrt((k23 - k1 + np.sqrt(disc)) / (2 * k23))

def r_unstable(k1, k23):
    disc = (k1 + k23)**2 - 8*k23
    if disc < 0 or k23 == 0:
        return np.nan
    return np.sqrt((k23 - k1 - np.sqrt(disc)) / (2 * k23))

k1_vals = np.arange(-1, 4, 0.01)
k23_vals = [0.1, 2, 5, 10]

plt.figure(figsize=(8, 6))
for k23 in k23_vals:
    y1 = [r_stable(k1, k23) for k1 in k1_vals]
    y2 = [r_unstable(k1, k23) for k1 in k1_vals]
    plt.plot(k1_vals, y1, label=f" k23={k23}")
    plt.plot(k1_vals, y2, '--')

plt.xlabel("k1")
plt.ylabel("r")
plt.title("Stable and Unstable Fixed Points vs k1")
plt.legend()
plt.grid(True)
plt.show()

# k23_vals = np.arange(0, 12, 0.03)
# k1_vals = [2.2,2,1.8,1,-0.5]

# # plt.figure(figsize=(8, 6))
# for k1 in k1_vals:
#     y1 = [r_stable(k1, k23) for k23 in k23_vals]
#     y2 = [r_unstable(k1, k23) for k23 in k23_vals]
#     plt.plot(k23_vals, y1, label=f" (k1={k1})")
#     plt.plot(k23_vals, y2, '--')

# plt.xlabel("k2+3")
# plt.ylabel("r")
# plt.title("Stable and Unstable Fixed Points vs 2+3")
# plt.legend()
# plt.grid(True)
# plt.show()
