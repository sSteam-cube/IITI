import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def r_dot(k1, k23):
    return [-k23/2, 0, -k1/2 + k23/2, 0, -1+(k1/2), 0]

def rdot(r, k1, k23):
    return r*(-1+(k1/2)) + r**3 * (-k1/2 + k23/2) + r**5 * (-k23/2)

def central_difference(f, x, h, k1, k23):
    return (f(x + h, k1, k23) - f(x - h, k1, k23)) / (2 * h)

def stability_check(x0, k1, k23):
    return 1 if central_difference(rdot, x0, 0.001, k1, k23) <= 0 else 0

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

k1_values =   [0, 2, 5, 8, 10] #[-0.5,1,1.8,2.2]

color_map = plt.get_cmap('tab10')  # 10 distinct categorical colors

for idx, i in enumerate(k1_values):
    k23 = np.linspace(-1, 4, 100)
    roots = [np.roots(r_dot(j,i)) for j in k23]
    stable = []
    unstable = []

    for l in range(len(roots)):
        stab, unstab = [], []
        for m in range(len(roots[l])):
            if np.isreal(roots[l][m]) and np.real(roots[l][m]) >= 0:
                r_val = np.real_if_close(roots[l][m])
                if stability_check(r_val, k23[l],i):
                    stab.append(r_val)
                else:
                    unstab.append(r_val)
        stable.append((k23[l], stab))
        unstable.append((k23[l], unstab))

    xstab, ystab, xunstab, yunstab = [], [], [], []
    for p in range(len(stable)):
        for j in range(len(stable[p][1])):
            xstab.append(stable[p][0])
            ystab.append(stable[p][1][j])
    for p in range(len(unstable)):
        for j in range(len(unstable[p][1])):
            xunstab.append(unstable[p][0])
            yunstab.append(unstable[p][1][j])

    base_color = color_map(idx % 10)
    faded_color = mcolors.to_rgba(base_color, alpha=0.3)

    plt.scatter(xstab, ystab, marker='.', color=base_color, s=10)
    plt.scatter(xunstab, yunstab, marker='d', color=base_color, s=10)

    y1 = [r_stable(q,i) for q in k23]
    y2 = [r_unstable(q,i) for q in k23]
    plt.plot(k23, y1, label=f"K23={i} (stable)", color=base_color)
    plt.plot(k23, y2, '--', color=base_color)

# Final plot tweaks
# plt.title('Bifurcation Diagram')
plt.xlabel('K1')
plt.ylabel('r')
plt.grid(True)
plt.legend()
plt.ylim(-0.5,1)
# plt.tight_layout()
plt.show()
