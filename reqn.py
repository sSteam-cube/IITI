import numpy as np
import matplotlib.pyplot as plt
from sympy import diff

# K1=1
# K23=8
def r_dot(k1,k23):
    return [-k23/2, 0, -k1/2 + k23/2, 0, -1+(k1/2), 0]

def rdot (r, k1,k23):
    return r*(-1+(k1/2)) + r**3 * (-k1/2 + k23/2) + r**5 * (-k23/2)

def central_difference(f, x, h ,k1,k23):
    return (f(x + h, k1, k23) - f(x - h, k1, k23)) / (2 * h)

def stability_check(x0, k1, k23):
    if central_difference(rdot, x0, 0.001, k1, k23) <=0:
        return 1
    return 0


for i in [2,8]:
    k23=np.linspace(0,4,100)
    roots = [np.roots(r_dot(j,i)) for j in k23]
    stable = []
    unstable = []

    for l in range(len(roots)):
        stab=[]
        unstab=[]
        for m in range(len(roots[l])):
            if np.isreal(roots[l][m]) and np.real(roots[l][m]) >= 0:
                if stability_check(roots[l][m], k23[l], i):
                    stab.append(roots[l][m])
                else:
                    unstab.append(roots[l][m])

        stable.append((k23[l], stab))
        unstable.append((k23[l], unstab))
    xstab=[]
    xunstab=[]
    ystab=[]
    yunstab=[]
    for p in range(len(stable)):
        for j in range(len(stable[p][1])):
            xstab.append(stable[p][0])
            ystab.append(stable[p][1][j])


    for p in range(len(unstable)):
        for j in range(len(unstable[p][1])):
            xunstab.append(unstable[p][0])
            yunstab.append(unstable[p][1][j])
    # plt.plot(xstab,ystab, label=f'K23={i}')
    # plt.plot(xunstab,yunstab)
    plt.scatter(xstab,ystab,marker='.',c='g',s=5, label=f'K23={i}')
    plt.scatter(xunstab,yunstab,marker='.',c='r',s=5)
    plt.title('Bifurcation Diagram')
    plt.xlabel('K1')
    plt.ylabel('R')
    plt.legend()
    plt.grid(True)
                    
plt.show()
# print(roots)



