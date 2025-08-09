import numpy as np
import matplotlib.pyplot as plt
#finding the roots and analysing the stability

global k1
global k23
k1=2
k23=8

def f(r):
    return r*(-1+(k1/2)) + r**3 * (-k1/2 + k23/2) + r**5 * (-k23/2)

def stable_vals(k1, k23):
    disc = (k1 + k23)**2 - 8*k23
    if disc < 0 or k23 == 0:
        return np.nan
    return np.sqrt((k23 - k1 + np.sqrt(disc)) / (2 * k23))

def unstable_vals(k1, k23):
    disc = (k1 + k23)**2 - 8*k23
    if disc < 0 or k23 == 0:
        return np.nan
    return np.sqrt((k23 - k1 - np.sqrt(disc)) / (2 * k23))

def root_finder(f,interval,accur):
    roots=[]
    x=np.arange(interval[0], interval[1], accur)
    for i in range(1,len(x)-1):
        if f(x[i-1])*f(x[i+1]) <=0 :
            roots.append(x[i] )
        elif abs(f(x[i]))<= accur and abs(f(x[i-1]))>=abs(f(x[i])) and abs(f(x[i+1]))>=abs(f(x[i])) :
            roots.append(x[i])
    refined_roots = []
    for r in roots:
        if not any(abs(r - existing) < accur for existing in refined_roots):
            refined_roots.append(r)
    for i in range(len(refined_roots)):
        refined_roots[i] = np.round(refined_roots[i], 3)
    return refined_roots

def stability(f,x0,h=0.0001):
    if (f(x0+h)-f(x0-h))/(2*h) <= 0:
        return 1
    return 0 

#plot 1
color_map = plt.get_cmap('tab10')
# k2list=[0,2,5,8,10]
# for i in k2list:
#     r_stable=[]
#     r_unstable=[]
#     k_stable=[]
#     k_unstable=[]
#     for j in np.linspace(-1,4,50):
#         k1,k23 = j,i
#         temp = root_finder(f,[0,1],0.0001)
#         for l in range(len(temp)):
#             if stability(f,temp[l]):
#                 r_stable.append(temp[l])
#                 k_stable.append(j)
#             else:
#                 r_unstable.append(temp[l])
#                 k_unstable.append(j)
#     plt.scatter(k_stable,r_stable,marker='o',c='g',s=5)
#     plt.scatter(k_unstable,r_unstable,marker='o',c='r',s=5)
#     tempk=np.linspace(-1,4,10000)
#     y1 = [stable_vals(j,i) for j in tempk]
#     y2 = [unstable_vals(j,i) for j in tempk]
#     base_color = color_map(k2list.index(i))
#     plt.plot(tempk, y1, label=f" k23={i}", c=base_color)
#     plt.plot(tempk, y2, '--', c=base_color)


#plot2

# klist=[2.2,2,1.8,1,-0.5]
# for i in klist:
#     r_stable=[]
#     r_unstable=[]
#     k_stable=[]
#     k_unstable=[]
#     for j in np.linspace(0,12,50):
#         k1,k23 = i,j
#         temp = root_finder(f,[0,1],0.001)
#         for l in range(len(temp)):
#             if stability(f,temp[l]):
#                 r_stable.append(temp[l])
#                 k_stable.append(j)
#             else:
#                 r_unstable.append(temp[l])
#                 k_unstable.append(j)
#     plt.scatter(k_stable,r_stable,marker='o',c='g',s=10)
#     plt.scatter(k_unstable,r_unstable,marker='d',c='r',s=10)
#     tempk=np.linspace(0,12,10000)
#     y1 = [stable_vals(i,j) for j in tempk]
#     y2 = [unstable_vals(i,j) for j in tempk]
#     base_color = color_map(klist.index(i))
#     plt.plot(tempk, y1, label=f" k1={i}", c=base_color)
#     plt.plot(tempk, y2, '--', c=base_color)
    
# plt.xlabel('K23')
# plt.ylabel('r')
# plt.grid(True)
# plt.legend()
# plt.ylim(-0.5,1)
# # To plot branches correctly, sort the points for each branch before plotting
# plt.show()

# Optional: If you want to plot the branches as lines instead of scattered points,
# you can group and sort the points by k_stable/k_unstable and r_stable/r_unstable.
# Here's an example for the last loop (plot2):
klist=[2.2,-0.5]
for i in klist:
    r_stable=[]
    r_unstable=[]
    k_stable=[]
    k_unstable=[]
    for j in np.linspace(0,12,50):
        k1,k23 = i,j
        temp = root_finder(f,[0,1],0.001)
        for l in range(len(temp)):
            if stability(f,temp[l]):
                r_stable.append(temp[l])
                k_stable.append(j)
            else:
                r_unstable.append(temp[l])
                k_unstable.append(j)
    # Sort the stable and unstable points by k value for proper line plotting
    stable_points = sorted(zip(k_stable, r_stable))
    unstable_points = sorted(zip(k_unstable, r_unstable))
    if stable_points:
        k_stable_sorted, r_stable_sorted = zip(*stable_points)
        plt.plot(k_stable_sorted, r_stable_sorted, 'g-', linewidth=1)
    if unstable_points:
        k_unstable_sorted, r_unstable_sorted = zip(*unstable_points)
        plt.plot(k_unstable_sorted, r_unstable_sorted, 'r--', linewidth=1)
    tempk=np.linspace(0,12,10000)
    y1 = [stable_vals(i,j) for j in tempk]
    y2 = [unstable_vals(i,j) for j in tempk]
    base_color = color_map(klist.index(i))
    plt.plot(tempk, y1, label=f" k1={i}", c=base_color)
    plt.plot(tempk, y2, '--', c=base_color)
plt.xlabel('K23')
plt.ylabel('r')
plt.grid(True)
plt.legend()
plt.ylim(-0.5,1)
plt.show()