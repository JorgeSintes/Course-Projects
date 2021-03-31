from mpl_toolkits.mplot3d import axes3d 
import matplotlib.pyplot as plt
from PCA1 import *
import seaborn as sns
from scipy.stats import norm

def originalnames(NaMes):
    names=[]
    for i in NaMes:
        a=i.split('_')
        names.append(a[-1])
    return names

i = 0
j = 1
k = 2

#plot original dat
f = plt.figure(figsize=(8,4))
ax = plt.axes(projection ="3d")
plt.title('SpamBase data')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    ax.scatter3D(X[class_mask,i], X[class_mask,j], X[class_mask,k], 'o',alpha=.9)
    ax.view_init(azim=110)

plt.legend(classNames)
plt.xlabel(attributeNames[i])
plt.ylabel(attributeNames[j])
ax.set_zlabel(attributeNames[k])

# Output result to screen
plt.show()

# Plot PCA of the data
fig = plt.figure(figsize=(8,4))
ax = plt.axes(projection ="3d")
plt.title('SpamBase: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    ax.scatter3D(Z[class_mask,i], Z[class_mask,j],Z[class_mask,k], '.', alpha=.9)
    ax.view_init(azim=110)
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
ax.set_zlabel('PC{0}'.format(k+1))

# Output result to screen
plt.show()

mu, std = norm.fit(X[:,15])
plt.hist(X[:,0], bins=25, density=True, alpha=0.9, color='r')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Word's 'free' frequency distribution: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.show()




