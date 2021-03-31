import numpy as np
import pandas as pd
from matplotlib.pyplot import *
from scipy.linalg import svd
import matplotlib.pyplot as plt
def originalnames(NaMes):
    names=[]
    for i in NaMes:
        a=i.split('_')
        names.append(a[-1])
    return names


# Load the Iris csv data using the Pandas library
filename = 'C:/Users/george/Desktop/MLrep1/spambase.csv'
df = pd.read_csv(filename)

# Convert pandas framework to numpy array
raw_data = df.to_numpy()

# Collect the headers of the columns that we will deal with
attributeNames = np.asarray(df.columns[:])

#dataset excluding spam/not spam column
X=raw_data[:,0:57]
classNames = np.array(['not_spam','spam'])
y = raw_data[:,-1]

classDict = dict(zip(classNames, range(2)))

# Number of classes
C = len(classNames)
N = len(y)
M = len(attributeNames)
Y = (X - np.ones((N,1))*X.mean(axis=0))/(np.ones((N,1))*X.std(axis=0))

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)
V=V.T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
N,M = X.shape
r = np.arange(1,M+1)
fig=plt.figure(figsize=(15,4))
for i in pcs:
    attr=r+i*bw    
    plt.bar(attr[:30], V[:30,i], width=bw)
name=originalnames(attributeNames)
plt.xticks(attr[:30], name[:30],rotation='70')
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Spambase: PCA Component Coefficients')
plt.show()

fig=plt.figure(figsize=(15,4))
for i in pcs:
    attr=r+i*bw    
    plt.bar(attr[30:], V[30:,i], width=bw)
name=originalnames(attributeNames)
for i in range(4):
    name.pop(-1)
name.append("CRL_Aver")
name.append("CRL_Longest")
name.append("CRL_Total")
plt.xticks(attr[30:], name[30:],rotation='75')
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Spambase: PCA Component Coefficients')
plt.show()