import numpy as np
import pandas as pd
from matplotlib.pyplot import *
from scipy.linalg import svd


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

# Data attributes to be plotted
i = 0
j = 1

##
# Make a simple plot of the i'th attribute against the j'th attribute
# Notice that X is of matrix type (but it will also work with a numpy array)
X = np.array(X) #Try to uncomment this line
plot(X[:, i], X[:, j], 'o')

# %%
# Make another more fancy plot that includes legend, class labels, 
# attribute names, and a title.
f = figure()
title('SpamBase data')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)

legend(classNames)
xlabel(attributeNames[i])
ylabel(attributeNames[j])

# Output result to screen
show()
# %%
# Subtract mean value from data
Y = (X - np.ones((N,1))*X.mean(axis=0))/(np.ones((N,1))*X.std(axis=0))

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

#threshold = 0.9

# Plot variance explained
figure()
plot(range(1,len(rho)+1),rho,'x-')
plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
#plot([1,len(rho)],[threshold, threshold],'k--')
plot([1,len(rho)],[0.90, 0.90], 'r--')
title('Variance explained by principal components');
xlabel('Principal component');
ylabel('Variance explained');
legend(['Individual','Cumulative','Threshold'])
grid()
show()


# Project the centered and standardized data onto principal component space
V=V.T
Z = Y @ V
# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('Spambase data: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()





