from matplotlib.pyplot import (figure, plot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show)
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from numpy import cov

from classification_variables import *

# Maximum number of nearest neighbors
L = 25
N, M = X.shape
CV = model_selection.KFold(n_splits=20)
errors = np.zeros((20,L))
i=0
for train_index, test_index in CV.split(X):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,20))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    
    
    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l, p=2,
                                            metric='cosine');
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors[i,l-1] = np.sum(y_est!=y_test)/len(y_test)

    i+=1


figure()
plot(100*sum(errors,0)/20)
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
show()