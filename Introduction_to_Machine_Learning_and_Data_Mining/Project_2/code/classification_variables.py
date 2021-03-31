import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#Loading and preprocessing data using this course's structure.
filename = 'spambase.csv'
df = pd.read_csv(filename)
raw_data = df.to_numpy() 
cols = range(0, 57) 
X = raw_data[:, cols]
attributeNames = np.asarray(df.columns[cols])
y = raw_data[:,-1]
classTypes = np.unique(y)
N, M = X.shape
C = len(classTypes)

#Order the attributes from highest correlation with Spam to lowest:
corr = np.corrcoef(X.T, y)
corr = corr**2
indeces = np.argsort(corr)[-1,:-1][::-1]

#Take the n most correlated attributes for X
n = 10
corr = np.corrcoef(X.T, y)
corr = corr**2
indeces = np.argsort(corr)[-1,:-1][::-1]
indeces = indeces[:n]
X = X[:, indeces]

'''
#Take n first attributes for X
n = 10
X = X[:, :n]
'''

#Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)