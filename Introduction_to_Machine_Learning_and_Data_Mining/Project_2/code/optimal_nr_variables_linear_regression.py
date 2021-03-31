import numpy as np
import pandas as pd
import seaborn as sns
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import numpy as np
from matplotlib.pylab import (figure, plot, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from sklearn import model_selection
from toolbox_02450 import rlr_validate

def normalize(df):
    df2=(df-df.mean())/df.std()
    return df2

#reading the data
filename='C:/Users/klara/Documents/Studium/Master/IntroductiontoMachineLearning/project1/spambase.csv'
df = pd.read_csv(filename)
raw_data = df.to_numpy() 
cols = range(0, 58) 
X = raw_data[:, cols]
attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:,-1] # -1 takes the last column
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.array([classDict[cl] for cl in classLabels])
N, M = X.shape
C = len(classNames)

#find most correlated attributes in descending order
corrMatrix= df.corr()
spamCorrelation=corrMatrix.loc[' char_freq_!']
maxSpamCorr = spamCorrelation.abs().sort_values(ascending=False)

trainingErrors=np.zeros((57,10))
testingErrors=np.zeros((57,10))

#loop through number of attributes in descending correlation and give out test and 
#training error
for j in range (2,57):
    #initialize the essential values
    XforLinReg=df[maxSpamCorr[1:j].index.to_list()]
    XforLinReg=normalize(XforLinReg)
    yforLinReg=df[' char_freq_!']
    

    X1 = np.concatenate((np.ones((XforLinReg.shape[0],1)),XforLinReg),1)
    N, M = XforLinReg.shape
    attributeNames = [u'Offset']+maxSpamCorr[1:j].index.to_list()
    yforLinReg=np.array(yforLinReg)
    M = M+1
    
    ## Crossvalidation
    # Create crossvalidation partition for evaluation
    K = 10
    CV = model_selection.KFold(K, shuffle=True)
    #CV = model_selection.KFold(K, shuffle=False)
    
    # no regularization parameter in this case
    lambdas = 0
    
    # Initialize variables
    #T = len(lambdas)
    Error_train = np.empty((K,1))
    Error_test = np.empty((K,1))
    Error_train_rlr = np.empty((K,1))
    Error_test_rlr = np.empty((K,1))
    Error_train_nofeatures = np.empty((K,1))
    Error_test_nofeatures = np.empty((K,1))
    w_rlr = np.empty((M,K))
    mu = np.empty((K, M-1))
    sigma = np.empty((K, M-1))
    w_noreg = np.empty((M,K))
    
    k=0
    for train_index, test_index in CV.split(X1,yforLinReg):
        
        # extract training and test set for current CV fold
        X_train = X1[train_index]
        y_train = yforLinReg[train_index]
        X_test = X1[test_index]
        y_test = yforLinReg[test_index]
        internal_cross_validation = 10    
        
        mu[k, :] = np.mean(X_train[:, 1:], 0)
        sigma[k, :] = np.std(X_train[:, 1:], 0)
        
        X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
        X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
        
        #fit the model and save the test and training error 
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        
        m = lm.LinearRegression().fit(X_train, y_train)
        Error_train = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
        Error_test = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]  
        trainingErrors[j-2][k]=Error_train
        testingErrors[j-2][k]=Error_test
        
        k+=1

#compute and print the mean test and training error
meanTrainingErrors=np.mean(trainingErrors, axis=1)
meanTestingErrors=np.mean(testingErrors, axis=1)
plt.figure(figsize=(11,7))
plt.plot(meanTrainingErrors[:-2], color='blue', label='Training Error')
plt.plot(meanTestingErrors[:-2], color='red', label='Testing Error')
plt.title('Errors for different number of included variables')
plt.legend()
plt.show()
