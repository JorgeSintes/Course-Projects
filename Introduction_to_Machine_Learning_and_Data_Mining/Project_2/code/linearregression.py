import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import preprocessing
import numpy as np
from matplotlib.pylab import (figure, plot, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from sklearn import model_selection
from toolbox_02450 import rlr_validate

def normalize(df):
    df2=(df-df.mean())/df.std()
    return df2

# Load the data
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
C = len(classNames)


#finding out most correlated values with our chosen column
corrMatrix= df.corr()
spamCorrelation=corrMatrix.loc[' char_freq_!']
maxSpamCorr = spamCorrelation.abs().sort_values(ascending=False)


XforLinReg=df[maxSpamCorr[1:10].index.to_list()]  #need to exclude 1rst var (is y-column itself)
#XforLinReg=normalize(XforLinReg) #is going to be done in the validation later again
yforLinReg=df[' char_freq_!']
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(XforLinReg,yforLinReg)
# Compute model output:
y_est = model.predict(XforLinReg)

# Plot original data and the model output
plt.xkcd()
f = figure(figsize=(11,7))
plt.scatter(range(len(yforLinReg)),yforLinReg,color="red",alpha=0.5, label='True Values')
plt.scatter(range(len(y_est)),y_est,color="blue", label='Estimated Values')
plt.title('Linear Regression with most correlated variables')
plt.legend()
show()


#test: linear regression on reduced vector of major values
XforLinReg2=df[maxSpamCorr[0:10].index.to_list()]
XforLinReg2=XforLinReg2[XforLinReg2.iloc[:,0]>1.5]
XforLinReg2=normalize(XforLinReg2)
yforLinReg2=XforLinReg2.iloc[:,0]
XforLinReg2=XforLinReg2.drop(columns=[' char_freq_!'], axis=1)

#fit a linear regression
model2 = lm.LinearRegression(fit_intercept=True)
model2 = model2.fit(XforLinReg2,yforLinReg2)
plt.figure(figsize=(11,7))
y_est2 = model.predict(XforLinReg2)

#plot the result
plt.scatter(range(len(yforLinReg2)),yforLinReg2, color="red", label='Real Values')
plt.scatter(range(len(y_est2)),y_est2,color="blue", label='Estimated Values')
plt.title('Linear Regression with most correlated variables - only values >1.5')
plt.legend()
plt.show()

#start the 10-fold validation
# Add offset attribute
X1 = np.concatenate((np.ones((XforLinReg.shape[0],1)),XforLinReg),1)
N, M = XforLinReg.shape
attributeNames = [u'Offset']+ maxSpamCorr[1:10].index.to_list()
yforLinReg=np.array(yforLinReg)
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.arange(10,500,10)
#for the optimal lambda:
#lamdas=[252]

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
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
    print('optimal lambda in Fold ',k,':',opt_lambda)
    print('Testing error at optimal lambda', k, min(test_err_vs_lambda))
    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))
