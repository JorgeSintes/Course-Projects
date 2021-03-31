#Loading and preprocessing data using this course's structure.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from classification_variables import *

#Let's implement logistic regression on our data-set
def log_reg(X, y, n, plot_error = True, plot_lambda = False):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    font_size = 20
    plt.rcParams.update({'font.size': font_size})
    
    # Create crossvalidation partition for evaluation
    # using stratification and 95 pct. split between training and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # Try to change the test_size to e.g. 50 % and 99 % - how does that change the 
    # effect of regularization? How does differetn runs of  test_size=.99 compare 
    # to eachother?
    
    # Standardize the training and set set based on training set mean and std
    
    # Fit regularized logistic regression model to training data to predict 
    # the type of wine
    lambda_interval = 1.52642
    train_error_rate = 0
    test_error_rate = 0
    coefficient_norm = 0
    #try for all the lambdas
    mdl = LogisticRegression(penalty='l2', C=1/lambda_interval, max_iter=10000)
    
    mdl.fit(X_train, y_train)
    coefs = mdl.coef_[0]
    print(coefs, attributeNames[indeces])
    
    
    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
    
    train_error_rate = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate = np.sum(y_test_est != y_test) / len(y_test)

    w_est = mdl.coef_[0] 
    coefficient_norm = np.sqrt(np.sum(w_est**2))
    '''
    error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]
    '''
    #plots
    if plot_error:
        plt.figure(figsize=(8,8))
        #plt.plot(np.log10(lambda_interval), train_error_rate*100)
        #plt.plot(np.log10(lambda_interval), test_error_rate*100)
        #plt.plot(np.log10(opt_lambda), min_error*100, 'o')
        plt.semilogx(lambda_interval, train_error_rate*100)
        plt.semilogx(lambda_interval, test_error_rate*100)
        plt.semilogx(opt_lambda[n], min_error[n]*100, 'or')
        plt.text(1e-5, 15, "Minimum test error: " + str(np.round(min_error[n]*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda[n]),2)))
        plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
        plt.ylabel('Error rate (%)')
        plt.title('Classification error')
        plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
        plt.ylim([0, 30])
        plt.grid()
        plt.show()
    elif plot_lambda:
        plt.figure(figsize=(8,8))
        plt.semilogx(lambda_interval, coefficient_norm,'k')
        plt.ylabel('L2 Norm')
        plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
        plt.title('Parameter vector L2 norm')
        plt.grid()
        plt.show()

min_error = np.zeros(58)
opt_lambda = np.zeros(58)
log_reg(X, y, n, plot_error=False)