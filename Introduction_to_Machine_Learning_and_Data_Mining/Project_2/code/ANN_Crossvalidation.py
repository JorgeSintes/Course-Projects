# exercise 8.2.6
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
import pandas as pd
from toolbox_02450 import rlr_validate
import numpy as np, scipy.stats as st

# Load the Iris csv data using the Pandas library
filename = 'C:/Users/george/Desktop/MLrep1/spambase.csv'
df = pd.read_csv(filename)

# Convert pandas framework to numpy array
raw_data = df.to_numpy()

# Collect the headers of the columns that we will deal with
attributeNames1 = np.asarray(df.columns[:])


correl = df.corr()
correl = correl.abs()

a=correl.nlargest(11, [" char_freq_!"]).index[1:]

data = df[df.columns[df.columns.isin(a)]]

raw_data = data.to_numpy()

attributeNames = list(np.asarray(data.columns[:]))

y = df[" char_freq_!"].to_numpy()
X = raw_data
y = y.reshape((4601,1))

# Normalize data
#X = stats.zscore(X)
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1


              
## Normalize and compute PCA (change to True to experiment with PCA preprocessing)
do_pca_preprocessing = False
if do_pca_preprocessing:
    Y = stats.zscore(X,0)
    U,S,V = np.linalg.svd(Y,full_matrices=False)
    V = V.T
    #Components to be included as features
    k_pca = 3
    X = X @ V[:,:k_pca]
    N, M = X.shape


# Parameters for neural network classifier
n_hidden_units = [10,11,12,13,14,15,16,17,18,19]
h_num = len(n_hidden_units)      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000

# K-fold crossvalidation
K1 = 10
K2 = 10                   # only three folds to speed up this example
CV = model_selection.KFold(K1, shuffle=True)
errors_final = np.empty([K1,5]) # make a list for storing generalizaition error in each loop  
#############################################
lambdas = np.arange(10,500,10)
Error_train_rlr = np.empty((K1,1))
Error_test_rlr = np.empty((K1,1))
Error_train_nofeatures = np.empty((K1,1))
Error_test_nofeatures = np.empty((K1,1))
w_rlr = np.empty((M,K1))
mu = np.empty((K1, M-1))
sigma = np.empty((K1, M-1))
w_noreg = np.empty((M,K1))
k=0
#############################################
#######   LOOP FOR OUTER FOLD      #########
############################################
for (k1, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k1+1,K1))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    
    CV1 = model_selection.KFold(K2, shuffle=True)
    ##########################################
    y_est_baseline = y_train.mean()
    baseline_pred = y_est_baseline * np.ones([len(y_test),1])
    baseline_loss = (y_test - baseline_pred)**2
    baseline_error = sum(baseline_loss)/(len(y_test))
    errors_final[k1,4]=baseline_error
    
    ###########################################
    #   HOLD DATA FOR STATISTIC TEST
     ##########################################
    XX_train = X[train_index,:]
    yy_train = y[train_index]
    XX_test = X[test_index,:]
    yy_test = y[test_index]
    ##########################################
    #        tRAIN  RLR model
    ############################################
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, K2)
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
    Error_train_rlr[k] = np.square(y_train.reshape(len(y_train))-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test.reshape(len(y_test))-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    y_est_reg = X_test @ w_rlr[:,k]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    errors_final[k1,2] = opt_lambda
    errors_final[k1,3] = Error_test_rlr[k]
    k+=1
    #################################################
    inside_error_matrix = np.empty([K2,h_num])
    #############################################
#######   LOOP FOR inner FOLD      #########
############################################
    for (k2, (train_index1, test_index1)) in enumerate(CV1.split(X_train,y_train)):
        X_train1 = torch.Tensor(X[train_index1,:])
        y_train1 = torch.Tensor(y[train_index1])
        X_test1 = torch.Tensor(X[test_index1,:])
        y_test1 = torch.Tensor(y[test_index1])
        
        #############################################
#######   LOOP for model selection ANN      #########
############################################
        
        for h_index in range(h_num):
            
            h=n_hidden_units[h_index]
            print('\nCrossvalidation fold: {0}/{1} for inside {2} fold and {3} hid. units'.format(k1+1,K1,k2+1, n_hidden_units[h_index]))
            # Define the model
            model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, h), #M features to n_hidden_units
                            torch.nn.Tanh(),   # 1st transfer function,
                            torch.nn.Linear(h, 1), # n_hidden_units to 1 output neuron
                            # no final tranfer function, i.e. "linear output"
                            )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            print('Training model of type:\n\n{}\n'.format(str(model())))
    # Train the net on training data
            net1, final_loss1, learning_curve1 = train_neural_net(model,
                                                                  loss_fn,
                                                                  X=X_train1,
                                                                  y=y_train1,
                                                                  n_replicates=n_replicates,
                                                                  max_iter=max_iter)
    
            print('\n\tBest loss: {}\n'.format(final_loss1))
    
    # Determine estimated class labels for test set
            y_test_est1 = net1(X_test1)
    
    # Determine errors and errors
            se1 = (y_test_est1.float()-y_test1.float())**2 # squared error
            mse1 = (sum(se1).type(torch.float)/len(y_test1)).data.numpy() #mean
            inside_error_matrix[k2,h_index] = mse1
        #errors.append(mse) # store error rate for current CV fold 
    Error_of_each_model = inside_error_matrix.mean(axis=0)
    h_opt_index = np.argmin(Error_of_each_model)
    h_opt = n_hidden_units[h_opt_index]
    
    # Train again on the outer testset for the h opt model
    model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, h_opt), #M features to n_hidden_units
            torch.nn.Tanh(),   # 1st transfer function,
            torch.nn.Linear(h_opt, 1), # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
            )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    print('Training model of type:\n\n{}\n'.format(str(model())))
    #############Turn arrays to tensor
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    y_test_est = net(X_test)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    errors_final[k1,0]=h_opt
    errors_final[k1,1]=mse
#####################################################################
Table = pd.DataFrame(errors_final,columns=['h_star','ANN Error','lambda_star','RLR Error', "Baseline"])

print('\n Table of Regresion Part B \n##################################\n{}'.format(Table))   
    
######################################################################  
#           Prepare data for statistic test
#############################################  
Y_est_base = baseline_pred
Y_est_reg = y_est_reg 
Y_est_ANN = y_test_est.data.numpy()
Y_est_ANN =Y_est_ANN.reshape([len(Y_est_base),1])
Y_est_reg = Y_est_reg.reshape([len(Y_est_base),1])


zA = np.abs(yy_test - Y_est_ANN ) ** 2
zB = np.abs(yy_test - Y_est_reg ) ** 2
zC = np.abs(yy_test - Y_est_base ) ** 2


z1 = zA - zB
z2 = zA - zC
z3 = zB - zC
alpha = 0.05
#############    T-STatistic ###########################
#####################################################################
CI1 = st.t.interval(1-alpha, len(z1)-1, loc=np.mean(z1), scale=st.sem(z1))  # Confidence interval
p1 = st.t.cdf( -np.abs( np.mean(z1) )/st.sem(z1), df=len(z1)-1)  # p-value
##############################################################
CI2 = st.t.interval(1-alpha, len(z2)-1, loc=np.mean(z2), scale=st.sem(z2))  # Confidence interval
p2 = st.t.cdf( -np.abs( np.mean(z2) )/st.sem(z2), df=len(z2)-1)  # p-value
#####################################################################
CI3 = st.t.interval(1-alpha, len(z3)-1, loc=np.mean(z3), scale=st.sem(z3))  # Confidence interval
p3 = st.t.cdf( -np.abs( np.mean(z3) )/st.sem(z3), df=len(z3)-1)  # p-value