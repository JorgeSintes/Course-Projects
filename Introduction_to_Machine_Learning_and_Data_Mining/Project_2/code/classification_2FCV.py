import matplotlib.pyplot as plt
from sklearn import model_selection
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from tabulate import tabulate

from classification_variables import *

#no. of folds
F1 = 10
F2 = 10

#Range of complexity parameters
lambda_interval = np.logspace(-7, 4, 50)
maxiter = 10000
L = 20


#Error vectors
logistic_inner = np.zeros((F2, len(lambda_interval)))
logistic_outer = np.zeros(F2)
KNN_inner = np.zeros((F2, L))
KNN_outer = np.zeros(F2)
baseline_outer = np.zeros(F1)

#Result vectors

logistic_opt_error = np.zeros(F1)
logistic_opt_lambda = np.zeros(F1)
KNN_opt_error = np.zeros(F1)
KNN_opt_neighbors = np.zeros(F1)
baseline_opt = np.zeros(F1)

#Fold selectors
OuterFold = model_selection.KFold(F1, shuffle = True)
InnerFold = model_selection.KFold(F2, shuffle = True)

i = 0
for traino_index, testo_index in OuterFold.split(X, y):
    #Split dataset in train and test for OuterFold
    X_traino = X[traino_index, :]
    y_traino = y[traino_index]
    X_testo = X[testo_index, :]
    y_testo = y[testo_index]
    
    j = 0
    
    for traini_index, testi_index in InnerFold.split(X_traino, y_traino):
        #Split outer train data in train and test for InnerFold
        X_traini = X[traini_index, :]
        y_traini = y[traini_index]
        X_testi = X[testi_index, :]
        y_testi = y[testi_index]
        
        #Compute all complexity parameters for the 3 models:
        #1.Logistic regression
        for k in range(0, len(lambda_interval)):
            logregi = LogisticRegression(penalty='l2', C=1/lambda_interval[k], 
                                        max_iter=maxiter)
            logregi.fit(X_traini, y_traini)
            y_testi_est = logregi.predict(X_testi).T
            
            logistic_inner[j, k] = sum(y_testi_est != y_testi)/len(y_testi)
        
        #2.K-Nearest Neighbours
        for k in range(1, L+1):
            KNNi = KNeighborsClassifier(n_neighbors = k,
                                       p = 2,
                                       metric = 'cosine')
            KNNi.fit(X_traini, y_traini)
            y_testi_est = KNNi.predict(X_testi).T
            
            KNN_inner[j, k-1] = sum(y_testi_est != y_testi)/len(y_testi)
        
        #3.Baseline
        #Baseline doesn't need the inner fold, it can be done on the outer.
        
        j += 1
        print('Outer fold: ' + str(i+1)+'/'+str(F1)+'. Inner fold: '+str(j)+'/'+str(F2))
    #Once the inner fold is complete, we obtain the best value for each model's
    #complexity parameters, and train and test a model with that value on the outer
    #data-set
    
    #1.Logistic regression
    #Sum the error result for every value of lambda
    logistic_outer = np.sum(logistic_inner, axis = 0)/F2
    #Choosing the optimal value of lambda
    logistic_opt_lambda[i] = lambda_interval[np.argmin(logistic_outer)]
    #Compute with the optimal lambda the lg model with the outer training set
    logrego = LogisticRegression(penalty='l2', C=1/logistic_opt_lambda[i], 
                                 max_iter=maxiter)
    logrego.fit(X_traino, y_traino)
    #Obtain the error over the outer test set
    y_testo_est = logrego.predict(X_testo).T
    logistic_opt_error[i] = sum(y_testo_est != y_testo)/len(y_testo)
    
    #2.K-Nearest Neighbours
    #Sum the error result for every value of K
    KNN_outer = np.sum(KNN_inner, axis=0)/F2
    #Choosing the optimal value of K
    KNN_opt_neighbors[i] = np.argmin(KNN_outer) + 1
    #Compute the optimal KNN model
    KNNo = KNeighborsClassifier(n_neighbors = int(KNN_opt_neighbors[i]), 
                                p = 2, metric = 'cosine')
    KNNo.fit(X_traino, y_traino)
    #Obtain the error over the outer test set
    y_testo_est = KNNo.predict(X_testo).T
    KNN_opt_error[i] = sum(y_testo_est != y_testo)/len(y_testo)
    
    #3.Baseline
    baseline_outer[i] = stats.mode(y_traino)[0][0]
    baseline_opt[i] = sum(baseline_outer[i] != y_testo)/len(y_testo)
    
    i += 1

#Save the results on a DataFrame and print them
data = {'Outer fold': list(np.arange(1, F1+1)),
'lambda': list(logistic_opt_lambda),
'Log error': list(logistic_opt_error*100),
'K': list(KNN_opt_neighbors),
'KNN error': list(KNN_opt_error*100),
'Baseline error': list(baseline_opt*100)}

df = pd.DataFrame(data, columns = ['Outer fold', 'lambda', 'Log error', 'K', 
                                   'KNN error', 'Baseline error'])
print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

#Save it on a file
df.to_csv(r'Table2_'+ str(n)+'.csv', index = False)

#Plots

#Plot baseline for last outer fold
plt.figure(figsize=(9,8))
plt.plot(y_testo,'bo')
plt.hlines(baseline_outer[-1],0,len(y_testo)-1, 'r')
plt.gca().axes.get_xaxis().set_visible(False)
#plt.title("Baseline classification")
plt.ylabel('Class')
plt.legend(["True class","Predicted class"])
plt.show()

#Plot choice of K for KNN in last outer fold
plt.figure(figsize=(9,8))
plt.plot(np.arange(1,L+1,1),KNN_outer*100)
ticklabs=["","","","","5","","","","","10","","","","","15","","","","","20"]
plt.xticks(np.arange(1,L+1,1),ticklabs)
#plt.title("Generalization error in KNN classification")
plt.ylabel("Generalization error (%)")
plt.xlabel("Number of neighbors (K)")
plt.grid()
plt.show()

plt.figure(figsize=(9,8))
plt.semilogx(lambda_interval, logistic_outer*100)
#plt.title("Generalization error in Logistic classification")
plt.ylabel("Generalization error (%)")
plt.xlabel("Complexity parameter (λ)")
plt.grid()
