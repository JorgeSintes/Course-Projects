from scipy import stats
from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Get values from Table 2 file

df = pd.read_csv('/home/yorch/Documents/DTU/02450_Machine_Learning/Project 2/Plots 10/Table2_10.csv')

print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

lambdas = df['lambda'].to_numpy()
log_error = df['Log error'].to_numpy()/100
KNN_K = df['K'].to_numpy()
KNN_error = df['KNN error'].to_numpy()/100
baseline_error = (df['Baseline error'].to_numpy())

#no. of folds
F1 = 10
L = 20

#Calculate 95% credibility intervals for model comparisons
#Baseline vs KNN
z = (baseline_error - KNN_error)
zb = z.mean()
nu = F1-1
sig =  (z-zb).std()  / np.sqrt(F1-1)
alpha = 0.05
zL = zb + sig * stats.t.ppf(alpha/2, nu);
zU = zb + sig * stats.t.ppf(1-alpha/2, nu);
print("95% CI for Baseline error - KNN error:")
print("[",round(zL,4),",",round(zU,4),"]")
p = stats.t.cdf(-np.abs(np.mean(z))/stats.sem(z), df=len(z)-1)  # p-value
print('p-value:', p)

#Baseline vs Logistic regression
z = (baseline_error - log_error)
zb = z.mean()
nu = F1-1
sig =  (z-zb).std()  / np.sqrt(F1-1)
alpha = 0.05
zL = zb + sig * stats.t.ppf(alpha/2, nu);
zU = zb + sig * stats.t.ppf(1-alpha/2, nu);
print("95% CI for Baseline error - Logistic error:")
print("[",round(zL,4),",",round(zU,4),"]")
p = stats.t.cdf(-np.abs(np.mean(z))/stats.sem(z), df=len(z)-1)  # p-value
print('p-value:', p)

#KNN vs Logistic regression
z = (KNN_error - log_error)
zb = z.mean()
nu = F1-1
sig =  (z-zb).std()  / np.sqrt(F1-1)
alpha = 0.05
zL = zb + sig * stats.t.ppf(alpha/2, nu);
zU = zb + sig * stats.t.ppf(1-alpha/2, nu);
print("95% CI for KNN error - Logistic error:")
print("[",round(zL,4),",",round(zU,4),"]")
p = stats.t.cdf(-np.abs(np.mean(z))/stats.sem(z), df=len(z)-1)  # p-value
print('p-value:', p)