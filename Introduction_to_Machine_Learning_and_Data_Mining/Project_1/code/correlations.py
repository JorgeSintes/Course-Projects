#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
import seaborn as sns
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt

# Load the Spambase csv data using the Code provided in class
filename = 'C:/Users/klara/Documents/Studium/Master/IntroductiontoMachineLearning/spambase.csv'
df = pd.read_csv(filename)

raw_data = df.to_numpy() 
cols = range(0, 58) 
X = raw_data[:, cols]

# Extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.array([classDict[cl] for cl in classLabels])
N, M = X.shape
C = len(classNames)


# In[45]:


#function which returns top correlations 
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df, 5))


# In[46]:


#function which gets the lowest correlation
def get_lowest_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=True)
    return au_corr[0:n]

print("Lowest Absolute Correlations")
print(get_lowest_abs_correlations(df, 5))


# In[47]:


#scatter the variables 857 and 415, classified by spam/no spam
a=plt.scatter(df[df['spam(Y/N)']==0][' word_freq_857'], df[df['spam(Y/N)']==0][' word_freq_415'], color='green', label='No Spam')
b=plt.scatter(df[df['spam(Y/N)']==1][' word_freq_857'], df[df['spam(Y/N)']==1][' word_freq_415'], color='red', alpha=0.4, label='Spam')
plt.ylabel('Frequency of 415')
plt.xlabel('Frequency of 857')
plt.legend(handles=[a,b])
plt.show()


# In[48]:


#scatter the variables direct and 415, classified by spam/no spam
a=plt.scatter(df[df['spam(Y/N)']==0][' word_freq_415'], df[df['spam(Y/N)']==0][' word_freq_direct'], color='green', label='No Spam')
b=plt.scatter(df[df['spam(Y/N)']==1][' word_freq_415'], df[df['spam(Y/N)']==1][' word_freq_direct'], color='red', alpha=0.4, label='Spam')
plt.ylabel('Frequency of direct')
plt.xlabel('Frequency of 415')
plt.legend(handles=[a,b])
plt.show()


# In[49]:


#print the correlation matrix 
corrMatrix= df.corr()
#print the values most correlated with "spam"
spamCorrelation=corrMatrix[-1:]
maxSpamCorr = spamCorrelation.abs().unstack()
maxSpamCorr=maxSpamCorr.sort_values(ascending=False)
print(maxSpamCorr[:10])
#print the correlation between george and spam
print(corrMatrix[' word_freq_george']['spam(Y/N)'])


# In[38]:


#scatter frequency of "your"
x=np.random.normal(0,1, (len(df[df['spam(Y/N)']==0])))
x2=np.random.normal(0,1, (len(df[df['spam(Y/N)']==1])))
plt.figure(figsize=(3,7))
plt.scatter(x, df[df['spam(Y/N)']==0][' word_freq_your'], color='green', label='No Spam')
plt.scatter(x2, df[df['spam(Y/N)']==1][' word_freq_your'], color='red', alpha=0.4, label='Spam')
plt.legend(handles=[a,b])
plt.ylabel('Frequency of your')


# In[39]:


#scatter frequency of "george"
x=np.random.normal(0,1, (len(df[df['spam(Y/N)']==0])))
x2=np.random.normal(0,1, (len(df[df['spam(Y/N)']==1])))
plt.figure(figsize=(3,7))
a=plt.scatter(x, df[df['spam(Y/N)']==0][' word_freq_george'], color='green', label='No Spam')
b=plt.scatter(x2, df[df['spam(Y/N)']==1][' word_freq_george'], color='red', alpha=0.4, label='Spam')
plt.legend(handles=[a,b])
plt.ylabel('Frequency of the word george')


# In[40]:


#heatmap of the correlation matrix
corrMatrix= df.corr()
sns.heatmap(corrMatrix, annot=False)
plt.show()


# In[41]:


#slice the heatmap to zoom in on the more significant parts
indexes=np.append(np.arange(22,40),-2)
indexes
corrMatrix2= df.iloc[:,indexes].corr()
sns.heatmap(corrMatrix2, annot=False)
plt.show()






