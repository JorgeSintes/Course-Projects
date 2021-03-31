import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
import seaborn as sns
from scipy.stats import mode

def originalnames(NaMes):
    names=[]
    for i in NaMes:
        a=i.split('_')
        names.append(a[-1])
    return names


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
#BoX Plots
X1=raw_data[raw_data[:,-1]==1]
X0=raw_data[raw_data[:,-1]==0]
X1=X1[:,0:57]
X0=X0[:,0:57]
X0means=X0.mean(0)
X1means=X1.mean(0)
X0std=X0.std(0)
X1std=X1.std(0)


labels=["Not_spam","Spam"]


#dot plot for means
my_xticks=originalnames(attributeNames[:54])
fig=plt.figure(figsize=(15,4))
ax = fig.add_axes([-1, -1, 2, 2])
x0means=X0[:,:54].mean(0)
x1means=X1[:,:54].mean(0)
plt.xticks(np.arange(54), my_xticks,rotation='vertical',fontsize=30)
plt.yticks(fontsize=30)
plt.plot(np.arange(54),x0means,'og',markersize=20,label = "Not Spam")
plt.plot(np.arange(54),x1means,'or',markersize=20,label = "Spam")
plt.grid()
plt.title('Average occurence of words',fontsize=30)
plt.legend(prop={"size":40})
#ax.xaxis.set_ticklabels(originalnames(attributeNames[:54]), rotation='vertical', fontsize=18)
plt.show()
#############################################
#dot plot for std
X1std=X1.std(0)
X2std=X1.std(0)
my_xticks=originalnames(attributeNames[:54])
fig=plt.figure(figsize=(15,4))
ax = fig.add_axes([-1, -1, 2, 2])
x0std=X0[:,:54].std(0)
x1std=X1[:,:54].std(0)
plt.xticks(np.arange(54), my_xticks,rotation='vertical',fontsize=30)
plt.yticks(fontsize=30)
plt.plot(np.arange(54),x0std,'og',markersize=20,label = "Not Spam")
plt.plot(np.arange(54),x1std,'or',markersize=20,label = "Spam")
plt.grid()
plt.title('St. Deviation of occurence of words',fontsize=30)
plt.legend(prop={"size":40})
#ax.xaxis.set_ticklabels(originalnames(attributeNames[:54]), rotation='vertical', fontsize=18)
plt.show()
#############################################

Titles1=["Average","Longest","Total"]
for i in range(54,57):
    data0=X0[:,i]
    data1=X1[:,i]
    fig = plt.figure(figsize =(2, 2))
    ax = fig.add_axes([-1, -1, 2, 2])
    bp = ax.boxplot([data0,data1],showfliers=False,vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
    ax.set_title('Capital run length {0}'.format(Titles1[i-54])) 
plt.show()
################################################
print("Mode of not spam")
a=mode(X0,axis=0)[0]
print(a)
print("Mode of spam")

b=mode(X1,axis=0)[0]
print(b)

################################################
#outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers=((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
outliers=outliers.drop('spam(Y/N)')

#outliers.plot(kind="bar", figsize=(30,10))
my_xticks=originalnames(attributeNames[:54])
fig=plt.figure(figsize=(15,4))
outliers = np.array(outliers.values.tolist())
plt.bar(np.arange(54),outliers[:54],align='center', alpha=0.8)
plt.xticks(np.arange(54), my_xticks,rotation='vertical',fontsize=15)
plt.title("Outliers")

plt.show()


