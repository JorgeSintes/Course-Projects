from PCA1 import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import svd

def originalnames(NaMes):
    names=[]
    for i in NaMes:
        a=i.split('_')
        names.append(a[-1])
    return names

Xcol=[X[:,:54],X[:,54:]]
for i in range(len(Xcol)):
        #plt.xticks(np.arange(54), my_xticks,rotation='vertical')
    r = np.arange(1,Xcol[i].shape[1]+1)
    if i==0:
        f=figure(figsize=(25,10))
        plt.bar(r, np.std(Xcol[i],0))
        my_xticks=originalnames(attributeNames[:54])
        plt.xticks(r,my_xticks, rotation = 'vertical',fontsize=29)
        plt.yticks(fontsize=25)
        plt.title('Spambase: attribute standard deviations',fontsize=30)
        plt.ylabel('Standard deviation',fontsize=25)
    if i==1:
        f=figure(figsize=(10,10))
        name1=['Capital \n Length \n Average','Capital \n Length \n Longest','Capital \n Length \n Total']
        plt.xticks(r, name1,fontsize=25)
        plt.yticks(fontsize=20)
        plt.bar(r, np.std(Xcol[i],0),width=0.5)
        plt.title('Spambase: attribute standard deviations',fontsize=20)
        plt.ylabel('Standard deviation',fontsize=15)
    plt.xlabel('Attributes',fontsize=15)
    
    plt.show()
    
###########################################

# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot
plt.figure(figsize=(10,10))

    
    # Compute the projection onto the principal components

name=originalnames(attributeNames) 
for att in [8,18,20,24,26,29,30,39,42]:  #range(V.shape[1])
    plt.arrow(0,0, V[att,i], V[att,j])
    plt.text(1.3*V[att,i], 1.4*V[att,j], name[att],fontsize=15)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('PC'+str(i+1),fontsize=15)
plt.ylabel('PC'+str(j+1),fontsize=15)
plt.grid()
    # Add a unit circle
plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
     np.sin(np.arange(0, 2*np.pi, 0.01)));
plt.title( '\n'+'Attribute coefficients',fontsize=15)
plt.axis('equal')

i = 0
j = 1


# Make the plot
plt.figure(figsize=(10,10))
for i in range(4):
    name.pop(-1)
name.append("CRL_Aver")
name.append("CRL_Longest")
name.append("CRL_Total")
# Make the plot
plt.figure(figsize=(10,10))

    

    # Compute the projection onto the principal components
#%%
plt.figure(figsize=(10,10))
i = 1
j = 2
for att in [18,19,26,42,36,55]:  #range(V.shape[1])
    plt.arrow(0,0, V[att,i], V[att,j])
    plt.text(1.5*V[att,i], V[att,j], name[att],fontsize=15)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('PC'+str(i+1),fontsize=15)
plt.ylabel('PC'+str(j+1),fontsize=15)
plt.grid()
    # Add a unit circle
plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
     np.sin(np.arange(0, 2*np.pi, 0.01)));
plt.title( '\n'+'Attribute coefficients',fontsize=15)
plt.axis('equal')