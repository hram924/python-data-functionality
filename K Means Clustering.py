import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report
%matplotlib inline

df = pd.read_csv('College_Data.csv',index_col=0)
df.head()
df.info()
df.describe()

#Room.board vs grad rate
sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

#tuition vs f.undergrad           
sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)
   
#tuition vs private
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)

#gradrate vs private
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

#one college obviously has an incorrect grad rate of over 100% so i fixed it and changed to to 100
df[df['Grad.Rate'] > 100]
df['Grad.Rate']['Cazenovia College'] = 100
df[df['Grad.Rate'] > 100]
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

#Kcluster
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private',axis=1))
kmeans.cluster_centers_

#evaluation
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0
df['Cluster'] = df['Private'].apply(converter)
df.head()
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))