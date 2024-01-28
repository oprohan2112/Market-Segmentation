#!/usr/bin/env python
# coding: utf-8

# ![Marketing%20Segmentation.jpeg](attachment:Marketing%20Segmentation.jpeg)

# #  .......................................... Market Segmentition ...............................................

# ### Importing Libraries 

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore") 


# ### Loading Dataset 

# In[2]:


rohan_df = pd.read_csv("Customer Data.csv")
rohan_df


# ### Exploring Data ( EDA ) 

# In[3]:


rohan_df.head(6)


# In[4]:


rohan_df.tail(6)


# In[5]:


rohan_df.shape


# In[6]:


rohan_df.info()


# In[7]:


rohan_df.describe()


# ### Finding and Filling Null Values 

# In[8]:


rohan_df.isnull().sum()


# In[9]:


# filling mean value in place of missing values in the dataset
rohan_df["MINIMUM_PAYMENTS"] = rohan_df["MINIMUM_PAYMENTS"].fillna(rohan_df["MINIMUM_PAYMENTS"].mean())
rohan_df["CREDIT_LIMIT"] = rohan_df["CREDIT_LIMIT"].fillna(rohan_df["CREDIT_LIMIT"].mean()) 


# In[10]:


rohan_df.isnull().sum()


# In[11]:


# checking for duplicate rows in the dataset
rohan_df.duplicated().sum()


# In[12]:


# drop CUST_ID column because it is not used
rohan_df.drop(columns=["CUST_ID"],axis=1,inplace=True)


# In[13]:


rohan_df.columns


# In[14]:


plt.figure(figsize=(30,45))
for i, col in enumerate(rohan_df.columns):
    if rohan_df[col].dtype != 'object':
        ax = plt.subplot(9, 2, i+1)
        sns.kdeplot(rohan_df[col], ax=ax)
        plt.xlabel(col)
        
plt.show()


# In[15]:


plt.figure(figsize=(10,60))
for i in range(0,17):
    plt.subplot(17,1,i+1)
    sns.distplot(rohan_df[rohan_df.columns[i]],kde_kws={'color':'b','bw': 0.1,'lw':3,'label':'KDE'},hist_kws={'color':'g'})
    plt.title(rohan_df.columns[i])
plt.tight_layout()


# In[16]:


plt.figure(figsize=(12,12))
sns.heatmap(rohan_df.corr(), annot=True)
plt.show()


# ### Feature Scaling 

# In[18]:


from sklearn.preprocessing import StandardScaler
StandardScalar = StandardScaler()
rohan_df1 = StandardScalar.fit_transform(rohan_df)
rohan_df1


# ### Dimensionalty Reduction 

# In[20]:


pca = PCA(n_components=2)
principal_components = pca.fit_transform(rohan_df1)
pca_df = pd.DataFrame(data=principal_components ,columns=["PCA1","PCA2"])
pca_df


# ### Hyperparameter Tunning  

# ### Finding "K" Value by using Elbow Method

# In[21]:


inertia = []
range_val = range(1,15)
for i in range_val:
    kmean = KMeans(n_clusters=i)
    kmean.fit_predict(pd.DataFrame(rohan_df))
    inertia.append(kmean.inertia_)
plt.plot(range_val,inertia,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show()


# ### Model Building using KMeans 

# In[23]:


kmeans_model=KMeans(4)
kmeans_model.fit_predict(rohan_df1)
pca_df_kmeans= pd.concat([pca_df,pd.DataFrame({'cluster':kmeans_model.labels_})],axis=1)


# # Visualizing the Clustered Dataframe 

# In[24]:


plt.figure(figsize=(8,8))
ax=sns.scatterplot(x="PCA1",y="PCA2",hue="cluster",data=pca_df_kmeans,palette=['red','green','blue','black'])
plt.title("Clustering using K-Means Algorithm")
plt.show()


# In[34]:


# find all cluster centers
cluster_centers = pd.DataFrame(data=kmeans_model.cluster_centers_,columns=[rohan_df.columns])
cluster_centers


# In[36]:


# Creating a target column "Cluster" for storing the cluster segment
cluster_df = pd.concat([rohan_df,pd.DataFrame({'Cluster':kmeans_model.labels_})],axis=1)
cluster_df


# In[37]:


cluster_1_df = cluster_df[cluster_df["Cluster"]==0]
cluster_1_df


# In[38]:


cluster_2_df = cluster_df[cluster_df["Cluster"]==1]
cluster_2_df


# In[39]:


cluster_3_df = cluster_df[cluster_df["Cluster"]==2]
cluster_3_df


# In[40]:


cluster_4_df = cluster_df[cluster_df["Cluster"] == 3]
cluster_4_df


# In[41]:


#Visualization
sns.countplot(x='Cluster', data=cluster_df)


# In[43]:


for c in cluster_df.drop(['Cluster'],axis=1):
    grid= sns.FacetGrid(cluster_df, col='Cluster')
    grid= grid.map(plt.hist, c)
plt.show()


# 
# Type Markdown and LaTeX: 𝛼^2

# ## Saving the kmeans clustering model and the data with cluster label

# In[44]:


#Saving Scikitlearn models
import joblib
joblib.dump(kmeans_model, "kmeans_model.pkl")


# In[45]:


cluster_df.to_csv("Clustered_Customer_Data.csv")


# ## .................................. Preparing the Data for Model ........................................

# In[46]:


#Split Dataset
X = cluster_df.drop(['Cluster'],axis=1)
y= cluster_df[['Cluster']]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3)


# In[47]:


print("X_train :", X_train.size)
print("X_test :", X_test.size)
print("y_train :" ,y_train.size)
print("y_test :", y_test.size)


# ## .......................................... Applying Machine Learning Algorithm .......................................... 

# ### Machine Learning Algorithm : Logistic Regression 

# In[48]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

lr = LogisticRegression()
model1 = lr.fit(X_train , y_train)
Prediction1 = model1.predict(X_test)

print( "Testing Accurancy :" , accuracy_score(y_test , Prediction1))


# ### Machine Learning Algorithm : SVC 

# In[52]:


from sklearn.svm import SVC
SVC = SVC()
model3 = SVC.fit(X_train,y_train)
Prediction3 = model3.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction3))


# ### Machine Learning Algorithm : DecisionTreeClassifier

# In[53]:


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
model4 = DT.fit(X_train,y_train)
Prediction4 = model4.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction4))


# ### Machine Learning Algorithm : GaussianNB 

# In[54]:


from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
model5 = GNB.fit(X_train,y_train)
Prediction5 = model5.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction5))


# ### Machine Learning Algorithm : RandomForestClassifier 

# In[55]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
model6 = RF.fit(X_train, y_train)
Prediction6  = model6.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,Prediction6))


# In[56]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['Log-Reg', 'SVC', 'Des-Tree', 'Gaus-NB', 'RandomForest']
accuracy = [82.08 , 80.78 , 94.04, 91.99 , 96.38 ]
ax.bar(langs,accuracy)
plt.show()


# ### The Best Accuracy is given by RandomForest  Classifier is 96.38  .

# In[57]:


#Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , Prediction6)
cm


# In[58]:


sns.heatmap(cm , annot = True , cmap = "BuPu")
plt.show()


# ### precision and recall of the model

# In[59]:


from sklearn.metrics import classification_report
print( classification_report(y_test , Prediction6))


# ### Saving the Decision tree model for future prediction 

# In[61]:


import pickle
filename = 'final_model.sav'
pickle.dump(model6 , open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result,'% Acuuracy')


# ### The Best Accuracy is given by RandomForestClassifier is 96.38 . Hence we will use RandomForestClassifier algorithms for training my model
# 
