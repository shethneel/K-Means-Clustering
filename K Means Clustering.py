#!/usr/bin/env python
# coding: utf-8

# # K MEANS CLUSTERING

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


cus = pd.read_csv('/Users/neelsheth/Desktop/anjan/003_Machine Learning/UnSupervised_learning/K_Means/Customers.csv')


# In[3]:


cus.head()


# In[4]:


X = cus.iloc[:,3:5].values


# # Elbow method to find clusters
# 

# In[5]:


from sklearn.cluster import KMeans


# In[6]:


man = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    man.append(kmeans.inertia_)


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


plt.plot(range(1,11),man)
plt.title('Elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Man')
plt.show()


# # PREDECTING VALUES

# In[9]:


kmeans = KMeans (n_clusters = 5, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)


# # VISUALISING THE CLUSTERS

# In[10]:


# 5 cluster labling and points
plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s = 100, c = 'red', label = 'cluster 1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s = 100, c = 'green', label = 'cluster 2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s = 100, c = 'blue', label = 'cluster 3')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s = 100, c = 'cyan', label = 'cluster 4')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s = 100, c = 'yellow', label = 'cluster 5')

# centroid labling and point
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 200, c = 'black', label = 'centroid')
            
# Final Graph
            
plt.title ('Clusters of customers')
plt.xlabel ('Annual Income')
plt.ylabel ('Spending Score')
plt.legend ()
plt.show ()

