#%%

import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


# df = pandas.DataFrame(data=numpy.random.normal(0, 1, (20, 10)))
filename = "/Users/francis/Desktop/SuppTable1_DF_all_corr_20200203.xlsx"
df_full = pd.read_excel(filename, skiprows=3)

df = df_full.iloc[1:,8:-2]
dff = df_full.iloc[1:,:]
df = df.drop(labels = [ 'Average Basal width',
'Minimum Basal width ',
 'Average plateau width',
 'Width ratio (Wp/Wb)',
 'Aspect ratio (H/Wb) ',
 'Elongation (Lb/Wb)',
 'Elongation ratio           ((Lb – Wb )/ Lb)',
 'Flank width ratio  ((Wb – Wp ) / Lb)',
 'Summit plateau ratio (Wp/Lb)'
 ], axis=1)


#%%

df_full = pd.read_csv('/Users/francis/Desktop/downloadSupplement.txt',delimiter='\t', engine='python', header=1)
df = df_full.drop(columns=['Mountain', 'Shape'])
df['Approximate Width, km'][2] = 98     # test

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
df_st = pd.DataFrame(scaled_data, columns = [cname for cname in df.columns])
pca_out = PCA(n_components=scaled_data.shape[1])
pca_out.fit(scaled_data)
scores = PCA().fit_transform(scaled_data)
#%%

# df_st=(df - df.mean()) / df.std()
# pca_out = PCA(n_components=df.shape[1])
# pca_out.fit(df_st)
# scores = PCA().fit_transform(df_st)
#%%
# Reformat and view results
loadings = pd.DataFrame(pca_out.components_.T,
columns=['PC%s' % _ for _ in range(len(df_st.columns))],
index=df.columns)
print(loadings)

variance = pca_out.explained_variance_ratio_
plt.plot(variance)
plt.ylabel('Explained Variance')
plt.xlabel('Components')
plt.show()

#%%
plt.figure()
for ii in range(3):
    plt.plot(loadings.iloc[:,ii],marker='.', linestyle='None')
plt.xticks(rotation=270)
plt.grid()
# %%
%matplotlib auto
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(scores[:,0], scores[:,1], scores[:,2])
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)

#%% 
color = np.arange(0, scores.shape[0])
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
ax1.scatter(scores[:,0], scores[:,1], c=df['Approximate Volume, km3']) 
ax2.scatter(scores[:,0], scores[:,2], c=df['Approximate Volume, km3'])  
ax3.scatter(scores[:,1], scores[:,2], c=df['Approximate Volume, km3'])  
ax4.scatter(scores[:,1], scores[:,3], c=df['Approximate Volume, km3'])  



# %%
# try to cluster in the PC space:

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    print(num_clusters)
    kmeans = KMeans(n_clusters=num_clusters, max_iter=1000)
    kmeans.fit(scores)
    
    ssd.append(kmeans.inertia_)

labels = kmeans.labels_
plt.figure()
plt.plot(range_n_clusters, ssd)

#%%

kmeans = KMeans(n_clusters=3, max_iter=1000)
kmeans.fit(scores)
labels = kmeans.labels_

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
ax1.scatter(scores[:,0], scores[:,1], c=labels) 
ax2.scatter(scores[:,0], scores[:,2], c=labels)  
ax3.scatter(scores[:,3], scores[:,2], c=labels)  
ax4.scatter(scores[:,1], scores[:,3], c=labels)  

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(scores[:,0], scores[:,1], scores[:,2],c=labels)
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)

#%%

# silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=1000)
    kmeans.fit(scores)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(scores, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
#%% plot the SSDs for each n_clusters
# ssd
plt.figure()
plt.plot(range_n_clusters, ssd)
# %%
