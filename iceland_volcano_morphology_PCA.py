#%%

import pandas as pd
from sklearn.decomposition import PCA
import numpy
import matplotlib.pyplot as plot

# df = pandas.DataFrame(data=numpy.random.normal(0, 1, (20, 10)))
filename = "/Users/francis/Desktop/SuppTable1_DF_all_corr_20200203.xlsx"
df = pd.read_excel(filename, skiprows=3)

#%%

# You must normalize the data before applying the fit method
df_normalized=(df - df.mean()) / df.std()
pca = PCA(n_components=df.shape[1])
pca.fit(df_normalized)

# Reformat and view results
loadings = pd.DataFrame(pca.components_.T,
columns=['PC%s' % _ for _ in range(len(df_normalized.columns))],
index=df.columns)
print(loadings)

plot.plot(pca.explained_variance_ratio_)
plot.ylabel('Explained Variance')
plot.xlabel('Components')
plot.show()