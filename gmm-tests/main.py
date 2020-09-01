import astropy
import sklearn as sk
from sklearn.mixture import GaussianMixture as GMM
import os
import PyAstronomy
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("NGC3114.csv")
if df.isnull().values.any():
    df = df.dropna()
#print(tabulate(df, headers='keys', tablefmt='psql'))
#df.to_html('temp.html')

"""PROBANDO KMEANS"""
km = KMeans(n_clusters=3).fit(df[['pmra', 'pmdec']])
plt.scatter(df['ra'], df['dec'], c=km.labels_.astype(float), s=10, alpha=0.1)

"""PROBANDO GMM"""
dfgmm = df[['ra', 'dec', 'parallax']]
gmm = GMM(n_components=16, covariance_type='full').fit(dfgmm)
labels = gmm.predict(dfgmm)
probs = gmm.predict_proba(dfgmm)
print(probs[:5].round(3))
size = 50 * probs.max(1) ** 2  # square emphasizes differences
plt.scatter(df['ra'], df['dec'], c=labels, cmap='viridis', s=5, alpha=0.5)
