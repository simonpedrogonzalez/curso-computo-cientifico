import numpy as np
import astropy
from sklearn.mixture import GaussianMixture as GMM
import PyAstronomy
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

df = pd.read_csv("NGC3114.csv")
if df.isnull().values.any():
    df = df.dropna()

X = df["dec"].to_numpy().reshape(-1, 1)
plt.hist(X, 80, normed=True)
clf = GMM(n_components=10, covariance_type='full',
          max_iter=500, random_state=3).fit(X)
# puntos a tomar estimaciones de densidad
xpdf = np.linspace(np.amin(X), np.amax(X), 1000).reshape(-1, 1)
# densidad es exponente de score, donde score es el negativo de la likelihood
density = np.exp(clf.score_samples(xpdf))

plt.hist(X, 80, normed=True, alpha=0.5)
plt.plot(xpdf, density, '-r')
plt.show()

# kernelDensity: generalization of GMM except that for n points there are n gaussians.
from sklearn.neighbors import KernelDensity as KD

kd = KD(0.05).fit(X)
density_kd = np.exp(kd.score_samples(xpdf))
plt.hist(X, 80, normed=True, alpha=0.5)
plt.plot(xpdf, density, '-b', label='GMM')
plt.plot(xpdf, density_kd, '-r', label='KDE')
plt.legend()
plt.show()