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


#TUTORIAL
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

# FROM DATA
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')
zdata = df["parallax"]
xdata = df["dec"]
ydata = df["ra"]
ax.scatter3D(xdata, ydata, zdata, c=df["phot_g_mean_mag"], cmap='viridis', marker='.', s=1, linewidths=0)
#ax.plot_trisurf(xdata, ydata, zdata, cmap='viridis', edgecolor='none')
plt.show()