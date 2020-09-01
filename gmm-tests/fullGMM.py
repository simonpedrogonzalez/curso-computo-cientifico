import numpy as np
import astropy
from sklearn.mixture import GaussianMixture as GMM
import PyAstronomy
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def best_fit_gmm(df, max_n_components=10, max_iter=100, graph=True, best_fit_criterion='bic'):
    df = df.dropna()
    data = df
    n = np.arange(1, max_n_components+1)
    models = [GMM(n_components=i, covariance_type='full', max_iter=max_iter).fit(data) for i in n]
    aic = [m.aic(data) for m in models]
    bic = [m.bic(data) for m in models]
    if best_fit_criterion == 'bic':
        best_n_components = np.argmin(bic)
    else:
        best_n_components = np.argmin(aic)
    best = models[best_n_components]


    if graph:

        labels = best.predict(data)
        prob = best.predict_proba(data)
        df.insert(6, "label", labels)
        label_prob = prob.max(1)
        df.insert(7, "label_prob", label_prob)

        # per group graph
        # global classification graph
        #   TODO: MAKE A GLOBAL FIGURE OF ALL THE GROUPS
        # fig_global = plt.figure()
        figs = []
        for i in range(best_n_components):
            data2plot = df.loc[df['label'] == i]
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_title("label "+str(i))
            z_data = data2plot["parallax"]
            x_data = data2plot["dec"]
            y_data = data2plot["ra"]
            color_data = data2plot["label"]
            marker_size = 50*data2plot["label_prob"] ** 2  # square emphasizes differences

            ax.scatter3D(df["dec"], df["ra"], df["parallax"], c='grey', marker='.', s=1, linewidths=0)
            ax.scatter3D(x_data, y_data, z_data, c=color_data, cmap='viridis', linewidth=0.001, marker='o', s=marker_size)
            figs.append(fig)

        #   full graph
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_title("full")
        zdata = df["parallax"]
        xdata = df["dec"]
        ydata = df["ra"]
        marker_size = 100*df["label_prob"] ** 2
        ax.scatter3D(xdata, ydata, zdata, c=labels, cmap='viridis', linewidth=0.001, marker='o', s=marker_size)
        figs.append(fig)

        # BIC graph
        fig = plt.figure()
        # plot 2: AIC and BIC
        ax2 = fig.add_subplot(1, 1, 1)
        ax2.plot(n, aic, '-k', label='AIC')
        ax2.plot(n, bic, '--k', label='BIC')
        ax2.set_xlabel('n. components')
        ax2.set_ylabel('information criterion')
        ax2.legend(loc=2)
        figs.append(fig)

    return best, figs


df = pd.read_csv("NGC3114.csv")
if df.isnull().values.any():
    df = df.dropna()

model, figs = best_fit_gmm(df, 15)

plt.show()

print(tabulate(model.means_, headers=['ra', 'dec', 'parallax', 'pmdec', 'pmra', 'mag'], tablefmt='psql'))
print(tabulate(model.covariances_[7], headers=['ra', 'dec', 'parallax', 'pmdec', 'pmra', 'mag'], tablefmt='psql'))