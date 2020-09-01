import numpy as np
import astropy
from sklearn.mixture import GaussianMixture as GMM
import PyAstronomy
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt


def best_fit_gmm(df, max_n_components=8, max_iter=100, graph=True, best_fit_criterion='bic', var_name='X'):
    df = df.dropna()
    data = df.to_numpy().reshape(-1, 1)
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
        fig = plt.figure(figsize=(5, 1.7))
        fig.suptitle(str(var_name))
        fig.subplots_adjust(left=0.12, right=0.97,
                            bottom=0.21, top=0.9, wspace=0.5)

        # plot 1: data + best-fit mixture
        min_x = np.amin(data)
        max_x = np.amax(data)
        ax = fig.add_subplot(131)
        x = np.linspace(min_x, max_x, 1000)
        log_prob = best.score_samples(x.reshape(-1, 1))
        responsibilities = best.predict_proba(x.reshape(-1, 1))
        pdf = np.exp(log_prob)
        pdf_individual = responsibilities * pdf[:, np.newaxis]  # makes it as a column vector and multiplies
        ax.hist(data, 30, density=True, histtype='stepfilled', alpha=0.4)
        ax.plot(x, pdf, '-k')
        ax.plot(x, pdf_individual, '--k')
        ax.text(0.04, 0.96, "Best-fit Mixture",
                ha='left', va='top', transform=ax.transAxes)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$p(x)$')

        # plot 2: AIC and BIC
        ax = fig.add_subplot(132)
        ax.plot(n, aic, '-k', label='AIC')
        ax.plot(n, bic, '--k', label='BIC')
        ax.set_xlabel('n. components')
        ax.set_ylabel('information criterion')
        ax.legend(loc=2)

        # plot 3: posterior probabilities for each component
        ax = fig.add_subplot(133)

        p = responsibilities
        p = p.cumsum(1).T

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(0, 1)
        ax.set_xlabel('$x$')
        ax.set_ylabel(r'$p({\rm class}|x)$')

        alpha_step = float(1)/float(best_n_components)
        for i in range(best_n_components):
            alpha = alpha_step*(i+1)
            if alpha == 1:
                alpha -= 0.001
            ax.fill_between(x, p[i], p[i+1], color='gray', alpha=alpha)
            #ax.text(best.means_[i], 0.5, 'class '+str(i+1), rotation='vertical')

        #plt.show()
    return best, plt


df = pd.read_csv("NGC3114.csv")
if df.isnull().values.any():
    df = df.dropna()

columns = list(df)
models = []
plots = []
for i in range(len(columns)):
    model, plot = best_fit_gmm(df[columns[i]], 4, var_name=columns[i])
    plots.append(plot)
    models.append(models)
plt.show()

