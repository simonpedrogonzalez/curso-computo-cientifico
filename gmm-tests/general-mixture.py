import pomegranate as pg
import pandas as pd


def general_mixture_classification(df, gaussian_count=10):
    data = df[["dec", "ra"]]

    distributions = [pg.UniformDistribution]
    for i in range(gaussian_count):
        distributions.append(pg.NormalDistribution)

    model = pg.GeneralMixtureModel.from_samples(distributions, n_components=len(data.columns), X=data)



df = pd.read_csv("NGC3114.csv")
if df.isnull().values.any():
    df = df.dropna()


mix = pg.GeneralMixtureModel([pg.MultivariateGaussianDistribution([1, 6], [[1, 0], [0, 1]]),
                              pg.IndependentComponentsDistribution([pg.UniformDistribution(1, 1),
                                                                    pg.UniformDistribution(0, 1)])])

field = pg.IndependentComponentsDistribution([pg.UniformDistribution.from_samples(df["dec"].to_numpy()),
                                              pg.UniformDistribution.from_samples(df["ra"].to_numpy())])

cluster = pg.MultivariateGaussianDistribution.from_samples(df[["dec", "ra"]].to_numpy())

mix = pg.GeneralMixtureModel.from_samples([pg.MultivariateGaussianDistribution,
                                           pg.IndependentComponentsDistribution([pg.UniformDistribution,
                                                                                 pg.UniformDistribution])],
                                          n_components=3, X=df[["dec", "ra"]].to_numpy())


model = pg.GeneralMixtureModel.from_samples([pg.NormalDistribution, pg.ExponentialDistribution, pg.LogNormalDistribution], n_components=5, X=df)


d1 = pg.NormalDistribution(5, 2)
d2 = pg.LogNormalDistribution(1, 0.3)
d3 = pg.ExponentialDistribution(4)
d = pg.IndependentComponentsDistribution([d1, d2, d3])


x = np.linspace(0, 10, 100)