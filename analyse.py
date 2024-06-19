import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

def plot(gb, filename):

    X = gb["datetime"].to_numpy().reshape(-1, 1)
    X = scaler.fit_transform(X)

    y = gb["price"].to_numpy()
    y = y - y[0]

    X_train = X[:-1]
    y_train = y[:-1]

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ConstantKernel

    kernel = DotProduct() * RBF() # + WhiteKernel(noise_level=0, noise_level_bounds=(1e-5, 1e-3))
    # kernel = WhiteKernel() + DotProduct()
    # kernel = 1.0 * RBF()
    gaussian_process = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=20)
    gaussian_process.fit(X_train, y_train)
    print(gaussian_process.kernel_)

    mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

    plt.plot(X, y, label=r"Data", linestyle="dotted")
    plt.scatter(X_train, y_train, label="Observations")
    plt.plot(X, mean_prediction, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    _ = plt.title(filename)
    plt.savefig(filename)

    plt.close()

    bounds = np.array([mean_prediction[-1] - 1.96 * std_prediction[-1], mean_prediction[-1] + 1.96 * std_prediction[-1]]) + y[0]
    print(bounds)

df = pd.read_csv("coinbaseUSD.csv", engine="pyarrow",  names=['unixtime','price','amount'])

df['datetime'] = pd.to_datetime(df['unixtime'], unit='s')
df = df[(df['datetime'] > '2015-01-18')]
print(df.tail(1))

print(df.set_index('datetime').resample(timedelta(hours = 1), origin='end').mean().tail(60))

# gb = df.groupby([df.datetime.dt.month, df.datetime.dt.year]).mean()
# print(gb.head())

# gb = df.groupby([df.datetime.dt.year])['price'].mean().reset_index()

# gb = df[df.datetime.dt.year == 2016].groupby([df.datetime.dt.month])['price'].mean().reset_index()

# gb = df[(df.datetime.dt.year == 2016) & (df.datetime.dt.month == 1)].groupby([df.datetime.dt.day])['price'].mean().reset_index()

# gb = df[(df.datetime.dt.year == 2016) & (df.datetime.dt.month == 1) & (df.datetime.dt.day == 1)].groupby([df.datetime.dt.hour])['price'].mean().reset_index()

# gb = df[(df.datetime.dt.year == 2016) & (df.datetime.dt.month == 1) & (df.datetime.dt.day == 1) & (df.datetime.dt.hour == 0)].groupby([df.datetime.dt.minute])['price'].mean().reset_index()

# gb = df[(df.datetime.dt.year == 2019) & (df.datetime.dt.month == 1) & (df.datetime.dt.day == 7) & (df.datetime.dt.hour == 21)].groupby([df.datetime.dt.minute])['price'].mean().reset_index()

gb = df.set_index('datetime').resample(timedelta(days = 365.2422), origin='end').mean().reset_index()
plot(gb, 'years.pdf')

# gb = df.set_index('datetime').resample(timedelta(days = 30.437), origin='end').mean().reset_index().tail(12)

# gb = df.set_index('datetime').resample(timedelta(days = 1), origin='end').mean().reset_index().tail(31)

# gb = df.set_index('datetime').resample(timedelta(hours = 1), origin='end').mean().reset_index().tail(24)

gb = df.set_index('datetime').resample(timedelta(minutes= 1), origin='end').mean().reset_index().tail(60)
plot(gb, 'minutes.pdf')

print(gb)

# df[(df.datetime.dt.year == 2016) & (df.datetime.dt.month == 1) & (df.datetime.dt.day == 1) & (df.datetime.dt.hour == 0)].plot(x="datetime", y="price", kind="line")
# plt.show()
