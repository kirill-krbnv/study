import pandas as pd


class MyKNNReg:
    def __init__(
            self,
            k: int = 3,
            metric = 'euclidean',
            weight = 'uniform'
    ):
        self.k = k
        self.train_size = None
        self.X_train = None
        self.y_train = None
        self.metric = metric
        self.weight = weight

    def __repr__(self):
        return f'MyKNNReg class: k={self.k}'

    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series
    ):
        self.train_size = X_train.shape
        self.X_train = X_train.to_numpy()
        self.y_train = y_train.to_numpy()

    def _get_distance(self, X_test):
        x1 = X_test
        x2 = self.X_train
        metrics = dict(euclidean=lambda: np.sqrt(np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)),
                       chebyshev=lambda: np.max(np.abs(x1[:, None, :] - x2[None, :, :]), axis=2),
                       manhattan=lambda: np.sum(np.abs(x1[:, None, :] - x2[None, :, :]), axis=2),
                       cosine=lambda: 1 - (np.sum(x1[:, None, :] * x2[None, :, :], axis=2)) / (
                               (np.linalg.norm(x1[:, None, :], axis=2)) * np.linalg.norm(x2[None, :, :], axis=2)))

        return metrics.get(self.metric)()

    def predict(self, X_test: pd.DataFrame):
        test_index = X_test.index
        X_test = X_test.to_numpy()

        dist_array = self._get_distance(X_test)
        k_nearest = np.argsort(dist_array, axis=1)[:, :self.k]
        ranks = np.arange(1, self.k + 1)

        if self.weight != 'uniform':

            if self.weight == 'rank':
                weights = (1 / ranks) / np.sum(1 / ranks)

            elif self.weight == 'distance':
                rows = np.arange(dist_array.shape[0])[:, None]
                k_dist = dist_array[rows, k_nearest]
                weights = (1 / k_dist) / np.sum(1 / k_dist, axis=1, keepdims=True)

            y_pred = np.sum(self.y_train[k_nearest] * weights, axis=1)

        else:
            y_pred = np.mean(self.y_train[k_nearest], axis=1)
        return pd.Series(y_pred, index=test_index)

    def _r_squared(self, y_true: pd.Series, y_pred: pd.Series):
        y_true = y_true.to_numpy()
        y_pred = y_pred.to_numpy()

        num = np.sum((y_true-y_pred)**2)
        den = np.sum((y_true - np.mean(y_true))**2)
        res = 1 - num / den

        return np.round(res, 4)

if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import make_regression, make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=100, n_features=10, random_state=42, n_informative=9)
    X, y = pd.DataFrame(X), pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = MyKNNReg(k=3, metric='cosine', weight='distance')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(model.predict(X_test=X_test))
    # print(model._r_squared(y_true=y_test, y_pred=y_pred))
    # metrics = ['euclidean', 'chebyshev', 'manhattan', 'cosine']

    # r2_dict = {}
    # for metric in metrics:
    #     r2_dict.setdefault(metric, [f'Metric: {metric}'])
    #
    # for metric in metrics:
    #     j = 3
    #     k = 8
    #     r2_dict[metric].append(f'K is range({j}, {k})')
    #
    #     for k in range (j, k):
    #         model = MyKNNReg(k=k, metric=metric)
    #         model.fit(X_train, y_train)
    #         y_pred = model.predict(X_test)
    #         r2_dict[metric].append(model._r_squared(y_true=y_test, y_pred=y_pred))
    #
    # print(*r2_dict.values(), sep='\n')