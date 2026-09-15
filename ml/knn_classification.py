import numpy as np
import pandas as pd


class MyKNNClf:
    def __init__(
            self,
            k: int = 3,
            metric: str = 'euclidean',
            weight: str = 'uniform'
    ) -> None:
        self.k = k
        self.train_size = None
        self.X_train = None
        self.y_train = None
        self.metric = metric
        self.weight = weight

    def __repr__(self) -> str:
        return f'MyKNNClf class: k={self.k}'


    def _get_distance(self, X_test_row):
        x1 = self.X_train
        x2 = X_test_row
        all_metrics = {
            'euclidean': lambda: pd.Series(np.sqrt(np.sum((x1 - x2) ** 2, axis=1))),
            'chebyshev': lambda: pd.Series(np.max(np.abs(x1 - x2), axis=1)),
            'manhattan': lambda: pd.Series(np.sum(np.abs(x1 - x2), axis=1)),
            'cosine': lambda: pd.Series(1 - np.sum(x1*x2, axis=1) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2)))
        }
        return all_metrics.get(self.metric)()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.train_size = X_train.shape
        self.X_train = X_train
        self.y_train = y_train


    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        return (self.predict_proba(X_test) >= 0.5).astype(int)

    def predict_proba(self, X_test: pd.DataFrame) -> list:
        proba_list = []

        for idx, x_row in X_test.iterrows():
            distances = self._get_distance(x_row)
            k_nearest = distances.nsmallest(self.k)
            tags = self.y_train.loc[k_nearest.index]

            if self.weight == 'uniform':
                proba = tags.sum() / self.k
                proba_list.append(proba)

            else:
                class_dist = pd.DataFrame({'tag': tags, 'distances': k_nearest})

                if self.weight == 'rank':
                    class_dist = class_dist.sort_values(by='distances', ignore_index=True)
                    class_dist.index += 1
                    class_dist['simple_weight'] = 1 / class_dist.index

                elif self.weight == 'distance':
                    class_dist['simple_weight'] = 1 / class_dist['distances']

                weight_1 = np.sum(
                    class_dist[class_dist['tag'] == 1]['simple_weight']
                ) / np.sum(class_dist['simple_weight'])
                proba_list.append(weight_1)

        return pd.Series(proba_list, index=X_test.index)

