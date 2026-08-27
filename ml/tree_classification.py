import pandas as pd
import numpy as np


class MyTreeClf:
    def __init__(
            self,
            max_depth: int=5,
            min_samples_split: int=2,
            max_leafs: int=20
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

    def __repr__(self):
        return (f'MyTreeClf class: max_depth={self.max_depth}, '
                f'min_samples_split={self.min_samples_split}, '
                f'max_leafs={self.max_leafs}')



if __name__ == '__main__':
    model = MyTreeClf(max_depth=6, min_samples_split=4, max_leafs=20)
    print(model)


def get_best_split(X: pd.DataFrame, y: pd.Series):
    n = len(y)
    y = y.to_numpy()
    _, count_uniq_y = np.unique(y, return_counts=True)
    s0 = -np.sum(count_uniq_y * np.log2(count_uniq_y / n) / n)
    col_name, split_value, ig_max = None, None, -np.inf

    for feature_name, features in X.items():
        features = features.to_numpy()
        unique_features = np.unique(features)
        splits = (unique_features[:-1] + unique_features[1:]) / 2

        for split in splits:
            mask = features <= split
            left, right = y[mask], y[~mask]
            _, left_count_uniq = np.unique(left, return_counts=True)
            _, right_count_uniq = np.unique(right, return_counts=True)
            s_left = -np.sum(left_count_uniq * np.log2(left_count_uniq / len(left)) / len(left))
            s_right = -np.sum(right_count_uniq * np.log2(right_count_uniq / len(right)) / len(right))
            ig = s0 - (len(left) * s_left / n) - (len(right) * s_right / n)
            if ig > ig_max:
                col_name, split_value, ig_max = feature_name, split, ig

    return col_name, split_value, ig_max