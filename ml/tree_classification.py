import pandas as pd
import numpy as np
from graphviz import Digraph


class TreeNode:
    def __init__(self, feature=None, split=None, proba=None, n_samples=None):
        self.feature = feature
        self.split = split
        self.left_child = None
        self.right_child = None
        self.proba = proba
        self.n_samples = n_samples

    def __repr__(self):
        if self.proba is None:
            return f'{self.feature} > {self.split}'
        else:
            return f'leaf = {self.proba}'


class MyTreeClf:
    def __init__(
            self,
            max_depth: int = 5,
            min_samples_split: int = 2,
            max_leafs: int = 20,
            bins=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        if self.max_leafs < 2:
            self.max_leafs = 2
        self.leafs_cnt = 1
        self.tree = None
        self.bins = bins

    def __repr__(self):
        return (f'MyTreeClf class: max_depth={self.max_depth}, '
                f'min_samples_split={self.min_samples_split}, '
                f'max_leafs={self.max_leafs}')

    def _get_best_split(self, X: pd.DataFrame, y: pd.Series):
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

    def _build_tree(self, X_train, y_train, depth=0):
        if (
                self.leafs_cnt >= self.max_leafs
                or len(X_train) < self.min_samples_split
                or depth >= self.max_depth
                or len(np.unique(y_train)) == 1
        ):
            return TreeNode(proba=np.mean(y_train), n_samples=len(y_train))

        feature, split, _ = self._get_best_split(X_train, y_train)

        if feature is None:
            return TreeNode(proba=np.mean(y_train), n_samples=len(y_train))
        else:
            self.leafs_cnt += 1
            node = TreeNode(feature=feature, split=split)
            node.n_samples = len(y_train)
            mask = X_train[node.feature] <= node.split
            x_left, x_right = X_train[mask], X_train[~mask]
            y_left, y_right = y_train[mask], y_train[~mask]
            left_child = self._build_tree(x_left, y_left, depth=depth + 1)
            right_child = self._build_tree(x_right, y_right, depth=depth + 1)
            node.left_child = left_child
            node.right_child = right_child
            return node

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        if self.bins:
            for col in X_train.to_numpy().T:
                nat_split_count = col.unique()
                if nat_split_count <= self.bins - 1:
                    # будем использовать native split
                else:
                    # строим гистограмму для фичи и используем разделители бинов для поиска наилучшего сплита
        self.tree = self._build_tree(X_train, y_train)

    def recursive_predict(self, node, row: pd.Series):
        if node.feature is None:
            return node.proba
        feature = node.feature
        if row[feature] <= node.split:
            return self.recursive_predict(node.left_child, row)
        else:
            return self.recursive_predict(node.right_child, row)

    def predict_proba(self, X: pd.DataFrame):
        proba_list = []
        for _, row in X.iterrows():
            predicted_proba = self.recursive_predict(self.tree, row)
            proba_list.append(predicted_proba)
        return proba_list

    def predict(self, X: pd.DataFrame):
        proba_list = self.predict_proba(X)
        return [0 if p <= 0.5 else 1 for p in proba_list]

    def print_tree(self, node=None, shift=0, direct=None):
        if node is None:
            node = self.tree
        if not node.left_child and not node.right_child:
            print(shift * "\t" + f'{direct} {node} /// depth = {shift} /// n_samples = {node.n_samples}')
        else:
            print(shift * "\t" + f'{node} /// depth = {shift} /// n_samples = {node.n_samples}')
            self.print_tree(node.left_child, shift + 1, direct='left')
            self.print_tree(node.right_child, shift + 1, direct='right')

    def draw_tree(self, filename='tree'):
        dot = Digraph(node_attr={'shape': 'box', 'style': 'rounded,filled', 'fontname': 'Helvetica'})
        self._add_node(dot, self.tree, '0')
        dot.render(filename, format='png', view=True, cleanup=True)
        return dot

    def _add_node(self, dot, node, node_id):
        if node.proba is not None:
            color = '#c8e6c9' if node.proba in (0.0, 1.0) else '#ffe0b2'
            dot.node(node_id, f'{node.proba:.3f}\nn = {node.n_samples}', fillcolor=color)
            return

        dot.node(node_id, f'{node.feature} > {node.split:.3f}\nn = {node.n_samples}', fillcolor='#eceff1')

        left_id, right_id = node_id + 'L', node_id + 'R'
        self._add_node(dot, node.left_child, left_id)
        self._add_node(dot, node.right_child, right_id)
        dot.edge(node_id, left_id, label='≤')
        dot.edge(node_id, right_id, label='>')


if __name__ == '__main__':
    model = MyTreeClf(max_depth=3, min_samples_split=2, max_leafs=6)
    print(model)

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, n_informative=9, n_redundant=1,
                               random_state=42)
    X, y = pd.DataFrame(X), pd.Series(y)
    X.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']  # просто для наглядности

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    model.draw_tree()

    # model.print_tree()





