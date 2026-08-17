import pandas as pd
import numpy as np
import random
from typing import Union

class MyLineReg:
    def __init__(self,
                 n_iter: int=100,
                 learning_rate=0.1,
                 weights=None,
                 metric: str=None,
                 reg: str=None,
                 l1_coef: float = 0,
                 l2_coef: float = 0,
                 sgd_sample: Union[int, float, None]=None,
                 random_state: int=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def _l_rate(self, step_n=0):
        if callable(self.learning_rate):
            return self.learning_rate(step_n)
        else:
            return self.learning_rate

    def _add_bias_column(self, X: pd.DataFrame) -> pd.DataFrame:
        ones_column = pd.DataFrame(np.ones(shape=(len(X), 1)), index=X.index, columns=['bias'])
        features_matrix = pd.concat(objs=[ones_column, X], axis=1, join='outer')
        return features_matrix

    def _grad(self, features_matrix, y_pred, y):
        return (2 * features_matrix.T.dot(y_pred - y)) / len(features_matrix) + self._reg_component(derivative=True)

    def _batches(self, features_matrix, y_pred, y):
        sgd_sample = self.sgd_sample
        if isinstance(sgd_sample, float):
            sample_size = round(len(features_matrix) * sgd_sample)
        else:
            sample_size = sgd_sample
        sample_rows_idx = random.sample(range(features_matrix.shape[0]), sample_size)
        features_batch = pd.DataFrame(features_matrix.iloc[sample_rows_idx])
        y_pred_batch = pd.Series(y_pred.iloc[sample_rows_idx])
        y_batch = pd.Series(y.iloc[sample_rows_idx])
        return features_batch, y_pred_batch, y_batch

    def _get_metric(self, y_pred, y) -> float:
        metrics = {
            'mae': lambda: (np.abs(y - y_pred)).mean(),
            'mse': lambda: ((y_pred - y) ** 2).mean(),
            'rmse': lambda: np.sqrt(((y_pred - y) ** 2).mean()),
            'mape': lambda: 100 * (np.abs((y - y_pred) / y)).mean(),
            'r2': lambda: 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        }
        return metrics.get(self.metric)()

    def _get_fit_log(self, step_n, mse, y, y_pred, lr) -> str:
        log_str = f'{step_n} | loss: {round(mse, 2)}'
        if self.metric:
            return log_str + f' | <{self.metric}>: {round(self._get_metric(y_pred=y_pred, y=y), 4)} | lr: {lr}'
        else:
            return log_str

    def _pred_and_mse(self, features_matrix, y) -> tuple:
        y_pred = features_matrix.dot(self.weights)
        mse = ((y_pred - y) ** 2).mean() + self._reg_component()
        return y_pred, mse

    def _reg_l1(self, derivative: bool=False):
        if derivative:
            return self.l1_coef * np.sign(self.weights)
        else:
            return self.l1_coef * np.sum(np.abs(self.weights))

    def _reg_l2(self, derivative: bool=False):
        if derivative:
            return self.l2_coef * 2 * self.weights
        else:
            return self.l2_coef * np.sum(self.weights**2)

    def _reg_elasticnet(self, derivative: bool=False):
        if derivative:
            return self._reg_l1(derivative=True) + self._reg_l2(derivative=True)
        else:
            return self._reg_l1() + self._reg_l2()

    def _reg_component(self, derivative: bool = False):
        if self.reg == 'l1':
            return self._reg_l1(derivative=derivative)
        elif self.reg == 'l2':
            return self._reg_l2(derivative=derivative)
        elif self.reg == 'elasticnet':
            return self._reg_elasticnet(derivative=derivative)
        else:
            return 0

    def __repr__(self) -> str:
        parameters = []
        for key, value in self.__dict__.items():
            parameters.append(f'{key}={value}')
        result = 'MyLineReg class: ' + ', '.join(parameters)
        return result

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            verbose: int=False) -> None:
        random.seed(self.random_state)
        W = np.ones(X.shape[1] + 1)
        self.weights = W
        features_matrix = self._add_bias_column(X)
        y_pred, mse = self._pred_and_mse(features_matrix, y)
        if verbose:
            print(self._get_fit_log(step_n='start', mse=mse, y=y, y_pred=y_pred, lr=self.learning_rate))
        for step_n in range(1, self.n_iter + 1):
            y_pred, mse = self._pred_and_mse(features_matrix, y)
            lr = self._l_rate(step_n)
            if self.sgd_sample:
                features_batch, y_pred_batch, y_batch = self._batches(features_matrix, y_pred, y)
                cur_grad = self._grad(features_batch, y_pred_batch, y_batch)
            else:
                cur_grad = self._grad(features_matrix, y_pred, y)
            self.weights = self.weights - lr * cur_grad
            if verbose:
                if step_n % verbose == 0:
                    print(self._get_fit_log(step_n=step_n, mse=mse, y=y, y_pred=y_pred, lr=lr))
        y_pred, _ = self._pred_and_mse(features_matrix, y)
        if self.metric:
            self.best_score = self._get_metric(y=y, y_pred=y_pred)

    def get_coef(self) -> np.ndarray:
        return self.weights[1:]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        features_matrix = self._add_bias_column(X)
        y_pred = features_matrix.dot(self.weights)
        return y_pred

    def get_best_score(self):
        return self.best_score