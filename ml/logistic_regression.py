import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import random



class MyLogReg:
    def __init__(
            self,
            n_iter=10,
            learning_rate=0.1,
            W=None,
            metric=None,
            reg: str = None,
            l1_coef: float = 0,
            l2_coef: float = 0,
            sgd_sample=None,
            random_state=42
    ) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.W = W
        self.metric = metric
        self.best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __repr__(self) -> str:
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int=False) -> None:
        random.seed(42)
        X = self._add_bias_column(X)
        self.W = np.ones(X.shape[1])
        z = X @ self.W
        y_pred = self._sigmoid(z)
        log_loss = self._logloss(y, y_pred)

        if verbose:
            metric_value = round(self._metric(y, y_pred), 2)
            print(self._log("start", log_loss, metric_value))

        for i in range(1, self.n_iter + 1):
            if callable(self.learning_rate):
                l_r = self.learning_rate(i)
            else:
                l_r = self.learning_rate
            z = X @ self.W
            y_pred = self._sigmoid(z)
            log_loss = self._logloss(y, y_pred)
            if self.sgd_sample:
                X_batch, y_pred_batch, y_batch = self.sgd_batches(X, y_pred, y)
                grad = self._grad(X_batch, y_batch, y_pred_batch)
            else:
                grad = self._grad(X, y, y_pred)

            self.W -= l_r * grad

            if verbose:
                if i % verbose == 0:
                    if self.metric:
                        metric_value = round(self._metric(y, y_pred), 2)
                        print(self._log(i, log_loss, metric_value))
                    else:
                        print(self._log(i, log_loss))

        if self.metric:
            z = X @ self.W
            y_pred = self._sigmoid(z)
            self.best_score = round(self._metric(y, y_pred), 10)

    def sgd_batches(self, X, y_pred, y):

        if isinstance(self.sgd_sample, int):
            batch_size = self.sgd_sample
        else:
            batch_size = round(self.sgd_sample * len(X))
        batch_rows_ids = random.sample(range(X.shape[0]), batch_size)
        X_batch = pd.DataFrame(X.iloc[batch_rows_ids])
        y_pred_batch = pd.Series(y_pred.iloc[batch_rows_ids])
        y_batch = pd.Series(y.iloc[batch_rows_ids])

        return X_batch, y_pred_batch, y_batch

    def _metric(self, y, y_pred):
        eps = 1e-15

        if self.metric == 'roc_auc':
            return self.roc_auc(y, y_pred)

        y_pred = self.binarization(y_pred)

        tp = np.sum((y_pred == 1) & (y == 1))
        tn = np.sum((y_pred == 0) & (y == 0))
        fp = np.sum((y_pred == 1) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))

        if self.metric == 'accuracy':
            return (tp + tn) / (tp + tn + fp + fn + eps)

        if self.metric == 'precision':
            return tp / (tp + fp + eps)

        if self.metric == 'recall':
            return tp / (tp + fn + eps)

        if self.metric == 'f1':
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            return 2 * precision * recall / (precision + recall + eps)

        return None

    def _add_bias_column(self, X) -> pd.DataFrame:
        X = X.copy()
        X.insert(0, "bias", 1.0)
        return X

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _logloss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) + self.regularization()

    def regularization(self, derivative: bool=False):
        if self.reg:
            if self.reg == 'l1':
                if derivative:
                    return  self.l1_coef * np.sign(self.W)
                else:
                    return self.l1_coef * np.sum(np.abs(self.W))
            elif self.reg == 'l2':
                if derivative:
                    return self.l2_coef * 2 * self.W
                else:
                    return self.l2_coef * np.sum(self.W ** 2)
            elif self.reg == 'elasticnet':
                if derivative:
                    return self.l1_coef * np.sign(self.W) + self.l2_coef * 2 *self.W
                else:
                    return self.l1_coef * np.sum(np.abs(self.W)) + self.l2_coef * np.sum(self.W ** 2)
            else:
                return 0
        else:
            return 0

    def _grad(self, X, y, y_pred):
        return (np.dot((y_pred - y), X)) / len(X) + self.regularization(derivative=True)

    def _log(self, i, log_loss, metric_value=None):
        if self.metric:
            return f'{i} | loss: {log_loss} | {self.metric}: {metric_value}'
        else:
            return f'{i} | loss: {log_loss}'


    def get_coef(self):
        return self.W[1:]

    def predict_proba(self, X):
        X = self._add_bias_column(X)
        z = X @ self.W
        return self._sigmoid(z)

    def predict(self, X):
        return self.binarization(self.predict_proba(X))

    def binarization(self, a):
         return np.where(a > 0.5, 1, 0)

    def roc_auc(self, y, y_pred):
        P = np.sum(y == 1)
        N = np.sum(y == 0)
        y_pred = pd.Series(y_pred.round(10))
        roc_df = pd.concat([y, y_pred], axis=1)
        roc_df.sort_values([1,0], axis=0, inplace=True, ascending=False)
        roc_df.reset_index(drop=True, inplace=True)
        res = 0
        for idx, row in roc_df[roc_df[0] == 0].iterrows():
            roc_slice = roc_df.iloc[:idx]
            res += np.sum(roc_slice[0] == 1)
            res += np.sum((roc_slice[0] == 1) & (roc_slice[1] == row[1])) / 2

        return res / (P * N)

    def get_best_score(self):
        return self.best_score



if __name__ == '__main__':
    # Готовим данные для тренировки и теста
    X_np, y_np = make_classification(1000, 4, n_classes=2, random_state=42, flip_y=0.05)
    X, y = pd.DataFrame(X_np), pd.Series(y_np)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Инициируем модель, запускаем тренировку и тест
    model = MyLogReg(n_iter=5000, learning_rate=lambda i: 0.5 * (0.85 ** i), metric='roc_auc', sgd_sample=40)
    print(model)
    model.fit(X, y, verbose=200)
    print(model.get_coef())
    y_pred = model.predict_proba(X)
    print(model.roc_auc(y, y_pred))
    print(model.get_best_score())
    # print(model.predict_proba(X_test))
    # print(model.predict(X_test))




    # print("Проверка на модели sklearn")
    # # Проверка на модели sklearn
    # from sklearn.linear_model import LogisticRegression
    #
    # sk_model = LogisticRegression()
    # sk_model.fit(X_train, y_train)
    # print(sk_model.intercept_, sk_model.coef_)


