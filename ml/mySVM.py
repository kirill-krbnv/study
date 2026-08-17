import numpy as np
import pandas as pd

class MySVM:
    def __init__(
            self,
            n_iter=10,
            learning_rate=0.001,
            weights = None,
            b = None
    ) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.b = b

    def __repr__(self) -> str:
        return f'MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def _compute_full_loss(self, X: pd.DataFrame, y: pd.Series):

        margins = 1 - y * (X.to_numpy() @ self.weights + self.b)
        classification_error = np.maximum(0, margins).mean()
        margin_error = np.sum(self.weights**2)

        return margin_error + classification_error

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=None):

        if (y==0).any():
            y = y.replace(0, -1)
        n_features = X.shape[1]
        self.weights = np.ones(n_features)
        self.b = 1
        alpha = self.learning_rate

        if verbose:
            svm_loss = self._compute_full_loss(X, y)
            print(f'start | loss: {svm_loss:.2f}')

        for epoch in range(1, self.n_iter + 1):

            for index, x_row in X.iterrows():

                y_true = y[index]
                x_vec = x_row.to_numpy()

                if y_true * (self.weights @ x_vec + self.b) >= 1:
                    grad_w = 2 * self.weights
                    grad_b = 0
                else:
                    grad_w = 2 * self.weights - y_true * x_vec
                    grad_b = -y_true

                self.weights = self.weights - alpha * grad_w
                self.b = self.b - alpha * grad_b

            if verbose:
                svm_loss = self._compute_full_loss(X, y)
                print(f'{epoch} | loss: {svm_loss:.2f}')

    def get_coef(self):
        return self.weights, self.b

    def predict(self, X=pd.DataFrame):
        y = np.sign(X @ self.weights + self.b).astype(int)
        if (y==-1).any():
            y[y == -1] = 0
        return y

from sklearn.datasets import make_classification
X_train, y_train = make_classification(
    n_samples=100,
    n_classes=2,
    n_features=4,
    flip_y=0.05,
    random_state=47
)

X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)

model = MySVM(100, 0.002)
model.fit(X=X_train, y=y_train, verbose=5)
print(model)

print(y_train)
y_pred = model.predict(X_train)
print((y_pred == y_train).mean())
