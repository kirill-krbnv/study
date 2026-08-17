from sklearn.linear_model import LogisticRegression

sk_model = LogisticRegression()
sk_model.fit(X_train, y_train)
print(sk_model.intercept_, sk_model.coef_)