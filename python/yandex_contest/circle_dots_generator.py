import random
import numpy as np


# method 1
def generate1():
    a = random.uniform(0, 1)
    b = random.uniform(0, 1)
    return a * np.cos(2 * np.pi * b), a * np.sin(2 * np.pi * b)

# method 2
def generate2():
    while True:
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x ** 2 + y ** 2 > 1:
            continue
        return x, y


n_dots = 1000
n_samples = 100

gen1 = []
gen2 = []
for s in range(n_samples):
    sample_gen1 = []
    sample_gen2 = []
    for d in range(n_dots):
        sample_gen1.append(generate1())
        sample_gen2.append(generate2())
    gen1.append(sample_gen1)
    gen2.append(sample_gen2)

# из списков можно получить массивы numpy, чтобы быстрее производить вычисления
g1_arr = np.array(gen1)
g2_arr = np.array(gen2)

# получили массивы размерностью (100, 1000, 2)
# посчитаем медианы радиусов для каждого из 100 объектов

g1_medians = np.median(np.sqrt(g1_arr[:, :, 0]**2 + g1_arr[:, :, 1]**2), axis=1)
g2_medians = np.median(np.sqrt(g2_arr[:, :, 0]**2 + g2_arr[:, :, 1]**2), axis=1)

# теперь разметим данные, соединим 2 массива и разделим их на трен и тест

zeros = np.zeros(100)
ones = np.ones(100)
X = np.concatenate((g1_medians, g2_medians), axis=0).reshape(-1, 1)
y = np.concatenate((zeros, ones), axis=0)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# импортировали нужную модель, теперь нужно создать объект класса и работать с моделью

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

w = model.coef_[0][0]
bias = model.intercept_[0]

print(w, bias)




