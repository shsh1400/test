import sys

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score

df = pd.DataFrame([[0, 1],
                   [1, 1],
                   [2, 1],
                   [3, -1],
                   [4, -1],
                   [5, -1],
                   [6, 1],
                   [7, 1],
                   [8, 1],
                   [9, -1]])

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

w1 = np.ones(df.shape[0]) / df.shape[0]

model1 = DecisionTreeClassifier(max_depth=1)
model1.fit(X, Y, sample_weight=w1)

e1 = sum(w1[model1.predict(X) != Y])
print(e1)

a1 = 0.5 * np.log((1 - e1) / e1)
print(a1)