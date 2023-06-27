import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

df = pd.DataFrame([[1, 10.56],
                   [2, 27],
                   [3, 39.1],
                   [4, 40.4],
                   [5, 58],
                   [6, 60.5],
                   [7, 79],
                   [8, 87],
                   [9, 90],
                   [10, 95]],
                  columns=['X', 'Y'])
M = []  ###用来存储弱学习器
n_trees = 200  ###构造的弱学习器的数量

for i in range(n_trees):
    tmp = df.sample(frac=1.0, replace=True) #放回采样,frac指的是抽取比例
    X = tmp.iloc[:, :-1]
    Y = tmp.iloc[:, -1]
    model = DecisionTreeRegressor(max_depth=1)
    model.fit(X, Y)
    M.append(model)

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

#取另一的model做比较
model01 = DecisionTreeRegressor(max_depth=1)
model01.fit(x, y)
y_hat_01 = model01.predict(x)
print(y_hat_01)
print(model01.score(x, y))
print("-" * 100)

#Bagging进行预测
res = np.zeros(df.shape[0]) #将我的输出的结果进行相加
for j in M:
    res += j.predict(x)
y_hat = res / n_trees #将结果求评价
print(y_hat)
print('R2:', r2_score(y, y_hat))