import pandas as pd
import numpy as np
import sys
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler  ##标准化，归一化
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

pd.set_option("display.max_columns", None) #显示Dataform的所有列

datas = pd.read_csv('./data/risk_factors_cervical_cancer.csv', sep=',')
print(datas.head())
print(datas.info())

datas.replace('?', np.nan, inplace=True)
names = datas.columns

imputer = SimpleImputer()
datas = imputer.fit_transform(datas)
datas = pd.DataFrame(datas, columns=names)

X = datas.iloc[:, :-4]
Y = datas.iloc[:, -4:].astype('int')

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

models = [Pipeline([('standarscaler', StandardScaler()),
                    ('pca', PCA()),
                    ('RF', RandomForestClassifier())]),
          Pipeline([
              ('pca', PCA(n_components=0.5)),
              ('RF', RandomForestClassifier(n_estimators=10, max_depth=20))])
          ]

params = {'pca__n_components':[0.5,0.6,0.7,0.8,0.9], ##对pca的n_components参数这些进行测试
          'RF__n_estimators':[50,100,150], ##对随机森林的n_estimators参数这些进行测试
          'RF__max_depth':[1,3,5,7]} ##

model = GridSearchCV(estimator=models[1],param_grid=params,cv=5) #进行5折交叉验证，通过上面设置的参数进行组合测试
model.fit(x_train, y_train)
print('最优参数：', model.best_params_)
print('最优模型：', model.best_estimator_)
print('最优模型的分数：', model.best_score_)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

model = models[1]
model.fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

import joblib
joblib.dump(model,'./risk01.m')