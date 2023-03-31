from pandas import *
from numpy import *
from sklearn.model_selection import *
from sklearn.tree import *
from sklearn.metrics import *
from matplotlib.pyplot import *
from sklearn.datasets import *  # 如果你不好自行找数据集，可以用此法导入内置数据集

ChatGPT_Score = read_csv(r'./cgptSCORE.csv', encoding='cp936', index_col=0)
X = ChatGPT_Score[['response', 'efficiency', 'function']]
y = ChatGPT_Score['sense']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44)
Regressor = DecisionTreeRegressor(criterion='mse', splitter='best')
RM = Regressor.fit(X_train, y_train)
Predict_Value = Regressor.predict(X_test)
print(Regressor.score(X_test, y_test))
print(sum((Predict_Value - y_test) ** 2) / len(y_test))
mean_squared_error(y_test, Predict_Value)

fig = figure(figsize=(36, 33), facecolor='lightyellow')
plot_tree(Regressor, feature_names=['response', 'efficiency', 'function'], class_names='sense')
savefig('ChatGPT_Score回归.png')

scores = cross_val_score(Regressor, X, y, cv=5, scoring='r2')
