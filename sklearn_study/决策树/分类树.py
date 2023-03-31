from pandas import *
from numpy import *
from sklearn.model_selection import *
from sklearn.tree import *
from sklearn.metrics import *
from matplotlib.pyplot import *
from sklearn.datasets import *  # 如果你不好自行找数据集，可以用此法导入内置数据集

travel = read_csv(r'./travel.csv', encoding='cp936', index_col=0)
X = travel[['landscape', 'ticket', 'traffic', 'malignant']]
y = travel['sense']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=48)
Traveling = DecisionTreeClassifier(criterion='gini', splitter='best')
RM = Traveling.fit(X_train, y_train)
Predict_Value = Traveling.predict(X_test)
print(Traveling.score(X_test, y_test))
print(accuracy_score(y_test, Predict_Value))
print(classification_report(y_test, Predict_Value))
# print(sum((Predict_Value-y_test)**2)/len(y_test))
# mean_squared_error(y_test,Predict_Value)

fig = figure(figsize=(36, 33), facecolor='lightyellow')
plot_tree(Traveling, feature_names=['landscape', 'ticket', 'traffic', 'malignant'], class_names='sense')
savefig('ChatGPT_Score分类.png')

scores = cross_val_score(Traveling, X, y, cv=5, scoring='accuracy')
