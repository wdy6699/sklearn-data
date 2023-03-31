from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
print('coef:\n', clf.coef_)
print('intercept:\n', clf.intercept_)
print('predict first two:\n', clf.predict(X_train[:2, :]))#预测
print('classification score:\n', clf.score(X_train, y_train))#测试得分
predict_y = clf.predict(X_test)
print('classfication report:\n', metrics.classification_report(y_test, predict_y))

