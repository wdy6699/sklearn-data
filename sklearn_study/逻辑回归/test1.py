from numpy import *
from pandas import *
from sklearn.linear_model import *
from matplotlib.pyplot import *
from sklearn.model_selection import *
from sklearn.datasets import *
from mpl_toolkits.mplot3d import *
from sklearn.tree import *
from sklearn import metrics
import warnings #导入警告库，目的是消除红色警告

def xianxintiankong():
    Data = read_csv(r'./credit-overdue.csv', encoding='cp936', index_col=False)
    X = array(Data['debt']).reshape(-1, 1)
    Y = Data['income'].values.reshape(-1, 1)
    Model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=1)
    Model.fit(X, Y)
    pre = Model.predict([[mean(Y)]])
    Score = Model.score(X, Y)
    print('预测值:', pre[0][0])
    print('检验值R^2:', Score)
    # print(Model.intercept_[0])
    # print(Model.coef_)
    scatter(Data['debt'], Data['income'], c=Data['overdue'], marker='^')
    plot([0, 2.5], [Model.intercept_[0], (pre[0][0]*Model.coef_+Model.intercept_)[0][0]], color='red')
    show()

def luojitiankong():
    Data = read_csv(r'./wine.csv', encoding='cp936', index_col=False)
    C0 = Data['Alcohol'].values.reshape(-1, 1)
    C1 = Data['Malic_acid'].values.reshape(-1, 1)
    A = array(Data['class']).reshape(-1, 1)
    fig = figure(figsize=(10, 6))
    ax = fig.add_subplot(2, 1, 1) # Axes3D(fig, elev=-152, azim=-26)
    ax.scatter(C0[A == 1], C1[A == 1], c='b', s=120, edgecolor='k', marker='^')
    ax.scatter(C0[A == 2], C1[A == 2], c='r', s=120, edgecolor='k', marker='.')
    ax.scatter(C0[A == 3], C1[A == 3], c='y', s=120, edgecolor='k', marker='.')
    X_train, X_test, y_train, y_test = train_test_split(C0, A, test_size=0.3, random_state=42)
    LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    LR.fit(X_train, y_train)
    predict = LR.predict(X_train[:2, :])
    print('预测值:', predict)
    print('检验值:', LR.score(X_train, y_train))

def fenleishu():
    data2, target2 = load_iris(return_X_y=True, as_frame=True)
    X2 = data2
    y2 = target2
    DTC = DecisionTreeClassifier(criterion='gini', splitter='best')
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3, random_state=42)
    DTC.fit(X_train, y_train)
    predict = DTC.predict(X_test)
    print(metrics.classification_report(y_test, predict))

if __name__ == '__main__':
    warnings.filterwarnings('ignore')  # 忽视在程序中红色警告（不为错，但只是为了控制台显示美观）
    # xianxintiankong()
    # luojitiankong()
    fenleishu()

