import numpy as np
from sklearn.model_selection import train_test_split
from luojihuigui import luoji
from sklearn.preprocessing import StandardScaler


X = np.genfromtxt('wine.data', delimiter=',', usecols=range(1, 14))
y = np.genfromtxt('wine.data', delimiter=',', usecols=[0])
idx = y != 3 #设置索引不为3的
X = X[idx]
y = y[idx] - 1

clf = luoji.LuojiRegress(n_iter=2000, eta=0.01, tol=0.0001)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

ss = StandardScaler()
ss.fit(X_train)
X_train_std = ss.transform(X_train)
X_test_std = ss.transform(X_test)
clf.train(X_train_std, y_train)
#为写完，是书上的手写代码实现！！！
