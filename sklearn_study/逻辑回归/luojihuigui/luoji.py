import numpy as np

class LuojiRegress:
    def __init__(self, n_iter=200, eta=1e-3, tol=None):
        self.n_iter = n_iter #训练迭代次数
        self.eta = eta #学习率
        self.tol = tol #误差变化阈值
        self.w = None #模型参数w（训练时初始化）

    def _z(self, X, w): #g(x)函数：计算x与w的内积
        return np.dot(X, w)

    def _sigmoid(self, z): #logistic函数
        return 1. / (1. + np.exp(-z))

    def _predict_proba(self, X, w):
        z = self._z(X, w) #h（x）函数：预测为整例（y=1）的概率
        return self._sigmoid(z)

    def _loss(self, y, y_proba):
        m = y.size #计算损失
        p = y_proba * (2 * y - 1) + (1 - y)
        return -np.sum(np.log(p)) / m

    def _gradient(self, X, y, y_proba): #计算梯度
        return np.matmul(y_proba - y, X) / y.size

    def _gradient_descent(self, w, X, y): #梯度下降算法
        if self.tol is not None: #如果用户指定tol，则启用早期停止法
            loss_old = np.inf
        for step_i in range(self.n_iter): #使用梯度下降，至多迭代n_iter次，更新w
            y_proba = self._predict_proba(X, w) #预测所有点为1的概率
            loss = self._loss(y, y_proba) #计算损失
            print('%4i Loss: %s' % (step_i, loss))

            if self.tol is not None: #早期停止法
                if loss_old - loss < self.tol: #如果损失下降小于阈值，则终止迭代
                    break
                loss_old = loss
            grad = self._gradient(X, y, y_proba) #计算梯度
            w -= self.eta * grad #更新参数w

    def _preprocess_data_X(self, X): #数据预处理
        m, n = X.shape #扩展X，添加Xo列并设置为1
        X_ = np.empty((m, n + 1))
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_

    def train(self, X_train, y_train): #训练
        X_train = self._preprocess_data_X(X_train) #预处理X_train（添加Xo=1）
        _, n = X_train.shape #初始化参数向量w
        self.w = np.random.randint(n) * 0.05
        self._gradient_descent(self.w, X_train, y_train) #执行梯度下降训练w

    def predict(self, X): #预测
        X = self._preprocess_data_X(X) #预处理X_test（添加Xo=1）
        y_pred = self._predict_proba(X, self.w) #预测为整例(y=1)的概率
        return np.where(y_pred >= 0.5, 1, 0) #根据概率为预测类别，p>=0.5为正例，否则为负例




