from sklearn.datasets import load_digits #手写数字数据集
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

X, y = load_digits(return_X_y=True)#预加载
# digits = load_digits()#导入数据（预加载）
# n_samples, n_feature = digits.data.shape #数据集的样本和特征
# print(f'共有:{n_samples}个样本数据，数据有{n_feature}个特征（数据集的大小为:{n_samples}X{n_feature}）')
# print(f'数据集的形状为:{digits.images.shape}')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# print(X_train.shape, X_test.shape)#查看数据集和训练集的大小，test_size=0.1为（1617,64）和（180,64）；0.2为(1437, 64)和(360, 64)
'''
liblinear：一般适用于小数据集
sag、saga：一般适用于大数据集，速度更快
newton-cg、lbfgs：中等数据集
max_iter = 100：最大迭代次数 
n_jobs：表示使用处理器的数量，内部进行异步处理，等于 -1 表示全部使用，一般可以给CPU数量或者2倍
'''
dlf = LogisticRegression(C=1.0, solver='lbfgs', n_jobs=-1).fit(X_train, y_train)#C为惩罚系数，越大拟合效果越好，反之（但要根据实际）
print('coef:\n', dlf.coef_)
print('intercept:\n', dlf.intercept_)
print('predict first two:\n', dlf.predict(X_train[:3, :]))#预测概率选择范围
print('classification score:\n', dlf.score(X_train, y_train))#预测得分
predict_y = dlf.predict(X_test)#预测
print('classfication report:\n', metrics.classification_report(y_test, predict_y))
