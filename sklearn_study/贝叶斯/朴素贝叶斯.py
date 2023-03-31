from pandas import * # 读取数据文件需要（如果不用其他方式读取数据的话）
from numpy import * # 机器学习最基本的库（如果你们装了anaconda，无需pip）
from sklearn.model_selection import * # 划分训练集和测试集需要的库（当然你也可以随机读取数据的几行作为训练集）
from sklearn.metrics import * # 分析模型性能需要的库（检验模型可不可用）
from sklearn.naive_bayes import * # 6大朴素贝叶斯分类器（工具）
from sklearn.datasets import * # 读取数据集的另一种方法（仅限内置）


Cosplay = read_csv(r'./PGE.csv', encoding='cp936', index_col=0) # 读取数据文件
# X=Cosplay[['where','selfdisp','profession']]
X = Cosplay[['where']] # 读取输入数据和输出数据
y = Cosplay['pass']
# X2=array(X)
# X2[:,0]=where(X2[:,0]>0.5,1,0)
# X2[:,1]=where(X2[:,1]>5,1,0)
# X2[:,2]=where(X2[:,2]>5,1,0)
# 模型建立过程
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=41)
c_predict = BernoulliNB(binarize=None)
cosornot = c_predict.fit(X_train, y_train)
# 预测并输出模型性能
cos_pred2 = c_predict.predict(X_test)
acc = accuracy_score(y_test, cos_pred2)
print(acc)
