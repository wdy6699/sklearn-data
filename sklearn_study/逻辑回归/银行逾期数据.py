import numpy as np #导入numpy科学库，重命名为np
import pandas as pd #导入pandas数据分析库，重命名为pd
import matplotlib #导入mlb库
import matplotlib.pyplot as plt #导入mlb绘图库重命名为plt
import warnings #导入警告库，目的是消除红色警告


def sigmoid(z):  # logistic回归函数 把值放缩到0 1之间
    return 1. / (1. + np.exp(-z)) #函数公式

def loss(h, y):  # 损失函数
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() #计算方法

def gradient(X, h, y):  # 梯度下降
    return np.dot(X.T, (h - y) / y.shape[0]) #梯度下降的计算公式

def LogisticRegress(x, y, lr, n_iter):  # 逻辑回归函数，n_iter为迭代次数，lr为学习率
    global l #定义logistic的全局变量
    intercept = np.ones((x.shape[0], 1)) #得到为输入的x值全为1的值[1. ]（初始化截距为1）
    x = np.concatenate((intercept, x), axis=1) #处理得到原数据加1.的数据 [1. 1.86 4.39]
    w = np.zeros(x.shape[1]) #w为x的全为0的值[0. 0. 0.]（初始化参数为0）
    for i in range(n_iter):  # 梯度下降迭代
        z = np.dot(x, w)  # 线性函数，计算x与w的内积
        h = sigmoid(z) #logistic函数
        g = gradient(x, h, y) #执行梯度下降计算（通过学习率lr计算步长并执行梯度下降）
        w -= lr * g #获取参数w（更新参数）
        z = np.dot(x, w) #计算x与w的内积（更新参数到原线性函数中）
        h = sigmoid(z) #计算logistic函数值
        l = loss(h, y) #计算损失函数值
    return l, w #返回迭代后的梯度和参数

def lossPlot(x, y, lr, n_iter):
    intercept = np.ones((x.shape[0], 1))
    x = np.concatenate((intercept, x), axis=1)  # 处理得到原数据加1.的数据 [1. 1.86 4.39]
    w = np.zeros(x.shape[1])  # w为x的全为0的值[0. 0. 0.]（初始化参数为0）
    l_list = [] #保存损失函数值
    for i in range(n_iter):  # 梯度下降迭代
        z = np.dot(x, w)  # 线性函数，计算x与w的内积
        h = sigmoid(z)  # logistic函数
        g = gradient(x, h, y)  # 执行梯度下降计算（通过学习率lr计算步长并执行梯度下降）
        w -= lr * g  # 获取参数w（更新参数）
        z = np.dot(x, w)  # 计算x与w的内积（更新参数到原线性函数中）
        h = sigmoid(z)  # 计算logistic函数值
        l = loss(h, y)  # 计算损失函数值
        l_list.append(l) #添加损失值
    return l_list # 返回损失值列表

def initialise():  # 定义初始化
    matplotlib.use('TkAgg')  # 定义在运行程序时不同时开启绘图（注释则为同时开启绘图）
    warnings.filterwarnings('ignore')  # 忽视在程序中红色警告（不为错，但只是为了控制台显示美观）
    hd = pd.read_csv(r"credit-overdue.csv")  # 读取csv文件
    print(hd.head(10))  # 输出CSV前十行的数据

    matplotlib.rcParams['font.family'] = 'SimHei'  # 指定中文黑体字体
    matplotlib.rcParams['font.size'] = 10  # 设置字体大小
    matplotlib.rcParams['axes.unicode_minus'] = False  # false修正坐标轴上负号（-）显示方块的问题

    plt.figure(figsize=(10, 6))  # 设置图形大小（长X高）
    plt.title('欠债和收入散点图')
    map_sizes = {0: 100, 1: 100} #设定散点大小尺寸
    sizes = list(map(lambda a: map_sizes[a], hd['overdue']))  # 根据0和1赋值为20和100的范围
    plt.scatter(hd['debt'], hd['income'], s=sizes, c=hd['overdue'],
                marker='.')  # 绘制散点图scatter(x,y,s=,c=, marker=marker) c=是指定颜色，或者要设置的分区，这里是overdue的值为分界
    plt.show()  # 展示散点图
    '''
    以上的就为模型的建立和回归模型参数的确定
    以下就为绘制模型的分类线，测试模型的性能
    '''
    x = hd[['debt', 'income']].values #得到负债和收入的值
    y = hd['overdue'].values #得到过期未付款(0和1)的值
    loss_y, gar_y = LogisticRegress(x, y, lr=0.001, n_iter=10000) #得到损失函数和梯度下降函数的值
    print(f"损失函数值:{loss_y}\n梯度下降值:{gar_y}")

    plt.figure(figsize=(10, 6))
    plt.title('回归模型图')
    map_size = {0: 100, 1: 100}
    size = list(map(lambda b: map_size[b], hd['overdue']))
    plt.scatter(hd['debt'], hd['income'], s=size, c=hd['overdue'], marker='^')# 将标记换为三角以便区分第一张图
    x1_min, x1_max = hd['debt'].min(), hd['debt'].max() #自变量debt（得到最小值和最大值）
    x2_min, x2_max = hd['income'].min(), hd['income'].max() #自变量income（得到最小值和最大值）
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max)) #生成坐标矩阵（linspace为生成连续数据）
    grid = np.c_[xx1.ravel(), xx2.ravel()] #将xx1和xx2变成一维（ravel）且拼接在一起（np.c_）
    L = LogisticRegress(x, y, lr=0.01, n_iter=10000)
    probs = (np.dot(grid, np.array([L[1][1:3]]).T) + L[1][0]).reshape(xx1.shape) #求斜率
    plt.contour(xx1, xx2, probs, levels=[0], linewidths=1, colors='red') #回归线
    plt.show()

    plt.figure(figsize=(10, 6))
    l_y = lossPlot(x, y, lr=0.01, n_iter=30000)
    plt.plot([i for i in range(len(l_y))], l_y) #绘制曲线，按照损失值绘制
    plt.xlabel('迭代次数')
    plt.ylabel('损失函数值')
    plt.show()

if __name__ == '__main__':
    initialise() #调用主函数
