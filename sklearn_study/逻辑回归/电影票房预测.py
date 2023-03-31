from sklearn.linear_model import * #导入所有模型
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt #导入mlp绘图库
import numpy as np #导入numpy科学库重命名为np，这是机器学习所用到的基本库
import pandas as pd #导入pandas数据分析库重命名为pd，这也是机器学习所用到的基本库
import matplotlib #导入mlp
import warnings #导入警告库，目的是消除红色警告


def drawPlt(): #自定义绘图库
    plt.figure(figsize=(10, 6)) #定义一个画板
    plt.title('票房收入（单位:百万元）') #总标题
    plt.xlabel('成本') #x轴标题
    plt.ylabel('收入') #y轴标题
    plt.axis([0, 25, 0, 60]) #设置x轴为0-25和y轴为0-60
    plt.grid(True) #显示网格线

def main(num):
    global num_d, wan  # 定义局部全局变量
    if num >= 10: #如果输入的数大于等于10
        num_d = int(num/10) #强转换为整数
        wan = '千万' #千万单位
    elif num < 10:
        num_d = num #如果输入的数是个位就不用转换
        wan = '百万' #百万单位
    matplotlib.use('TkAgg') #定义在运行程序时不同时开启绘图（注释则为同时开启绘图）
    warnings.filterwarnings('ignore')  # 忽视在程序中红色警告（不为错，但只是为了控制台显示美观）
    matplotlib.rcParams['font.family'] = 'SimHei'  # 指定中文黑体字体
    matplotlib.rcParams['font.size'] = 10  # 设置字体大小
    matplotlib.rcParams['axes.unicode_minus'] = False  # false修正坐标轴上负号（-）显示方块的问题
    df = pd.read_csv('cinema.csv') #导入文件
    X = df['cost'].values.reshape(-1, 1) #将成本（cost）转至为一列
    y = df['income'].values.reshape(-1, 1) #将收入（income）转至为一列
    model = LinearRegression() #导入线性回归模型
    model.fit(X, y) #拟合X和y
    pre = model.predict([[num]]) #预测模型
    coef = model.coef_ #特征系数
    inter = model.intercept_ #截距
    print('投资{}{}的电影预计票房收入为:{:.2f}百万元'.format(num_d, wan, pre[0][0])) #分别为数字，单位，预测值
    print("回归模型的系数是", coef[0][0])# 因为coef出来的数为[[num]],所以要用[0][0]取出
    print("回归模型的截距是", inter[0]) #intercept得出来的数为[num],用[0]取出
    print(f"最佳拟合线:y={int(inter)} + {int(coef)}x") #回归线定义，用整数
    drawPlt() #调用自定义绘图函数
    plt.scatter(df['cost'], df['income'], c='green', marker='^') #将X和y用三角的方式呈现,散点图
    s, j = inter[0], (num*coef[0][0]+inter[0]) #s为截距，j为数成特征系数+截距得到的值
    plt.plot([0, num], [s, j]) #拟合直线
    plt.show() #画图

if __name__ == '__main__':
    a = eval(input("输入投入成本(百万):")) #输入值
    main(a) #输入数
