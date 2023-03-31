"""假设某地某天的时段温度分别为[20,23,24,25,26,27,28,25,24,22,21,20]，编程使用preprocessing.scale()
函数对此数列进行标准化处理。"""
# from sklearn import preprocessing
#
# def temper(a):
#     prodict = a
#     return print(preprocessing.scale(prodict))
#
# if __name__ == "__main__":
#     temperature = [20, 23, 24, 25, 26, 27, 28, 25, 24, 22, 21, 20]
#     temper(temperature)

"""使用某模型对水果进行预测，真值为[1,0,0,1,1,0,0,1],预测结果为[0,1,1,1,1,1,0,1],
编程计算该模型的精确率、召回率和f1均值"""
# from sklearn.metrics import precision_score, recall_score, f1_score
#
# def precis_one(true_a, pre_b):
#     pre = precision_score(true_a, pre_b, average='micro')
#     return pre
#
# def recall_one(ture_c, pre_d):
#     rec = recall_score(ture_c, pre_d, average='micro')
#     return rec
#
# def f1_one(true_e, pre_f):
#     f1s = f1_score(true_e, pre_f, average='micro')
#     return f1s
#
# if __name__ == "__main__":
#     def_true = [1, 0, 0, 1, 1, 0, 0, 1]
#     def_pre = [0, 1, 1, 1, 1, 1, 0, 1]
#     print("精确率:", precis_one(def_true, def_pre))
#     print("召回率:", recall_one(def_true, def_pre))
#     print("f1均值:", f1_one(def_true, def_pre))
