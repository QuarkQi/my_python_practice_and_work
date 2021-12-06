import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
sonar=pd.read_csv("D:\\Sonar.csv",header=None)
sonar_data=sonar.values[0:208,0:61]
# 训练集
# 第一类取0：60为训练集 ，第二类取97：180为训练集
sonar_train_data=sonar_data[range(0,61),0:61]
sonar_train_data=np.vstack((sonar_train_data,sonar_data[range(97,180),0:61]))
sonar_train_data=np.array(sonar_train_data)
sonar_train_label=sonar_train_data[:,0]
sonar_train_data=sonar_train_data[:,1:61]
print(sonar_train_data)
print(sonar_train_label)

# 测试集
# 第一类取61：97为测试集 ，第二类取180：208为测试集
sonar_test_data=sonar_data[range(61,97),0:61]
sonar_test_data=np.vstack((sonar_test_data,sonar_data[range(180,208),0:61]))
sonar_test_data=np.array(sonar_test_data)
sonar_test_label=sonar_test_data[:,0]
sonar_test_data=sonar_test_data[:,1:61]
print(sonar_test_data)
print(sonar_test_label)
b = []
a = []
for num in range(1,11):
    clf = svm.SVC(C=num/10, kernel='rbf', decision_function_shape='ovr')
    rf = clf.fit(sonar_train_data, sonar_train_label)
    #C:惩罚函数，默认是1（相当于惩罚松弛变量）
    #kernel：核函数，默认是rbf高斯核，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    #ovr：一对多
    c = clf.score(sonar_train_data,sonar_train_label)
    print("训练次数：", num, "交叉验证验证准确率：", c)
    a.append(c)   #交叉验证准确率
    c = clf.score(sonar_test_data,sonar_test_label)
    print("训练次数：", num, "测试集准确率：", c)
b.append(c)  #测试集准确率
plt.figure(1)
plt.plot(range(1,11),a)
plt.grid()
plt.xlabel('C/10')
plt.ylabel('acc')

iris = pd.read_csv("D:\\iris.csv", header=0)
# 数据集分类，分训练集和测试集
iris_data=iris.values[0:150,0:6]
iris_data=np.array(iris_data[0:150,0:6])
# 训练集
iris_train_data=iris_data[range(0,30),0:6]
iris_train_data=np.vstack((iris_train_data,iris_data[range(50,80),0:6]))
iris_train_data=np.vstack((iris_train_data,iris_data[range(100,130),0:6]))
iris_train_data=np.array(iris_train_data)
iris_train_label=iris_train_data[:,5]
iris_train_data=iris_train_data[:,0:4]
iris_train_data=iris_train_data.astype('float64')
iris_train_label=iris_train_label.astype('float64')
print(iris_train_data.shape)
print(iris_train_label.shape)

# 测试集
iris_test_data=iris_data[range(30,50),0:6]
iris_test_data=np.vstack((iris_test_data,iris_data[range(80,100),0:6]))
iris_test_data=np.vstack((iris_test_data,iris_data[range(130,149),0:6]))
iris_test_data=np.array(iris_test_data)
iris_test_label=iris_test_data[:,5]
iris_test_data=iris_test_data[:,0:4]
iris_test_data=iris_test_data.astype('float64')
iris_test_label=iris_test_label.astype('float64')
print(iris_test_data.shape)
print(iris_test_label.shape)
b = []
a = []
for num in range(1,11):
    clf = svm.SVC(C=num/10, kernel='rbf', decision_function_shape='ovr')
    rf = clf.fit(iris_train_data, iris_train_label)
    #C:惩罚函数，默认是1（相当于惩罚松弛变量）
    #kernel：核函数，默认是rbf高斯核，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    #ovr：一对多
    c = clf.score(iris_train_data,iris_train_label)
    print("训练次数：", num, "交叉验证验证准确率：", c)
    a.append(c)   #交叉验证准确率
    c = clf.score(iris_test_data,iris_test_label)
    print("训练次数：", num, "测试集准确率：", c)
b.append(c)  #测试集准确率
plt.figure(4)
plt.subplot(1,2,1)
plt.plot(range(1,11),a)
plt.grid()
plt.xlabel('C/10')
plt.ylabel('acc')
plt.subplot(1,2,2)
plt.plot(range(1,11),b)
plt.grid()
plt.xlabel('C/10')
plt.ylabel('acc')

b = []
a = []
for num in range(1,11):
    clf = svm.SVC(C=num/10, kernel='linear', decision_function_shape='ovr')
    rf = clf.fit(iris_train_data, iris_train_label)
    #C:惩罚函数，默认是1（相当于惩罚松弛变量）
    #kernel：核函数，默认是rbf高斯核，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    #ovr：一对多
    c = clf.score(iris_train_data,iris_train_label)
    print("训练次数：", num, "交叉验证验证准确率：", c)
    a.append(c)   #交叉验证准确率
    c = clf.score(iris_test_data,iris_test_label)
    print("训练次数：", num, "测试集准确率：", c)
b.append(c)  #测试集准确率
plt.figure(4)
plt.subplot(1,2,1)
plt.plot(range(1,11),a)
plt.grid()
plt.xlabel('C/10')
plt.ylabel('acc')
plt.subplot(1,2,2)
plt.plot(range(1,11),b)
plt.grid()
plt.xlabel('C/10')
plt.ylabel('acc')

b = []
a = []
for num in range(1,11):
    clf = svm.SVC(C=num/10, kernel='poly', decision_function_shape='ovr')
    rf = clf.fit(iris_train_data, iris_train_label)
    #C:惩罚函数，默认是1（相当于惩罚松弛变量）
    #kernel：核函数，默认是rbf高斯核，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    #ovr：一对多
    c = clf.score(iris_train_data,iris_train_label)
    print("训练次数：", num, "交叉验证验证准确率：", c)
    a.append(c)   #交叉验证准确率
    c = clf.score(iris_test_data,iris_test_label)
    print("训练次数：", num, "测试集准确率：", c)
b.append(c)  #测试集准确率
plt.figure(5)
plt.subplot(1,2,1)
plt.plot(range(1,11),a)
plt.grid()
plt.xlabel('C/10')
plt.ylabel('acc')
plt.subplot(1,2,2)
plt.plot(range(1,11),b)
plt.grid()
plt.xlabel('C/10')
plt.ylabel('acc')
