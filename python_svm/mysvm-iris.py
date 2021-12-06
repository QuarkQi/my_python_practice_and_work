from sklearn import svm
from sklearn import datasets
import numpy as np

#读取数据集
iris = datasets.load_iris()

#取训练样本和测试样本
all = iris['data']
alltar = iris['target']
train = np.zeros([120, 4])
test = np.zeros([30, 4])
traintar = np.arange(0, 120, 1, dtype=int)
testtar = np.arange(0, 30, 1, dtype=int)
n1 = 0
n2 = 0
for i in range(150):
    if i > 0 and i <= 90 and i % 3 == 0:
        test[n1, :] = all[i, :]
        testtar[n1] = alltar[i]
        n1 = n1 + 1
    else:
        train[n2, :] = all[i, :]
        traintar[n2] = alltar[i]
        n2 = n2 + 1
    #print(n1, n2)

#svm训练和测试评分
c = 0.75
clf_lin = svm.SVC(C=c, kernel='linear')
clf_rbf = svm.SVC(C=c, kernel='rbf')
clf_pol = svm.SVC(C=c, kernel='poly')
clf_lin.fit(train, traintar)
clf_rbf.fit(train, traintar)
clf_pol.fit(train, traintar)
score_lin = clf_lin.score(test, testtar)
score_rbf = clf_rbf.score(test, testtar)
score_pol = clf_pol.score(test, testtar)
print('linear测试得分：', score_lin)
print('rbf测试得分：', score_rbf)
print('ploy测试得分：', score_pol)