from sklearn import svm
from sklearn import datasets
import numpy as np

#读取数据集
f = open('sonar.all-data')
sonor_datasets = np.zeros([208, 60])
sonor_target = np.arange(0, 208, 1, dtype=int)
for i in range(208):
    line = f.readline()
    line = line.strip('\n')
    line = line.split(',')
    for j in range(59):
        sonor_datasets[i, j] = line[j]
    if line[60] == 'R':
        sonor_target[i] = 1
    else:
        sonor_target[i] = 2
sonor = {'data':sonor_datasets, 'target':sonor_target}
f.close()

#取训练样本和测试样本
all = sonor['data']
alltar = sonor['target']
train = np.zeros([158, 60])
test = np.zeros([50, 60])
traintar = np.arange(0, 158, 1, dtype=int)
testtar = np.arange(0, 50, 1, dtype=int)
n1 = 0
n2 = 0
for i in range(208):
    if i > 0 and i <= 200 and i % 4 == 0:
        test[n1, :] = all[i, :]
        testtar[n1] = alltar[i]
        n1 = n1 + 1
    else:
        train[n2, :] = all[i, :]
        traintar[n2] = alltar[i]
        n2 = n2 + 1

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