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
clf = svm.SVC()
clf.fit(train, traintar)
score = clf.score(test, testtar)
print('模型测试得分：', score)