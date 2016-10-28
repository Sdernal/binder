import os
import sys 
import wave
import time
import numpy as np
import pandas as pd

from StringIO import StringIO

BP = "/home/ubuntu/ml-mipt-part2/2016/contest/01_music/task_descr/musicdata"

files = os.listdir(os.path.join(BP,"spectrograms"))
sorted_files = sorted(files)
trainfiles = sorted_files[:-1]
print trainfiles

import numpy as np
print "loading file: ", trainfiles[0]
X = np.load(os.path.join(BP, "spectrograms", trainfiles[0]), mmap_mode = 'r')
for trainfile in trainfiles[1:]:
    print "loading file: ", trainfile
    X = np.append(X, np.load(os.path.join(BP, "spectrograms", trainfile), mmap_mode = 'r'), axis = 0)

print shape(X)

f_in = open(os.path.join(BP, "train_genres.txt"), "rt")
y_lines = []
for line in f_in:
    y_lines.append(line)
print size(y_lines)

train_genres = y_lines
for i in range(len(train_genres)):
    train_genres[i] = train_genres[i][:-1]
genres_list = list(set(train_genres))
num_classes = len(genres_list)    
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(genres_list)
train = le.transform(train_genres)

y = train_genres
perm = np.random.permutation(len(y))
Xperm = X[perm]
y = train[perm]

Xreshape = Xperm.reshape((Xperm.shape[0], Xperm.shape[1]* Xperm.shape[2]))
print shape(Xreshape)

Xtest = np.load(os.path.join(BP, "spectrograms", trainfiles[0]), mmap_mode = 'r')
Xtest = Xtest.reshape((Xtest.shape[0], Xtest.shape[1]* Xtest.shape[2]))

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf = clf.fit(Xreshape, y)
y_pred = clf.predict(Xtest)

y_pred = np.vectorize(lambda x: genres_list[x].strip())(y_pred)
df = pd.DataFrame(y_pred, index=np.arange(12000, 14000))
df.to_csv('knn.csv', header=['Category'])

