import os
import sys 
import wave
import time
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from StringIO import StringIO


# In[3]:

BP = "/home/ubuntu/ml-mipt-part2/2016/contest/01_music/task_descr/musicdata"


# In[10]:

files = os.listdir(os.path.join(BP,"spectrograms"))
sorted_files = sorted(files)
trainfiles = sorted_files[:-1]
print trainfiles


# In[13]:

import numpy as np
print "loading file: ", trainfiles[0]
X = np.load(os.path.join(BP, "spectrograms", trainfiles[0]), mmap_mode = 'r')
for trainfile in trainfiles[1:]:
    print "loading file: ", trainfile
    X = np.append(X, np.load(os.path.join(BP, "spectrograms", trainfile), mmap_mode = 'r'), axis = 0)

print shape(X)


# Здесь вы должны получить train genres. Они идут подряд в соответствии с последовательностью треков в файлах с данными. Для содержимого первых шести файлов у вас есть жанры, а для последнего файла, вам их нужно предсказать.

# In[14]:

f_in = open(os.path.join(BP, "train_genres.txt"), "rt")
y_lines = []
for line in f_in:
    y_lines.append(line)
print size(y_lines)


# In[ ]:

train_genres = y_lines
for i in range(len(train_genres)):
    train_genres[i] = train_genres[i][:-1]
genres_list = list(set(train_genres))
num_classes = len(genres_list)    
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(genres_list)
train = le.transform(train_genres)


# In[ ]:

y = train_genres
perm = np.random.permutation(len(y))
Xperm = X[perm]
y = train[perm]


# In[ ]:

Xreshape = Xperm.reshape((Xperm.shape[0], Xperm.shape[1]* Xperm.shape[2]))
print shape(Xreshape)


# In[ ]:

Xtest = np.load(os.path.join(BP, "spectrograms", trainfiles[0]), mmap_mode = 'r')
Xtest = Xtest.reshape((Xtest.shape[0], Xtest.shape[1]* Xtest.shape[2]))


# In[ ]:

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf = clf.fit(Xreshape, y)
y_pred = clf.predict(Xtest)
# y_val_pred = clf.predict(X_valid.reshape((X_valid.shape[0], -1)))

# print accuracy_score(y_valid, y_pred)


# In[ ]:

y_pred = np.vectorize(lambda x: genres_list[x].strip())(y_pred)
df = pd.DataFrame(y_pred, index=np.arange(12000, 14000))
df.to_csv('knn.csv', header=['Category'])


# In[ ]:

# import theano
# import lasagne
# import theano.tensor as T

# from utils3 import train_net


# In[ ]:

# perm = np.random.permutation(len(y))
# X3, y = np.array(X)[perm].astype('float32'), np.array(train)[perm]
# Xreshape = X3.reshape(X3.shape[0], X3.shape[1], X3.shape[2])

# X_train, X_valid = Xreshape[:10000], Xreshape[10000:]
# y_train, y_valid = y[:10000], y[10000:]


# In[ ]:

# input_X, target_y = T.tensor3("X", dtype='float64'), T.vector("y", dtype='int32')
# nn = lasagne.layers.InputLayer(shape=(None, X.shape[1], X.shape[2]), input_var=input_X) 
# nn = lasagne.layers.Conv1DLayer(nn, 4, 4)
# nn = lasagne.layers.MaxPool1DLayer(nn, 2)
# nn = lasagne.layers.DenseLayer(nn, 100)
# # nn = ??? Сделайте свою сеть, используя: Conv1DLayer + MaxPool1DLayer + DenseLayer

# nn = lasagne.layers.DenseLayer(nn, num_classes, nonlinearity=lasagne.nonlinearities.softmax)


# In[ ]:

# y_predicted = lasagne.layers.get_output(nn)
# all_weights = lasagne.layers.get_all_params(nn)

# loss = lasagne.objectives.categorical_crossentropy(y_predicted, target_y).mean()
# accuracy = lasagne.objectives.categorical_accuracy(y_predicted, target_y).mean()
# updates_sgd = lasagne.updates.adam(loss, all_weights)


# In[ ]:

# train_fun = theano.function([input_X, target_y], [loss, accuracy], allow_input_downcast=True, updates=updates_sgd)
# test_fun  = theano.function([input_X, target_y], [loss, accuracy], allow_input_downcast=True)


# In[ ]:

# %time conv_nn = train_net(nn, train_fun, test_fun, X_train, y_train, X_valid, y_valid, num_epochs=10, batch_size=100)


# In[ ]:

# X7 = np.load(os.path.join(BP, "spectrograms",sorted_files[-1]), mmap_mode = 'r')


# In[ ]:

# def make_csv(X, net):
#     prediction = lasagne.layers.get_output(net, deterministic=True)
#     predict_function = theano.function([input_X], prediction)
#     y_valid_pred = predict_function(X_valid)
#     y_pred = np.zeros(y_valid_pred.shape[0], dtype=np.int8)
#     for i in range(y_valid_pred.shape[0]):
#         y_pred[i] = int(np.argmax(y_valid_pred[i]))
#     print y_pred
#         y_labels = le.inverse_transform(y_pred)

#     print 'Id,Category'
#     for i in range(X.shape[0]):
#         print (str(i+1) + ',' + y_labels[i])


# In[ ]:

# make_csv(X7, nn)

