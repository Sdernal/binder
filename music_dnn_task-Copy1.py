
# coding: utf-8

# In[2]:



import warnings
warnings.simplefilter("ignore")

import os
import sys 
import wave
import time
import librosa
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from StringIO import StringIO

get_ipython().magic(u'pylab inline')

# sudo pip install librosa
# ubuntu: sudo apt-get install libav-tools
# OSX   : brew install libav OMP_NUM_THREADS=2


# # Get the Data

# Датасет для контеста помещен вот здесь: https://yadi.sk/d/3jmeVCfSwTKgm (внимание! файл весит 3,5 Гб)
# 
# 
# Он распаковывается в папку "musicdata". Все пути в этом ноутбуке будут относительно этой папки: "./musicdata/\*". 
# 
# Вы можете изменить базовый путь, как вам удобно.

# In[1]:

BP = "./musicdata/"


# # Just a Sound

# Для начала попробуем поработать со звуком - послушайте любой трек, постройте график сырых данных и спектрограмму.

# In[3]:

import IPython
from IPython import display

def Audio(url):
    return display.HTML("<center><audio controls><source src='{}' type=\"audio/wav\"></audio>".format(url))


# In[4]:

sound_file = os.path.join(BP, "audiosamples", "short_206377.wav")
#sound_file = BP + "audiosamples/short_206377.wav"
y, sr = librosa.load(sound_file)

Audio(url=sound_file)


# In[5]:

x = np.zeros(y.size)
for i in range(1,size(x)):
    x[i] = x[i-1]+1./sr


# # Sound as 1D-Signal

# In[6]:

plt.figure(figsize=(20,4))
pylab.plot(x,y)
pylab.xlim([0, 10])
pylab.show()


# # Sound as 2D-Signal

# Нашу цель слелать из предыдущего графика картинку - для этого нам нужно оценить частоты в каждый момент времени. Благо за нас это умеет делать librosa, у которой внутри STFFT. Используйте librosa.feature.melspectrogram -- для получения спектрограммы, и librosa.logamplitude для выравнивания диапазонов частот. 
# 
# help: https://gist.github.com/mailletf/3484932dd29d62b36092

# In[7]:

S = librosa.feature.melspectrogram(y,sr=sr,n_mels=128)
log_S = librosa.logamplitude(S, ref_power=np.max)


# Нарисуйте спектрограмму, получилась красивая картинка?

# In[8]:

plt.figure(figsize=(12,4))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', cmap='hot')
plt.title('mel power spectrogram')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()


# # Prepare the Data

# В этой секции мы создадим датасет, прочитав все данные.

# In[3]:

files = os.listdir(os.path.join(BP,"spectrograms"))
sorted_files = sorted(files)
trainfiles = sorted_files[:-1]
print trainfiles


# In[ ]:

import numpy as np
print "loading file: ", trainfiles[0]
X = np.load(os.path.join(BP, "spectrograms", trainfiles[0]), mmap_mode = 'r')
for trainfile in trainfiles[1:]:
    print "loading file: ", trainfile
    X = np.append(X, np.load(os.path.join(BP, "spectrograms", trainfile), mmap_mode = 'r'), axis = 0)

print shape(X)


# Здесь вы должны получить train genres. Они идут подряд в соответствии с последовательностью треков в файлах с данными. Для содержимого первых шести файлов у вас есть жанры, а для последнего файла, вам их нужно предсказать.

# In[7]:

f_in = open(os.path.join(BP, "train_genres.txt"), "rt")
y_lines = []
for line in f_in:
    y_lines.append(line)
print size(y_lines)


# In[74]:

train_genres = y_lines[:2000]
# train_genres = y_lines
for i in range(len(train_genres)):
    train_genres[i] = train_genres[i][:-1]
genres_list = list(set(train_genres))
num_classes = len(genres_list)    
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(list(genres_list))
train = le.transform(train_genres)       


# In[75]:

print genres_list
print le.transform(genres_list)
print train_genres[:10]
print train[:10]


# # Nearest Neighbors genre classification

# Let's try to solve similarity task by the nearest neighbour approach.
# 
# <img src="./img/nn.png" width="300">

# Перемешаем датасет - используйте `np.random.permutation` и индексацию `a[np.random.permutation]`.

# In[13]:

y = train_genres
perm = np.random.permutation(len(y))
Xperm = X[perm]
y = train[perm]
# Xreshape = X.reshape(X.shape[0], X.shape[1], X.shape[2])


# Разобьем датасет на трейн и валидацию 80/20%.

# In[14]:

Xreshape = Xperm.reshape((Xperm.shape[0], Xperm.shape[1]* Xperm.shape[2]))
print shape(Xreshape)


# In[15]:


X_train, X_valid = Xreshape[:1800], Xreshape[1800:]
y_train, y_valid = y[:1800], y[1800:]


# Попробуем поклассифицировать жанры методом ближайших соседей.

# In[16]:

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_valid)
# y_val_pred = clf.predict(X_valid.reshape((X_valid.shape[0], -1)))

print accuracy_score(y_valid, y_pred)


# # Fully-Connected Neural Nets

# Использовать нейросети - это хорошая идея, давайте начнем с полносвязных сетей.

# In[17]:

import theano
import lasagne
import theano.tensor as T

from utils3 import train_net


# Перемешаем и разобьем датасет.

# In[18]:

perm = np.random.permutation(len(y))
X2, y = np.array(X)[perm].astype('float32'), np.array(train)[perm]
Xreshape = X2.reshape(X2.shape[0], 1, X2.shape[1], X2.shape[2])

X_train, X_valid = Xreshape[:1800], Xreshape[1800:]
y_train, y_valid = y[:1800], y[1800:]


# In[19]:

input_X, target_y = T.tensor4("X", dtype='float32'), T.vector("y", dtype='int32')
nn = lasagne.layers.InputLayer(shape=(None, 1, X2.shape[1], X2.shape[2]), input_var=input_X) 
nn = lasagne.layers.DenseLayer(nn, 64)
nn = lasagne.layers.DenseLayer(nn, 128)
nn  = lasagne.layers.DenseLayer(nn, num_classes)
y_predicted = lasagne.layers.get_output(nn)
all_weights = lasagne.layers.get_all_params(nn)

loss = lasagne.objectives.categorical_crossentropy(y_predicted, target_y).mean()
accuracy = lasagne.objectives.categorical_accuracy(y_predicted, target_y).mean()
updates_sgd = lasagne.updates.nesterov_momentum(loss,all_weights,0.9 )


# In[20]:

train_fun = theano.function([input_X, target_y], [loss, accuracy], allow_input_downcast=True, updates=updates_sgd)
test_fun  = theano.function([input_X, target_y], [loss, accuracy], allow_input_downcast=True)


# In[21]:

get_ipython().magic(u'time conv_nn = train_net(nn, train_fun, test_fun, X_train, y_train, X_valid, y_valid, num_epochs=10, batch_size=50)')


# # Convolution Neural Nets

# Да, вероятно, полносвязные сети не зажгли и kNN не обогнали. Давайте попробуем сверточные, хороший вариант делать одномерные свертки (сразу по всем частотам). Хотя двумерные тоже могут работать. 
# 
# Архитектуру можно подсмотреть [тут](http://benanne.github.io/2014/08/05/spotify-cnns.html).

# In[ ]:




# In[22]:

import theano
import lasagne
import theano.tensor as T

from utils3 import train_net


# In[23]:

perm = np.random.permutation(len(y))
X3, y = np.array(X)[perm].astype('float32'), np.array(train)[perm]
Xreshape = X3.reshape(X3.shape[0], X3.shape[1], X3.shape[2])

X_train, X_valid = Xreshape[:1800], Xreshape[1800:]
y_train, y_valid = y[:1800], y[1800:]


# In[25]:

input_X, target_y = T.tensor3("X", dtype='float64'), T.vector("y", dtype='int32')
nn = lasagne.layers.InputLayer(shape=(None, X.shape[1], X.shape[2]), input_var=input_X) 
nn = lasagne.layers.Conv1DLayer(nn, 4, 4)
nn = lasagne.layers.MaxPool1DLayer(nn, 2)
nn = lasagne.layers.DenseLayer(nn, 100)
# nn = ??? Сделайте свою сеть, используя: Conv1DLayer + MaxPool1DLayer + DenseLayer

nn = lasagne.layers.DenseLayer(nn, num_classes, nonlinearity=lasagne.nonlinearities.softmax)


# In[26]:

y_predicted = lasagne.layers.get_output(nn)
all_weights = lasagne.layers.get_all_params(nn)

loss = lasagne.objectives.categorical_crossentropy(y_predicted, target_y).mean()
accuracy = lasagne.objectives.categorical_accuracy(y_predicted, target_y).mean()
updates_sgd = lasagne.updates.adam(loss, all_weights)


# In[27]:

train_fun = theano.function([input_X, target_y], [loss, accuracy], allow_input_downcast=True, updates=updates_sgd)
test_fun  = theano.function([input_X, target_y], [loss, accuracy], allow_input_downcast=True)


# In[28]:

# Подумайте, какой размер батча нужен и сколько эпох вы можете себе позволить подождать
get_ipython().magic(u'time conv_nn = train_net(nn, train_fun, test_fun, X_train, y_train, X_valid, y_valid, num_epochs=10, batch_size=100)')


# Валидационная точность должна быть около 0.6 или лучше. 

# Давайте нарисуем каждый второй фильтр - можете ли вы проинтерпретировать эти фильтры?

# In[29]:

plt.figure(figsize=(5, 5), dpi=500)
W = lasagne.layers.get_all_params(nn)[0].get_value()
W[::2, :, :] = 0.2
W = np.hstack(W)
pylab.imshow(W, cmap='hot', interpolation="nearest")
pylab.axis('off')
pylab.show()


# # Maps of tracks by SVD and tSNE

# Хорошая идея посмотреть, как полученое представление отображается с сохранением относительных расстояний на плоскость. Используйте tSNE или PCA.
# 
# Help: https://lts2.epfl.ch/blog/perekres/category/visualizing-hidden-structures-in-datasets-using-deep-learning/

# In[30]:

from sklearn.manifold import TSNE


# In[31]:

represent = lasagne.layers.get_output(nn.input_layer)
represent_fun = theano.function([input_X], [represent], allow_input_downcast=True)


# In[32]:

f = lambda x: np.array(represent_fun([x])[0])
track_vectors = map(f, X_train) + map(f, X_valid)
track_vectors = np.concatenate(track_vectors, axis=0)

track_labels = np.array(list(y_train) + list(y_valid))


# получите 2d вектора

# In[34]:

tsne = TSNE()
X_tsne = tsne.fit_transform(track_vectors)


# нарисуйте получившиеся точки

# In[49]:

print shape(X_tsne)
print X_tsne[:10]


# In[61]:

print shape(track_labels)
print track_labels[:10]
print list(np.where(track_labels == 1))


# In[86]:

plt.figure(figsize=(10,10), dpi=500)
# colors = cm.hot(np.linspace(0, 1, num_classes))
colors = cm.rainbow(np.linspace(0, 1, num_classes))
for idx in range(num_classes):
    idx_= np.where(track_labels == idx)    
    if size(idx_)  >2 :
        pylab.scatter(X_tsne[idx_][0], X_tsne[idx_][1], c=colors[idx], label = genres_list[idx],cmap=cm.hot, s = 8*len(X_tsne[idx_]))
 
pylab.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=4)


# In[ ]:



