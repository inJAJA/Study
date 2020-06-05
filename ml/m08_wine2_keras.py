import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:/Study/data/csv/winequality-white.csv', index_col = None, header = 0, sep =  ';')

print(dataset.quality.value_counts(dropna = False))
# 6    2198
# 5    1457
# 7     880
# 8     175
# 4     163
# 3      20
# 9       5
# Name: quality, dtype: int64

np_dataset = dataset.values

print(np_dataset.shape) # (4898, 12)

x = np_dataset[:,:-1]
y = np_dataset[:, -1]

# one hot
y = to_categorical(y)

print(x.shape)            # (4898, 11)
print(y.shape)            # (4898, 10)


# scaler
scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# PCA
# pca = PCA(n_components= 10)
# pca.fit(x)
# x = pca.transform(x)

# train_test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 33, train_size = 0.8)

print(x_train.shape)      # (3918, 11)
print(y_train.shape)      # (3918, 10)

#2. model
model = Sequential()
model.add(Dense(30, input_dim = 11, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(80, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(150, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

# EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience= 50)

#3. fit
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 10, batch_size= 64, 
        validation_split = 0.2)

#4. predict
loss, acc = model.evaluate(x_test, y_test, batch_size = 64)
print('loss: ', loss)
print('acc: ', acc)

