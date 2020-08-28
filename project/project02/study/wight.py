from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import gc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
seed = 66

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

 

from keras.initializers import glorot_normal, glorot_uniform, he_normal, he_uniform
initializer = he_normal(seed=None)
m = 'glorot_normal'

activation = 'sigmoid'

# model
model = Sequential()
# model.add(Dense(100, activation = 'sigmoid', input_dim = 1))
# model.add(Dense(100, activation = 'sigmoid'))
# model.add(Dense(100, activation = 'sigmoid'))
# model.add(Dense(100, activation = 'sigmoid'))
# model.add(Dense(1, activation = 'sigmoid'))

model.add(Dense(1000, activation = 'sigmoid', input_dim = 1, kernel_initializer=initializer))
model.add(Dense(500, activation = 'sigmoid', kernel_initializer=initializer))
model.add(Dense(250, activation = 'sigmoid', kernel_initializer=initializer))
model.add(Dense(100, activation = 'sigmoid', kernel_initializer=initializer))
model.add(Dense(1, activation = 'sigmoid', kernel_initializer=initializer))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# model.fit(x, y, epochs = 10, batch_size = 1)

def plot_weights(n, title):
    '''Plot weights of model layers'''
    for i, layer in enumerate(model.layers):
        ax = fig.add_subplot(gs[n, i])
        weights = layer.get_weights()
        print(weights)
        print('================')
        if weights:
            ax.hist(weights[0].flatten(), bins = 20, label = layer.name, alpha = 0.5)
        plt.xlim([-3.5, 3.5])
        plt.legend()
        plt.title(title)
    plt.yscale('log')

# Creating plot of layers weights before train process
fig = plt.figure(figsize = (19, 12))
gs = gridspec.GridSpec(2, 5)

plot_weights(0, f'{m} initial weights')

model.fit(x, y, epochs = 20, batch_size=1, verbose = 0)

plot_weights(1, f'{m} learned weights')

plt.show()