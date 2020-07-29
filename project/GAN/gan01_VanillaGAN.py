# [참고] https://tykimos.github.io/2017/12/12/One_Slide_GAN/

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, _),(_, _) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5

x_train = x_train.reshape(60000, 784)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LeakyReLU, Dropout, Input
from tensorflow.keras.optimizers import Adam
np.random.seed(1000)
randomDim = 10

adam = Adam(lr=0.0002, beta_1=0.5)

# Generator
g = Sequential()
g.add(Dense(256, input_dim = randomDim))    # randomDim : 노이즈의 입력 크기
g.add(LeakyReLU(0.2))
g.add(Dense(512))
g.add(LeakyReLU(0.2))
g.add(Dense(1024))
g.add(LeakyReLU(0.2))
g.add(Dense(784, activation = 'tanh'))  # image 생성

g.summary()

# Discriminator
d = Sequential()
d.add(Dense(1024, input_dim = 784))
d.add(LeakyReLU(0.2))
d.add(Dropout(0.3))
d.add(Dense(512))
d.add(LeakyReLU(0.2))
d.add(Dropout(0.3))
d.add(Dense(256))
d.add(LeakyReLU(0.2))
d.add(Dropout(0.3))
d.add(Dense(1, activation = 'sigmoid')) 
d.compile(loss = 'binary_crossentropy', optimizer = adam)     # trainable = True

d.summary()

# GAN model
d.trainable = False     # 판별기의 가중치 고정
ganInput = Input(shape=(randomDim, ))
x = g(ganInput)
ganOutput = d(x)
gan = Model(inputs=ganInput, outputs = ganOutput)
gan.compile(loss = 'binary_crossentropy', optimizer = adam)   # Discriminator.trainable = False

dLosses = []
gLosses = []

# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./gan_loss_epoch_%d.png' % epoch)

# Create a wall of generated MNIST images
def saveGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = g.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./gan_generated_image_epoch_%d.png' % epoch)

def train(epochs = 1, batch_size = 128):
    batchCount = int(x_train.shape[0] / batch_size)
    print('Epochs :', epochs)
    print('Batch size :', batch_size)
    print('Batches per epoch :', batchCount)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d'%e, '-'*15)
        for _ in range(batchCount):
            # 랜덤한 입력 노이즈와 이미지를 얻는다.
            noise = np.random.normal(0, 1, size = [batch_size, randomDim])  # mean(평균) = 0, std(표준편차) = 1
            imageBatch = x_train[np.random.randint(0, x_train.shape[0], size = batch_size)]


            # 가짜 MNIST이미지 생성
            generatorImages = g.predict(noise)
            # np.shape(imageBatch), np.shape(generatorImages) 출력
            X = np.concatenate([imageBatch, generatorImages])
                                # 실제          거짓

            # 생성된 것과 실제 이미지의 레이블
            yDis = np.zeros(2*batch_size)
            # 편파적 레이블의 평활화
            yDis[:batch_size] = 0.9

            # 판별기 훈련
            d.trainable = True
            dloss = d.train_on_batch(X, yDis)   # train_on_batch : 하나의 데이터 배치에 대해서 경사 업데이트를 1회 실시

            # 생성기 훈련
            noise = np.random.normal(0, 1, size = [batch_size, randomDim])
            yGen = np.ones(batch_size)
            d.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

        dLosses.append(dloss)
        gLosses.append(gloss)

        if e == 1 or e%20 == 0:
            saveGeneratedImages(e)

    # Plot losses from every epoch
    plotLoss(e)

# 실행
train(200, 128)