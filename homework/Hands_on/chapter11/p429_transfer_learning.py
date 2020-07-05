from tensorflow import keras

model_A = keras.models.load_model('my_model_A.h5')
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation = 'sigmoid'))


# 모델 구조 복사 ( 가중치 복제 X)
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())


# : 새로운 출력층이 랜덤하세 초기화되어 있으므로 큰오차를 만듧(적어도 처음 몇 번의 epoch동안)
#   => 큰 오차 gradient가 재사용된 가중치를 망칠 수 있다.
# 처음 몇번의 epoch동안 재사용된 층을 동결하고 새로운 층에게 적절한 가중치를 학습할 시간을 주는 방법으로 해결
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False                   # 모든 층 동결

model_B_on_A.compile(loss = 'binary_crossentropy', optimizer = 'sgd',
                      metrics = ['accuracy'])

history = model_B_on_A.fit(x_train_B, y_train_B, epochs = 4,
                           validation_data = (x_val_B, y_val_B))

for layer in model_B_on_A.layers[:-1]:      
    layer.trainable = True                    # 동결 해제 후 다시 컴파일, 훈련

optimizer = keras.optimizers.SGD(lr = 1e-4)   # 학습률 낮추기 / 기본 학습률은 1e-2

model_B_on_A.compile(loss = 'binary_crossentropy', optimizer = optimizer,
                       metrics = ['accuracy'])

history = model_B_on_A.fit(x_train_B, y_train_B, epochs = 16,
                            validation_data = (x_val_B, y_val_B))

model_B_on_A.evaluate(x_test_B, y_test_B)