from tensorflow import keras

# 맥스-노름 규제(max-norm regularization)
# : (배치 정규화를 사용하지 않았을 때) 불안정한 gradient 문제를 완화하는데 도움을 줄 수 있음

keras.layers.Dense(100, activation = 'elu', kernel_initializer = 'he_normal',
                        kernel_constraint= keras.constraints.max_norm(1.))  
                                         # bias_constraint 매개변수 사용하여 편향 규제 가능