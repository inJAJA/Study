from tensorflow import keras
import numpy as np

# Monte_Carlo_dropout(몬테 카를로 드롭아웃)
# 모델을 재후년하거나 전혀 수정하지 않고 성능을 크게 향상 시킬 수 있는 드롭 아웃

y_probas = np.stack([model(x_test_scaled, training=True)  # training=True : Dropout층 활성화
                    for sample in range(100)])
                                  # 100번의 예측 만들어 쌓음

y_proba = y_probas.mean(axis = 0)

np.round(y_probas[:, :1], 2)