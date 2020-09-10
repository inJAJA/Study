from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_shape = (1, )))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

from keras.utils import plot_model
plot_model(model, "D:/model.png", show_shapes=True)
'''
# Window
1. pip install pydot
2. pip install pydotplus
2. pip install graphviz

# 에러 발생시 다음의 링크 참고
 : https://ndlessrain.tistory.com/entry/graphviz-pathpydot-failed-to-call-GraphViz
 
# Linux
1. pip install pydot>=1.2.4
2. sudo apt-get install graphviz
'''
