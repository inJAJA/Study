from keras.preprocessing.text import Tokenizer
import numpy as np 

docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화에요',           # x
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요','너무 재미없다', '참 재밋네요']
    
# 긍정 1, 부정 0
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])              # y

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# => 많이나온 순서대로 앞에 위치
# {'참': 1, '너무': 2, '재밋어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, 
#  '싶은': 9, '영화입니다': 10, '한번': 11, '더': 12, '보고': 13, '싶네요': 14, '글쎄요': 15, '별로에요': 16, 
#  '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밋네요': 23}


x = token.texts_to_sequences(docs)   # 문자 -> 수치화
print(x)                             # shape가 모두 다르다.
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23]]


from keras.preprocessing.sequence import pad_sequences    # defalut : padding='pre', value=0
pad_x = pad_sequences(x, padding = 'post', value = 0.0)   # shape를 맞춰준다./ padding = 'pre' : 앞에서부터 0을 채움
print(pad_x)                                              # (12, 5)
#      'pre'          'post'        value = 1.0
# [[ 0  0  2  3]  [[ 2  3  0  0]  [[ 2  3  1  1]
#  [ 0  0  1  4]   [ 1  4  0  0]   [ 1  4  1  1]
#  [ 1  5  6  7]   [ 1  5  6  7]   [ 1  5  6  7]
#  [ 0  8  9 10]   [ 8  9 10  0]   [ 8  9 10  1]
#  [11 12 13 14]   [11 12 13 14]   [11 12 13 14]
#  [ 0  0  0 15]   [15  0  0  0]   [15  1  1  1]
#  [ 0  0  0 16]   [16  0  0  0]   [16  1  1  1]
#  [ 0  0 17 18]   [17 18  0  0]   [17 18  1  1]
#  [ 0  0 19 20]   [19 20  0  0]   [19 20  1  1]
#  [ 0  0  0 21]   [21  0  0  0]   [21  1  1  1]
#  [ 0  0  2 22]   [ 2 22  0  0]   [ 2 22  1  1]
#  [ 0  0  1 23]]  [ 1 23  0  0]]  [ 1 23  1  1]]


word_size = len(token.word_index) + 1 # [0] 포함
print('전체 토큰 사이즈 :', word_size)                      # 전체 토큰 사이즈 : 25


#2. model
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

model = Sequential()
model.add(Embedding(word_size, 10, input_length = 5))    # word_size = 전체 단어의 숫자 / output_node / columns
# model.add(Embedding(20, 10, input_length = 5))             # 벡터화 시키는 수를 임의로 줄 수 있다 => 단지 acc에 영향
# model.add(Embedding(25, 10))

model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.summary()
# _________________________________________________________________ 
# Layer (type)                 Output Shape              Param #    
# ================================================================= 
# embedding_1 (Embedding)      (None, 5, 11)             275         # Embedding param = word_size * output_node
# _________________________________________________________________ 
# flatten_1 (Flatten)          (None, 55)                0
# _________________________________________________________________ 
# dense_1 (Dense)              (None, 1)                 56
# ================================================================= 
# Total params: 331
# Trainable params: 331
# Non-trainable params: 0
# _________________________________________________________________


#3. compile, fit
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
              metrics = ['acc'])

model.fit(pad_x, labels, epochs = 30)

acc = model.evaluate(pad_x, labels)[1]
print('acc : ', acc)                                   # acc :  1.0



