from keras.preprocessing.text import Tokenizer
import numpy as np 

'''
핸즈온 p525 ~ p530 참고
'''

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


x = token.texts_to_sequences(docs)   
print(x)                            


from keras.preprocessing.sequence import pad_sequences    
pad_x = pad_sequences(x, padding = 'post', value = 0.0)   
print(pad_x)                                             



word_size = len(token.word_index) + 1 # [0] 포함
print('전체 토큰 사이즈 :', word_size)                      # 전체 토큰 사이즈 : 25


#2. model
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM

model = Sequential()
# model.add(Embedding(word_size, 10, input_length = 5))    # word_size = 전체 단어의 숫자 / output_node / columns
# model.add(Embedding(20, 10, input_length = 5))               
model.add(Embedding(25, 10))                               # embedding과 lstm을 같이 쓰면 input_length를 쓰지 않아도 됌
                                                           # embedding 3차원 출력 -> LSTM 3차원 입력 (Conv1D도 가능)
                                                           # input_length를 쓰지 않아도 None에 알아서 입력한 데이터의 shape가 들어감
model.add(LSTM(3))                                         
# model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.summary()
# _________________________________________________________________       
# Layer (type)                 Output Shape              Param #
# =================================================================  = input_length는 paramter에 관여 X   
# embedding_1 (Embedding)      (None, None, 10)          250         = word_size * 10 
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 3)                 168         = 4 * (10 + 3 + 1) * 3
# _________________________________________________________________       
# dense_1 (Dense)              (None, 1)                 12
# =================================================================       
# Total params: 1,230
# Trainable params: 1,230
# Non-trainable params: 0
# _________________________________________________________________  


#3. compile, fit
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
              metrics = ['acc'])

model.fit(pad_x, labels, epochs = 30)

acc = model.evaluate(pad_x, labels)[1]
print('acc :', acc)                                   # acc : 0.75  



