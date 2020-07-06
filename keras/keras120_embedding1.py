from keras.preprocessing.text import Tokenizer

text = '나는 맛있는 밥을 먹었다'

token = Tokenizer()                   # 한개의 문장을 단어 단위로 잘라서 인덱싱(수치화)을 걸어 줌 
token.fit_on_texts([text])

print(token.word_index)               # {'나는': 1, '맛있는': 2, '밥을': 3, '먹었다': 4}


x = token.texts_to_sequences([text])
print(x)                              # [[1, 2, 3, 4]] 
                                      # 문제점 : '나는'과 '먹었다'의 가치가 다르다.

from keras.utils import to_categorical

word_size = len(token.word_index) + 1 # [0]추가
x = to_categorical(x, num_classes= word_size)
print(x)
# [[[0. 1. 0. 0. 0.]                  # 문제점 : 단어 수가 많아지면 data(컬럼)이 너무 많아짐
#   [0. 0. 1. 0. 0.]
#   [0. 0. 0. 1. 0.]
#   [0. 0. 0. 0. 1.]]]

