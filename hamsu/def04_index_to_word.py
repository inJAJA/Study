def index_to_word(dataset, data):
    word_to_index = dataset.get_word_index()
    index_to_word = {}
    for key, value in word_to_index.items():
        index_to_word[value] = key
    
    text = ' '.join([index_to_word[index] for index in data])
    print(text)
    
    return text

'''
from keras.datasets import reuters, imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 2000) # train:test = 50:50

index_to_word(imdb, x_train[0])
'''