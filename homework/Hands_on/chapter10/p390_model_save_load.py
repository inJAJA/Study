from tensorflow import keras

#1. save
model = keras.models.Sequwntial([...]) # 또는 keras.Model([...])
model.compile([...])
model.fit([...])
model.save('my_keras_model.h5')

#2. load
model = keras.models.load_model('my_keras_model.h5')