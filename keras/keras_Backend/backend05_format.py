import keras.backend as K

# image_data_format
K.image_data_format()                       # return the default image data format convention
                                            # return : a string : 'channels_first','channels_last'

# example 
f = K.image_data_format()
print(f)                                    # channels_last


# set_image_data_format
# K.set_image_data_format(data_format)      # set the value od the data format convention
                                            # data_format : string : 'channels_first', 'channels_last'


# example
data_format = K.image_data_format()
print(data_format)                          # channels_last

K.set_image_data_format('channels_first')
data_format = K.image_data_format()
print(data_format)                          # channels_first

