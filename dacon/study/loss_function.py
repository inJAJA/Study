from keras.models import Model
from keras.layers import Input, Dense
import keras.backend as K

''' constructiong a custiom metric '''
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['acc', mean_pred])


# loss function은 오직 2개의 인자만 받아 들인다 : y_true, y_pred 


''' create loss function ''' # 인자 개수의 제한 없이 사용
# Build model
inputs = Input(shape=(128, ))
layer1 = Dense(64, activation = 'relu')(inputs)
layer2 = Dense(64, activation = 'relu')(layer1)
predictions = Dense(10, activation = 'softmax')(layer2)
model = Model(inputs = input, outputs = predictions)

# Define custom loss
def custom_loss(layer):

    # Create a loss function that adds the MSE loss to  the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred = y_ture) + K.square(layer), axis = -1)

    # Return a function
    return loss

# Compile the model
model.compile(optimizer = 'adam',
              loss = custom_loss(layer), # Call the loss function with the selected layer
              metrics = ['acc'])

# train
model.fit(data, labels)


''''''
def model_loss(self):
    '''Wrapper function which calculate auxiliary values for the complete loss function.
       Returns a *function* which calcuates the  complete loss given only the input and target'''
    # KL loss
    kl_loss = self.calculate_kl_loss
    # Reconstruction loss
    md_loss_func = self.calculate_md_loss

    # KL weight (to be used by total loss and by annealing scheduler)
    self.kl_weight = K.variable(self.hps['kl_weight_start'], name = 'kl_weight')
    kl_weight =self.kl_weight

    def seq2seq_loss(y_true, y_pred):
        '''Final loss calculation function to be passed to optimizer'''
        # Reconstruction loss
        md_loss = md_loss_func(y_true, y_pred)
        # Full loss
        model_loss = kl_weight*kl_loss() + md_loss
        return model_loss

    return seq2seq_loss

# loss함수 만들기 참고 : https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618


# dacon 진동데이터
# 평가 함수
def kaeri_metric(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: KAERI metric
    '''
    
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)
                # X, Y                     # M, V

### E1과 E2는 아래에 정의됨 ###

def E1(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: distance error normalized with 2e+04
    '''
    
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]           # X, Y
    
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)
                                            # 각 컬럼에 대한 np.sum

def E2(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: sum of mass and velocity's mean squared percentage error
    '''
    
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]             # M, V
    
    
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))
                                                            # 각 컬럼에 대한 np.sum

# custom loss_function
weight1 = np.array([1,1,0,0])
weight2 = np.array([0,0,1,1])


def my_loss(y_true, y_pred):  # loss function은 오직 2개의 인자만 받아들임 : y_true, y_pred
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+0.000001)]) # [(y_pred-y_true),(y_true+0.000001)]을 대입
    return K.mean(K.square(divResult))


def my_loss_E1(y_true, y_pred):
    return K.mean(K.square(y_true-y_pred)*weight1)/2e+04

def my_loss_E2(y_true, y_pred):                              # 나누기 연산이 있을 때 분모가 0이 되는 것을 방지하기 위해서 더해줌
    divResult = Lambda(lambda x: x[0]/x[1])([(y_pred-y_true),(y_true+0.000001)])                 # 주로 K.epsilon를 더해준다.
    return K.mean(K.square(divResult)*weight2)                                                            