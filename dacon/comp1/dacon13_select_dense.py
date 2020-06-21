import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from xgboost import XGBRegressor, plot_importance
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import LeakyReLU

leaky = LeakyReLU(alpha = 0.2)

#1. data
train = pd.read_csv('./data/dacon/comp1/train.csv', index_col= 0 , header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', index_col= 0 , header = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', index_col= 0 , header = 0)

print('train.shape: ', train.shape)              # (10000, 75)  = x_train, test
print('test.shape: ', test.shape)                # (10000, 71)  = x_predict
print('submission.shape: ', submission.shape)    # (10000, 4)   = y_predict


trian = train.interpolate(axis = 0)
test= test.interpolate(axis = 0)

train = train.fillna(train.mean())
test = test.fillna(test.mean())


x = train.iloc[:, :71]                           
y = train.iloc[:, -4:]
print(x.shape)                                   # (10000, 71)
print(y.shape)                                   # (10000, 4)

x = x.values
y = y.values
x_pred = test.values

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)

pca = PCA(n_components= 10)
pca.fit(x)
x = pca.transform(x)
x_pred = pca.transform(x_pred)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size =0.8,
                                                    shuffle = True, random_state = 66)

#2. feature_importance
xgb = XGBRegressor()
multi_XGB = MultiOutputRegressor(xgb)
multi_XGB.fit(x_train, y_train)

print(len(multi_XGB.estimators_))   # 4


# print(multi_XGB.estimators_[0].feature_importances_)
# print(multi_XGB.estimators_[1].feature_importances_)
# print(multi_XGB.estimators_[2].feature_importances_)
# print(multi_XGB.estimators_[3].feature_importances_)

#2. model

def create_hyperparameter():
    batches = [64, 128, 256]
    epochs = [100, 150, 200]
    dropout = np.linspace(0.1, 0.5, 5).tolist()
    activation= ['relu', 'elu', leaky]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    return {'deep__batch_size': batches, 'deep__epochs':epochs, 'deep__act': activation, 'deep__drop': dropout,
            'deep__optimizer': optimizers}
                  

for i in range(len(multi_XGB.estimators_)):
    threshold = np.sort(multi_XGB.estimators_[i].feature_importances_)

    for thres in threshold:
        selection = SelectFromModel(multi_XGB.estimators_[i], threshold = thres, prefit = True)
    
        select_x_train = selection.transform(x_train)
        select_x_test = selection.transform(x_test)
        select_x_pred = selection.transform(x_pred)
    
        def build_model(drop=0.5, optimizer = 'adam', act = 'relu'):
            inputs = Input(shape= (select_x_train.shape[1], ))
            x = Dense(51, activation =act)(inputs)
            x = Dropout(drop)(x)
            x = Dense(100, activation = act)(x)
            x = Dropout(drop)(x)
            x = Dense(150, activation = act)(x)
            x = Dropout(drop)(x)
            x = Dense(200, activation = act)(x)
            x = Dropout(drop)(x)
            x = Dense(300, activation = act)(x)
            x = Dropout(drop)(x)
            x = Dense(250, activation = act)(x)
            x = Dropout(drop)(x)
            x = Dense(200, activation = act)(x)
            x = Dropout(drop)(x)
            x = Dense(128, activation = act)(x)
            x = Dropout(drop)(x)
            outputs = Dense(4, activation = act)(x)
            model = Model(inputs = inputs, outputs = outputs)
            model.compile(optimizer = optimizer, metrics = ['mae'],  loss = 'mae')
            return model

        # wrapper    
        model = KerasRegressor(build_fn = build_model, verbose =2)

        parameter = create_hyperparameter()

        pipe = Pipeline([('scaler', RobustScaler()), ('deep', model)])

        search = RandomizedSearchCV(pipe, parameter, cv = 3)
        search.fit(select_x_train, y_train )
        
        y_pred = search.predict(select_x_test)
        mae = mean_absolute_error(y_test, y_pred)
        score =r2_score(y_test, y_pred)
        print("Thresh=%.3f, n = %d, R2 : %.2f%%, MAE : %.3f"%(thres, select_x_train.shape[1], score*100.0, mae))
        
        y_predict = search.predict(select_x_pred)
        # submission
        a = np.arange(10000,20000)
        submission = pd.DataFrame(y_predict, a)
        submission.to_csv('./dacon/comp1/sub_XG%i_%.5f.csv'%(i, mae),index = True, header=['hhb','hbo2','ca','na'],index_label='id')
