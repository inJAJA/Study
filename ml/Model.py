# 1. 분류
from sklearn.svm import LinearSVC                     # 선형 분류    : 직선만 가능
from sklearn.svm import SVC                           # 다항식 커널

from sklearn.neighbors import KNeighborsClassifier    # 군집 분석

from sklearn.ensemble import RandomForestClassifier


# 2. 회귀
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor


# booster
from xgboost import XGBRegressor, XGBClassifier, plot_importance  

# keras_applications : Image
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import Xception
from keras.applications import ResNet50
from keras.applications import MobileNet
from keras.applications import MobileNetV2
from keras.applications import InceptionV3
from keras.applications import InceptionResNetV2

conv_base = VGG16(weights = 'imagenet', include_top = False, 
                  input_shape = (150, 150, 3))               # 위의 모델은 channel = 3 밖에 못씀 

# image = np.repeat(img[...,  np.newaxis], 3, -1)            # channel = 1 에서 3 으로 변경해준다. 
#       = np.repeat(img[:, :, np.newaxis], 3, axis = 2)

conv_base.summary()

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(conv_base)                  # VGG16 엮기
model.add(Dense(1, activation = ''))  # 내 데이터에 맞는 output_node설정

# ignore warning
import warnings                                

warnings.filterwarnings('ignore') 