from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

print(type(iris))            # <class 'sklearn.utils.Bunch'> : sklearn에서 제공하는 파일

x_data = iris.data
y_data = iris.target

print(type(x_data))        # <class 'numpy.ndarray'> : numpy 형태
print(type(y_data))         # <class 'numpy.ndarray'>

np.save('./data/iris_x.npy', arr= x_data) # numpy로 데이터 저장
np.save('./data/iris_y.npy', arr= y_data)

x_data_load = np.load('./data/iris_x.npy') # numpy로 저장된 데이터 불러오기
y_data_load = np.load('./data/iris_y.npy')
  
print(type(x_data_load))     # <class 'numpy.ndarray'>
print(type(y_data_load))     # <class 'numpy.ndarray'>

print(x_data_load.shape)     # (150, 4)
print(y_data_load.shape)     # (150, )


