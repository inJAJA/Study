from sklearn.datasets import load_boston
from sklearn.svm import SVC, LinearSVC 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1. data
boston = load_boston()

x = boston.data
y = boston.target 

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1, train_size =0.8)

#2. model
# model =SVC()                                        # ValueError
# model = LinearSVC()                                 # ValueError       
# model = KNeighborsClassifier(n_neighbors = 1)       # ValueError
# model = RandomForestClassifier()                    # ValueError : 분류모델
model = KNeighborsRegressor(n_neighbors = 1)
# model = RandomForestRegressor()


#3. fit
model.fit(x_train, y_train)

#4. predict
y_pred = model.predict(x_test)


score = model.score(x_test, y_test)
print('score: ', score)


r2 = r2_score(y_test, y_pred)
print('R2: ', r2)

# ValueError: Unknown label type: 'continuous'
# ValueError: Unknown label type: 'continuous'

# ValueError: Unknown label type: 'continuous'
# ValueError: Unknown label type: 'continuous'

# ValueError: Unknown label type: 'continuous'
# ValueError: Unknown label type: 'continuous'

# ValueError: Unknown label type: 'continuous'
# ValueError: Unknown label type: 'continuous'

# score:  0.7894467162891708
# R2:  0.7894467162891708

# score:  0.910737124510854
# R2:  0.910737124510854


