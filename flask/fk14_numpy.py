import pymssql as ms
import numpy as np

conn = ms.connect(server='127.0.0.1', user='bit2', password='1234', database='bitdb')

cursor = conn.cursor()

cursor.execute('SELECT * FROM iris2;')

row = cursor.fetchall()                         # 줄 바꿈 없이 모두 붙어서 나옴
print(row)
conn.close()

print('-------------------------------------')
aaa = np.array(row)
print(aaa)                                      # 잘 정렬되어 나옴
# [['5.1' '3.5' '1.4' '0.2' 'Iris-setosa']
#  ['4.9' '3.0' '1.4' '0.2' 'Iris-setosa']
#  ....
#  [['5.1' '3.5' '1.4' '0.2' 'Iris-setosa']
#  ['4.9' '3.0' '1.4' '0.2' 'Iris-setosa']
print(aaa.shape)                                # (150, 5)
print(type(aaa))                                # <class 'numpy.ndarray'>

np.save('./data/test_flask_iris2.npy', aaa)     # numpy로 저장