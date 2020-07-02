import numpy as np

gradient = lambda x: 2*x - 4       # 적분 : x^2 - 4x + b
                # x = 인력하는 변수
                

def gradient2(x):
    temp = 2*x - 4
    return temp
x = 3

print(gradient(x))   # 2

print(gradient2(x))  # 2


# 추가 
# lambda 매개변수들 : (식_1) if (조건_1) else (식_2) if (조건_2) else (식_3)
#                   [                 ][                     ][          ]
x = np.array([-3, 0, 5])
def lambda1(x):
    return list(map(lambda x: -x if x < 0 else 0 if x == 0 else x, x))

print(lambda1(x))         # [3, 0, 5]

def lambda2(x):
    y_list = []
    for x in x:
        if x < 0 :
            y = -x
        elif x == 0:
            y = 0
        else:
            y = x
        y_list.append(y)
    return y_list

print(lambda2(x))         # [3, 0, 5]