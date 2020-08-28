
import numpy as np
import matplotlib.pyplot as plt

# https://github.com/WegraLee/deep-learning-from-scratch/blob/master/ch06/weight_init_activation_histogram.py

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU
def ReLU(x):
    return np.maximum(0, x)

# tanh
def tanh(x):
    return np.tanh(x)

def weight_init(method=None):
    '''가중치 초기화 함수
    
    Args:
        - method: 가중치 초기화 방법(large, small, xavier, relu)
    Returns:
        - np.array형태의 가중치 초기값
    '''
    w = 0
    if method == 'large':
        w = np.random.randn(node_num, node_num) * 1      # np.random.randn : 가우시안 표준 정규 분포 (0 기준, 표준편차 1)
    elif method == 'small':
        w = np.random.randn(node_num, node_num) * 0.01
    elif method == 'xavier':
        w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)  # Xavier init
    elif method == 'he':
        w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)  # He init
    
    return w

#----------------------------------------------------------------------------------------
input_data = np.random.randn(1000, 100)     # 1000개의 데이터
node_num = 100                              # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5                       # 은닉층이 5개
activations = {}                            # 이곳에 활성화 결과를 저장

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # w = weight_init('large')
    # w = weight_init('small')
    w = weight_init('xavier')
    # w = weight_init('he')

    a = np.dot(x, w)

    # z = sigmoid(a)
    # z = ReLU(a)
    z = tanh(a)

    activations[i] = z

# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
#     plt.xlim(0.1, 1)
    plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(-1,1))
plt.show()
'''
###
# 가중치의 초기값을 모두 0으로 초기화하거나 동일한 값으로 초기화할 경우 모든 뉴런의 동일한 출력값을 내보낼 것이다. 
# => 역전파(backpropaation) 단계에서 각 뉴런이 모두 동일한 그래디언트 값을 가지게 된다.


## tanh, small
#  평균이 0이고 표준편차가 1인 정규분포를 따르는 값으로 랜덤하게 초기화하고 
#  tanh를 활성화 함수로 사용하였을 경우 : tanh의 출력이 -1과 1로 집중 
#                                       =>그래디언트 소실(vanishing gradient) 문제가 발생

# 학습이 제대로 이루어지기 위해서는 각 뉴런의 활성화 함수 출력값이 고르게 분포되어 있어야 한다
# => 레이어와 레이어 사이에 다양한 데이터가 흘러야(forward, backprop) 신경망 학습이 효율적



# Xavier
# 적절한 데이터가 흐르기 위해서 : 각 레이어의 출력에 대한 분산 = 입력에 대한 분산
#                                역전파에서 레이어를 통과하기 전과 후의 그래디언트 분산이 동일해야 한다고 주장

# Xavier 초기값은 활성화 함수가 선형(linear)이라고 가정
# ->  sigmoid 계열(sigmoid, tanh)의 활성화 함수는 좌우 대칭이며 가운데 부분이 선형



# He
# Xavier 초기값은 ReLU 활성화 함수에서는 레이어가 깊어질 수록 출력값이 0으로 치우치는 문제가 발생
# ReLU에 적합한 초기값을 제안

# Xavier 초기값에서 sqrt(2) 배
# -> ReLU는 입력이 음수일 때 출력이 전부 0이기 때문에 더 넓게 분포시키기 위해서
'''

'''
# np.random.normal(loc, scale, size) : 정규분포(normal disttribution)
 : 평균(기준) = loc
 : 표준편차   = scale
 : 난수 생성 크기 = size 

# np.random.randn()
 : 평균(기준) = 0
 : 표준편차   = 1

# np.random.unifrom(low, high, size) : 균등분포(uniform distribution)
 : 최솟값 = low
 : 최댓값 = high
 : 난수 생성 크기 = size
 : 균등한 분포로 생성
'''