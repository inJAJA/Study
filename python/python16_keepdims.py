import numpy as np

a = np.arange(10).reshape(2, 5)
print(a)                                    # [[0 1 2 3 4]
                                            #  [5 6 7 8 9]]

sum = np.sum(a)
print(sum)                                  # 45

# keepdims : 차원을 유지함
a1 = np.sum(a, keepdims=True)
print(a1)                                   # [[45]]

a2 = np.sum(a, axis = 0, keepdims= True)
print(a2)                                   # [[ 5  7  9 11 13]]

a3 = np.sum(a, axis = 1, keepdims= True)
print(a3)                                   # [[10]
                                            #  [35]]