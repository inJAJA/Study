import torch
import numpy as np

''' 차원 바꾸기 '''
# .view()  : 원소의 수를 유지하면서 tensor크기 변경
#       .=. reshape
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)

print(ft.shape)                 # torch.Size([2, 2, 3])

print(ft.view([-1, 3]))         # tensor([[ 0.,  1.,  2.],
                                #         [ 3.,  4.,  5.],
                                #         [ 6.,  7.,  8.],
                                #         [ 9., 10., 11.]])
print(ft.view([-1, 3]).shape)   # torch.Size([4, 3])


# .squeeze() : 1인 차원을 차원을 제거한다.
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)                               # tensor([[0.],
                                        #         [1.],
                                        #         [2.]])
print(ft.shape)                         # torch.Size([3, 1])

print(ft.squeeze())                     # tensor([0., 1., 2.])
print(ft.squeeze().shape)               # torch.Size([3])


# .unsqueeze() : 특정 위치에 1인 차원을 추가한다.
ft = torch.Tensor([0, 1, 2])
print(ft.shape)                 # torch.Size([3]) 

print(ft.unsqueeze(0))          # tensor([[0., 1., 2.]])
print(ft.unsqueeze(0).shape)    # torch.Size([1, 3])

# view도 동일하게 사용가능
print(ft.view(1, -1))           # tensor([[0., 1., 2.]])
print(ft.view(1, -1).shape)     # torch.Size([1, 3])


''' 연결 하기 '''
# .cat() : 연결하기 (concatenate)
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([x, y], dim = 0))   # tensor([[1., 2.],
                                    #        [3., 4.],
                                    #        [5., 6.],
                                    #        [7., 8.]])

print(torch.cat([x, y], dim = 1))   # tensor([[1., 2., 5., 6.],
                                    #         [3., 4., 7., 8.]])


# .stack() : 연결하기 (stacking)
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))           # tensor([[1., 4.],
                                        #         [2., 5.],
                                        #         [3., 6.]])

print(torch.stack([x, y, z], dim =1))   # tensor([[1., 2., 3.],
                                        #         [4., 5., 6.]])


''' Tensor 생성 '''
# .ones_like()  : 1으로 채워진 텐서
x = torch.FloatTensor([[0, 1, 2],[2, 1, 0]])    
print(x)                                        # tensor([[0., 1., 2.],
                                                #         [2., 1., 0.]])

print(torch.ones_like(x))                       # tensor([[1., 1., 1.],
                                                #         [1., 1., 1.]])

# .zeros_like() : 0으로 채워진 텐서
print(torch.zeros_like(x))                      # tensor([[0., 0., 0.],
                                                #         [0., 0., 0.]])


''' 덮어쓰기 연산 '''
# .연산_() : 기존의 값에 덮어쓰기
x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul_(2. ))                      # tensor([[2., 4.],
                                        #         [6., 8.]])
print(x)                                # tensor([[2., 4.],
                                        #         [6., 8.]])