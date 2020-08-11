import torch

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1:', m1.shape)   # Shape of Matrix 1: torch.Size([2, 2])
print('Shape of Matrix 2:', m2.shape)   # Shape of Matrix 2: torch.Size([2, 1])


# .matmul() : 행렬 곱셈
print(m1.matmul(m2))                    # tensor([[ 5.],
                                        #         [11.]])


m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1] , [2]])
print('Shape of Matrix 1:', m1.shape)   # Shape of Matrix 1: torch.Size([2, 2])
print('Shape of Matrix 2:', m2.shape)   # Shape of Matrix 2: torch.Size([2, 1])

# .mul() : 일반 곱셈
print(m1 * m2)                          # tensor([[1., 2.],
                                        #         [6., 8.]])
print(m1.mul(m2))                       # tensor([[1., 2.],
                                        #         [6., 8.]])


# .mean() : 평균
t = torch.FloatTensor([1, 2])
print(t.mean())                         # tensor(1.5000)

t = torch.FloatTensor([[1, 2],[3, 4]])
print(t.mean())                         # tensor(2.5000)

print(t.mean(dim = 0))                  # tensor([2., 3.])


# .sum()
t = torch.FloatTensor([[1, 2],[3, 4]])
print(t)

print(t.sum())                          # tensor(10.)
print(t.sum(dim = 0))                   # tensor([4., 6.])
print(t.sum(dim = 1))                   # tensor([3., 7.])
print(t.sum(dim = -1))                  # tensor([3., 7.])


# .max() : 원소의 최대값 리턴
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.max())                          # tensor(4.)
print(t.max(dim = 0))                   # torch.return_types.max(values=tensor([3., 4.])
                                        #                       ,indices=tensor([1, 1])) -> dim인자를 주면 argmax도 함께 리턴 함

print('Max    :', t.max(dim=0)[0])      # Max    : tensor([3., 4.])
print('Argmax :', t.max(dim=0)[1])      # Argmax : tensor([1, 1])


                                        




