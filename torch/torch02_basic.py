import torch

nums = torch.arange(9)
print(nums)                 # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])

print(nums.shape)           # torch.Size([9]) 

print(type(nums))           # <class 'torch.Tensor'>

np = nums.numpy()           # tensor를 numpy로 타입 변환
print(type(np))             # <class 'numpy.ndarray'>

re = nums.reshape(3, 3)     # reshape
print(re)                   # tensor([[0, 1, 2],     
                            #         [3, 4, 5],     
                            #         [6, 7, 8]])

nums = torch.arange(9).reshape(3, 3)
print(nums)                 # tensor([[0, 1, 2],     
                            #         [3, 4, 5],     
                            #         [6, 7, 8]])

print(nums + nums)          # tensor([[ 0,  2,  4],  
                            #         [ 6,  8, 10],  
                            #         [12, 14, 16]]) 