import torch

# Braodcasting
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2,2]])
print(m1 + m2)                      # tensor([[5., 5.]])


# Vector + Scaler
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3])
print(m1 + m2)                      # tensor([[4., 5.]])


# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])    # 1 x 2
m2 = torch.FloatTensor([[3],[4]])   # 2 x 1
print(m1 + m2)                      # tensor([[4., 5.],
                                    #         [5., 6.]])
