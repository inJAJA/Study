import torch
import torch.nn.functional as F

torch.manual_seed(1)

# Softmax Test
z = torch.FloatTensor([1, 2, 3])

hypothesis = F.softmax(z, dim = 0)
print(hypothesis)                   # tensor([0.0900, 0.2447, 0.6652])

print(hypothesis.sum())             # tensor(1.) : softmax의 합 = 1


#
z = torch.rand(3, 5, requires_grad = True)
hypothesis = F.softmax(z, dim = 1)
print(hypothesis)                   # tensor([[0.2645, 0.1639, 0.1855, 0.2585, 0.1277],
                                    #         [0.2430, 0.1624, 0.2322, 0.1930, 0.1694],
                                    #         [0.2226, 0.1986, 0.2326, 0.1594, 0.1868]], grad_fn=<SoftmaxBackward>)

y = torch.randint(5, (3,)).long()   # .randint() : 주어진 범위 내의 정수를 균등하게 생성 / .long() : int64
print(y)                            # tensor([0, 2, 1])


# One Hot Encoding
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1) # dim = 1, index =  y.unsqueeze(1), value = 1


# softmax 비용 함수
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim = 1).mean()
print(cost)


#----------------------------------------------------------------------------------------------
## F.softmax() + torch.log() = F.log_softmax()
# Low level
torch.log(F.softmax(z, dim = 1))

# High level
F.log_softmax(z, dim = 1)


## F.log_softmax() + F.nll_loss() = F.cross_entropy()
# Low level
# 첫번째 수식
(y_one_hot * -torch.log(F.softmax(z, dim = 1))).sum(dim = 1).mean()

# 두번째 수식
(y_one_hot * -F.log_softmax(z, dim = 1)).sum(dim = 1).mean()    # torch.log(F.softmax(z, dim = 1)) 
                                                                # = F.log_softmax(z, dim = 1)

# High level
# 세번째 수식
F.nll_loss(F.log_softmax(z, dim =1), y)     # One-Hot vector를 넣을 필요없이 바로 실제값을 인자로 사용
                                            # nll = Negative Log Likehood

# 네번째 수식
F.cross_entropy(z, y)
