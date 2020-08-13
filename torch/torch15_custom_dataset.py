import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
'''
# custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):             # dataset 전처리

    def __len__(self):              # dataset의 길이 = 총 샘플 수 

    def __getitem__(self, idx):     # dataset에서 특정 1개의 샘플을 가져오는 함수
'''

# Dataset 상속
class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):              # 총 data의 개수를 리턴
        return len(self.x_data)
    
    def __getitem__(self, idx):     # index를 입력받아 그에 mapping되는 입출력 데이터를 파이토치의 Tensor로 return
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size= 2, shuffle = True)


# model
model = torch.nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)


# train
nb_epochs = 20
for epoch in range(nb_epochs +1):
    for batch_idx, samples in enumerate(dataloader):
        # print(samples)
        # print(batch_idx)
        # print('-----------------')

        x_train, y_train = samples

        prediction = model(x_train)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))

new_var =  torch.FloatTensor([[73, 80, 75]]) 
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 