
import torch
import torchvision.datasets as dsets 
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu' # 만약 GPU가 사용 가능하면 device = cuda 그렇지 않으면 cpu

# 랜덤 시드 고정
torch.manual_seed(777)

# GPu 사용 가능시 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


# hyperparameter
learning_rate = 0.001
training_epochs =15
batch_size = 100


# data
mnist_train = dsets.MNIST(root='D:/data/MNIST_data/',           
                          train=True, 
                          transform=transforms.ToTensor(), 
                          download=True)

mnist_test = dsets.MNIST(root='D:/data/MNIST_data/', 
                         train=False, 
                         transform=transforms.ToTensor(), 
                         download=True)


# DataLoader
data_loader = DataLoader(dataset= mnist_train,
                            batch_size= batch_size,
                            shuffle= True,
                            drop_last= True)


# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째 층
        # ImgIn shape =(?, 28, 28, 1)
        #   Conv   -> ( ?, 28, 28, 32)
        #   Pool   -> ( ?, 14, 14, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 두번째 층
        # ImgIn shape =(?, 14, 14, 32)
        #   Conv   -> ( ?, 14, 14, 64)
        #   Pool   -> ( ?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # FC
        self.fc = nn.Linear(7 * 7 * 64, 10, bias = True)

        # FC 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)     # Flatten : out.size(0) = batch size
        out = self.fc(out) 
        return out


# CNN 모델 정의
model = CNN().to(device)


# loss
criterion = nn.CrossEntropyLoss().to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))      # 총 배치의 수 : 600


# Train
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)
        # print(X.shape)                            # torch.Size([100, 1, 28, 28])

        optimizer.zero_grad()

        h = model(X)

        cost = criterion(h, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch              # batch 마다의 cost의 평균값
    
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


# Prediction
# 학습을 진행하지 않을 것이므로 torch.no_grad()
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())