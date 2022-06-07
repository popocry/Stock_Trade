import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torchsummary import summary


# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:', device)

# 理解GRU網路結果https://blog.csdn.net/qq_27825451/article/details/99691258
class GRUNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, out_size, drop_out, n_layers=1):
        super(GRUNet, self).__init__()

        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size
        self.drop_out = drop_out

        # 這裡指定了BATCH FIRST,所以輸入時BATCH應該在第一維度
        self.gru = torch.nn.GRU(input_dim, hidden_size, n_layers, dropout=drop_out, batch_first=True)
        # relu
        self.relu = torch.nn.ReLU()
        # dropout
        self.drop_out = torch.nn.Dropout(drop_out)
        # 加了一個線性層，全連線
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(hidden_size, out_size))
        # 加入了第二個全連線層
        #self.fc2 = torch.nn.Linear(128, out_size)
        #self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # x的格式（batch,seq,feature）
        output, hidden = self.gru(x)
        output = self.relu(output)
        output = self.drop_out(output)
        # output是所有隱藏層的狀態，hidden是最後一層隱藏層的狀態
        output = self.fc1(output)
        #output = self.fc2(output)
        #output = self.softmax(output)

        # 僅僅獲取 time seq 維度中的最後一個向量
        # the last of time_seq
        output = output[:, -1, :]

        return output

# parameters
feature_dim = 7
hidden_size = 256
output_dim = 3
num_layers = 2
drop_out_gru = 0.1

# hyper parameters
BATCH_SIZE = 64  # batch_size
LEARNING_RATE = 0.001  # learning_rate
EPOCH = 200  # epochs

# build model
net = GRUNet(feature_dim, hidden_size, output_dim, drop_out_gru, num_layers)
net = net.to(device)# GPU
print(net)
summary(net, (20, 7))  # 輸出網路結構
net = net.double()

optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE) # optimizer
loss_func = torch.nn.CrossEntropyLoss()# loss function

def Split_Data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
    #轉成tensor
    x_train = torch.tensor(x_train)
    x_test = torch.tensor(x_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    #包裝成dataset
    torch_train_dataset = data.TensorDataset(x_train, y_train)
    torch_test_dataset = data.TensorDataset(x_test, y_test)
    #封裝成dataloader
    trainloader = data.DataLoader(
        dataset=torch_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # num_workers=2
    )
    testloader = data.DataLoader(
        dataset=torch_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # num_workers=2
    )
    return trainloader, testloader





# Training
def Train(epoch, trainloader):
    global train_acc, train_loss
    print('\n train Epoch: {}'.format( epoch + 1))
    net.train()
    train_loss_tmp = 0
    train_loss_avg = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #print(batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        #print(outputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss_tmp += loss.item()
        _, predicted = torch.max(outputs, 1)
        #print(predicted)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_loss_avg = train_loss_tmp / (batch_idx + 1)
        print('Epoch: [{}/{}] Batch [{}/{}] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(epoch + 1, EPOCH, batch_idx + 1, len(trainloader),
                                                                                 train_loss_avg, 100. * correct / total, correct, total))
    train_loss.append(train_loss_avg)
    train_acc.append(100. * correct / total)
    print('\n -----train Epoch Over: {}------\n'.format(epoch))
    print('Epoch: [{}/{}] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(epoch + 1, EPOCH, train_loss_avg, 100. * correct / total, correct, total))

def Test(epoch, testloader):
    global minimum_loss_tmp
    print('\n test Epoch: {}'.format( epoch + 1))
    global test_acc, test_loss, best_acc_tmp
    net.eval()
    test_loss_tmp = 0
    test_loss_avg = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = loss_func(outputs, targets)

            test_loss_tmp += loss.item()
            _, predicted = torch.max(outputs, 1)# 輸出最大值索引位置
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()# 預測的索引位置和真實值的索引位置比較是否相等，相等的數量加起來輸出(正確筆數)

            test_loss_avg = test_loss_tmp / (batch_idx + 1)
            print('Epoch: [{}/{}] Batch [{}/{}] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(epoch + 1, EPOCH, batch_idx + 1, len(testloader),
                                                                                     test_loss_avg, 100. * correct / total, correct, total))

    test_loss.append(test_loss_avg)
    test_acc.append(100. * correct / total)
    minimum_loss_tmp = min(test_loss) #最低loss
    print('\n -----test Epoch Over: {}------\n'.format(epoch))
    print('Epoch: [{}/{}] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(epoch + 1, EPOCH, test_loss_avg, 100. * correct / total, correct, total))


def Train_model(stock_code, dataset, labels):
    global train_loss
    global train_acc
    global test_acc
    global test_loss
    global minimum_loss
    global minimum_loss_tmp

    train_loss = []
    train_acc = []
    test_acc = []
    test_loss = []
    minimum_loss = 10000000
    minimum_loss_tmp = 0

    #切割資料集
    train_loader, test_loader = Split_Data(dataset, labels)

    #訓練
    for i in range(EPOCH):
        Train(i, train_loader)
        Test(i, test_loader)
        if minimum_loss_tmp < minimum_loss:
            minimum_loss = minimum_loss_tmp
            torch.save(net.state_dict(), './model/best'+ stock_code + '.pth')

    # loss圖
    t = np.arange(EPOCH)
    plt.figure()
    plt.plot(t, train_loss, label='Train')
    plt.plot(t, test_loss, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("./figures/" + str(stock_code) + "_loss.jpg")
    #plt.show()

    # 準確率圖
    plt.figure()
    plt.plot(t, train_acc, label='Train')
    plt.plot(t, test_acc, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("./figures/" + str(stock_code) + "_acc.jpg")
    #plt.show()

def build_model():
    net = GRUNet(feature_dim, hidden_size, output_dim, drop_out_gru, num_layers)
    net = net.cpu()# CPU
    net = net.double()
    return net