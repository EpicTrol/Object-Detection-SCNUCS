import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

BATCH_SIZE = 512 # 大概需要2G的显存
EPOCHS = 20 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# 训练集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./pytorch-mnist/data/', train = True, download = False,
        transform = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize((0.1037,), (0.3081,))
              ])),
batch_size = BATCH_SIZE, shuffle = True)

# 测试集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./pytorch-mnist/data/', train = False, 
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1037,), (0.3081,))
])),
batch_size = BATCH_SIZE, shuffle = True)

print('train_loader len:',len(train_loader))
print('test_loader len:',len(test_loader))

# 定义模型
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        #1*1*28*28
        self.conv1 = nn.Conv2d(1, 10, 5) #5x5
        self.conv2 = nn.Conv2d(10, 20, 3) #3x3
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        in_size = x.size(0)
        out= self.conv1(x) # 1* 10 * 24 *24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2) # 1* 10 * 12 * 12
        out = self.conv2(out) # 1* 20 * 10 * 10
        out = F.relu(out)
        out = out.view(in_size, -1) # 1 * 2000
        out = self.fc1(out) # 1 * 500
        out = F.relu(out)
        out = self.fc2(out) # 1 * 10
        out = F.log_softmax(out, dim = 1)
        return out

#生成模型和优化器
model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())

# 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 初始为0，清除上个batch的梯度信息 即清零所有参数的梯度缓存 否则梯度将会与已有的梯度累加
        optimizer.zero_grad()

        # 前向+后向+优化
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward() # 导数 进行随机梯度的反向传播
        optimizer.step() # 更新参数

        if (batch_idx + 1) % 30 == 0:# 每30个batch进行输出
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 定义测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss =0
    correct = 0
    with torch.no_grad(): # 进行评测的时候网络不更新梯度    
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction = 'sum') # 将一批的损失相加
            pred = output.max(1, keepdim = True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item() # 预测正确的数目
    
    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100.* correct / len(test_loader.dataset)
            ))

# 最后开始训练和测试
for epoch in range(1, EPOCHS + 1):
    train(model,  DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)