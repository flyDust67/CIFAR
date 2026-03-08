
from CIFAR_Data import *

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.init()

    def init(self):
        """初始化"""
        #模型介绍
        #三层卷积层+三层标准化+一层最大池化
        #损失函数:交叉熵损失
        #激活函数: relu和softmax
        #优化器:SGD
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.active1=F.relu
        self.maxpool=F.max_pool2d
        self.linear=nn.Linear(in_features=128, out_features=10)

        self.loss=nn.CrossEntropyLoss()
        self.optimizer=optim.SGD(self.parameters(),lr=0.001,momentum=0.9)

    def forward(self,x):
        x = self.bn1(self.conv1(x))
        x=self.bn2(self.conv2(x))
        x=self.bn3(self.conv3(x))
        x = self.maxpool(self.active1(x),32)
        x=x.view(x.size(0),-1)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    net = CNN()
    net.cuda()