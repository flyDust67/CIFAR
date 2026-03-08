from CIFAR_Data import *


class CNN_Model_Basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.init(in_planes,planes,stride)

    def init(self,in_planes,planes,stride):
        """初始化"""
        #模型介绍：两层卷积，两层标准化，本质是残差块
        #激活函数：relu
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # 批量归一化，加速收敛
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.active1=F.relu

        self.shortcut = nn.Sequential()  # 残差连接
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.active1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = self.active1(out)
        return out


class CNN_Model_SimpleResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.init()

    def init(self):
        """初始化"""
        #模型介绍：一层卷积，两层残差块
        #损失函数：交叉熵损失
        #激活函数：softmax
        #优化器：SGD
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = CNN_Model_Basic(64, 64, stride=1)
        self.layer2 = CNN_Model_Basic(64, 128, stride=2)
        self.linear = nn.Linear(128, 10)  # 最终映射到10个类别

        self.avgpool = F.avg_pool2d
        #损失函数，优化器
        self.loss=nn.CrossEntropyLoss()
        self.optimizer=torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)



    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avgpool(out,16)  # 全局平均池化
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

if __name__=='__main__':
    model = CNN_Model_SimpleResNet()
