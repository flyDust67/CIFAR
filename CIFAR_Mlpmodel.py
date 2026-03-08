
from CIFAR_Data import *

class Mlp_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.init()

    def init(self):
        """初始化"""
        #模型信息：三层隐藏层
        #损失函数：交叉熵损失
        #激活函数：relu和softmax

        self.linear1=nn.Linear(32*32*3,32*3)
        self.linear2=nn.Linear(32*3,32)
        self.linear3=nn.Linear(32,10)
        self.active1=F.relu
        self.active2=F.softmax
        self.loss=nn.CrossEntropyLoss()
        self.optimizer=optim.SGD(self.parameters(),lr=0.001,momentum=0.9)


    def forward(self,x):
        x=self.active1(self.linear1(x))
        x=self.active1(self.linear2(x))
        output=self.active1(self.linear3(x))
        return output


if __name__=='__main__':
    model = Mlp_Model()