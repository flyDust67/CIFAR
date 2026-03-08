from CIFAR_Data import *
from CIFAR_Cnnmodel import *
from CIFAR_Cnnmodel_common import *
from CIFAR_Mlpmodel import *

from tqdm import tqdm
from CIFAR_Train import save_path, save_dir, os, writer

#存储数据
acc_list = {"cnn":0,"cnn_resnet":0,"mlp":0}


def ceshi_cnn(model,name):
    acc_all=0
    model.eval()
    with torch.no_grad():
        for index,(x,y) in tqdm(enumerate(testloader)):
            x, y = x.to(device), y.to(device)
            x = model(x)
            acc_all += (x.argmax(dim=1) == y).sum().item()
        acc_avg = acc_all / (len(testloader.dataset))
        acc_list[name] = acc_avg


def ceshi_mlp(model):
    acc_all=0
    model.eval()
    with torch.no_grad():
        for index,(x,y) in tqdm(enumerate(testloader)):
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            x = model(x)
            acc_all+=(x.argmax(dim=1)==y).sum().item()
        acc_avg=acc_all/(len(testloader.dataset))
        acc_list["mlp"]=acc_avg


if __name__ == '__main__':
    device="cuda" if torch.cuda.is_available() else "cpu"

    #加载模型
    if  os.path.exists(save_dir):
        model_cnn_common=CNN().to(device)
        model_mlp=Mlp_Model().to(device)
        model_cnn = CNN_Model_SimpleResNet().to(device)
        model_cnn_common.load_state_dict(torch.load(save_path[0], map_location=device))
        model_mlp.load_state_dict(torch.load(save_path[1],map_location=device))
        model_cnn.load_state_dict(torch.load(save_path[2],map_location=device))
        ceshi_mlp(model_mlp)
        ceshi_cnn(model_cnn,"cnn_resnet")
        ceshi_cnn(model_cnn_common,"cnn")
        print(acc_list)
        for i in acc_list.keys():
            writer.add_text(i,str(acc_list[i]))
    else:
        print("模型不存在")
