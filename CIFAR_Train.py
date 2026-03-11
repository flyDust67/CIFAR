from CIFAR_Cnnmodel import *
from CIFAR_Cnnmodel_common import  *
from CIFAR_Mlpmodel import *
from tqdm import tqdm
from datetime import  datetime
from torch.utils.tensorboard import SummaryWriter
import os
epoches=50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

save_path=["./models/cnn.pth","./models/mlp.pth","./models/cnn_resnet.pth"]
save_dir = "./models"

model_cnn = CNN_Model_SimpleResNet().to(device)
model_cnn_common=CNN().to(device)
model_mlp=Mlp_Model().to(device)

writer = SummaryWriter()

#常规训练
def train(epoch,model,name):
    acc_train=0
    acc_test=0
    train_loss = 0
    test_loss = 0
    model.train()
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        model.optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.loss(outputs, targets)
        acc_train+=(outputs.argmax(1)==targets).sum().item()
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
    model.eval()
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        with torch.no_grad():
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = model.loss(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            acc_test+=torch.sum(predicted==targets).item()
    acc_train=acc_train/len(trainloader.dataset)
    acc_test=acc_test/len(testloader.dataset)
    print(f'Epoch {epoch} | Loss: {train_loss/len(trainloader):.3f}')
    writer.add_scalar(name+'loss',train_loss/len(trainloader),epoch)
    writer.add_scalar(name+'acc_train',acc_train,epoch)
    writer.add_scalar(name+'acc_test',acc_test,epoch)

def train_mlp(epoch,model,name):
    acc_train = 0
    acc_test = 0
    train_loss = 0
    test_loss = 0
    model.train()
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.view(inputs.size(0), -1) #增加展平而已
        model.optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.loss(outputs, targets)
        loss.backward()
        acc_train+=torch.sum(outputs.argmax(1)==targets).item()
        model.optimizer.step()
        train_loss += loss.item()
    model.eval()
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        with torch.no_grad():
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            loss = model.loss(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            acc_test+=torch.sum(predicted==targets).item()

    acc_train=acc_train/len(trainloader.dataset)
    acc_test=acc_test/len(testloader.dataset)

    print(f'Epoch {epoch} | Loss: {train_loss / len(trainloader):.3f}')
    writer.add_scalar(name+'loss', train_loss / len(trainloader), epoch)
    writer.add_scalar(name+'acc_train', acc_train, epoch)
    writer.add_scalar(name+'acc_test', acc_test, epoch)

#训练
if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    for epoch in range(1, 51):
        train(epoch,model_cnn,"带残差的CNN")
        train(epoch,model_cnn_common,"一般CNN")
        train_mlp(epoch,model_mlp,"MLP")

    # 确保文件夹存在

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 自动创建 models 文件夹
        print(f"检测到文件夹不存在，已创建: {save_dir}")

    checkpoint_mlp={
        'epoch':epoches,
        'model_state_dict': model_mlp.state_dict(),
        'optimizer_state_dict': model_mlp.optimizer.state_dict(),

    }
    checkpoint_cnn={
        'epoch':epoches,
        'model_state_dict': model_cnn.state_dict(),
        'optimizer_state_dict': model_cnn.optimizer.state_dict(),

    }
    checkpoint_cnn_common={
        'epoch':epoches,
        'model_state_dict': model_cnn_common.state_dict(),
        'optimizer_state_dict': model_cnn_common.optimizer.state_dict(),

    }
    #储存训练好的模型
    torch.save(checkpoint_cnn_common, save_path[0])
    torch.save(checkpoint_mlp, save_path[1])
    torch.save(checkpoint_cnn, save_path[2])