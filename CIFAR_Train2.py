#该训练是基于原有模型继续训练
from CIFAR_Train import *

if __name__ == '__main__':
    model_cnn_common = CNN().to(device)
    model_mlp = Mlp_Model().to(device)
    model_cnn = CNN_Model_SimpleResNet().to(device)
    model_cnn_common.load_state_dict(torch.load(save_path[0], map_location=device))
    model_mlp.load_state_dict(torch.load(save_path[1], map_location=device))
    model_cnn.load_state_dict(torch.load(save_path[2], map_location=device))

    np.random.seed(42)
    torch.manual_seed(42)
    for epoch in range(1, 51):
        train(epoch, model_cnn, "带残差的CNN")
        train(epoch, model_cnn_common, "一般CNN")
        train_mlp(epoch, model_mlp, "MLP")

    # 确保文件夹存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 自动创建 models 文件夹
        print(f"检测到文件夹不存在，已创建: {save_dir}")

    # 储存训练好的模型
    torch.save(model_cnn_common.state_dict(), save_path[0])
    torch.save(model_mlp.state_dict(), save_path[1])
    torch.save(model_cnn.state_dict(), save_path[2])