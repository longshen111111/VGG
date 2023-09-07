import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model import vgg
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available()else 'cpu')
    print("using {} device.".format(device))
    data_transform = {
        'train':transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]),#对于训练集的操作
        'val':transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])#对于验证集的操作
    }
    data_root = os.path.abspath(os.path.join(os.getcwd()))#D:\pythonProject2
    image_path = os.path.join(data_root,'data_set','flower_data')#D:\pythonProject2\data_set\flower_data
    assert os.path.exists(image_path),"{} path does not exist.".format(image_path)#检测image_path是否存在
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path,'train'),#D:\pythonProject2\data_set\flower_data\train
                                         transform=data_transform['train'])#root传入数据的根目录   transform作为对于训练集的操作
    train_num = len(train_dataset)
    flower_list = train_dataset.class_to_idx#将训练集中的每一个类别的索引作为字典形式进行储存
    cla_dict = dict((val,key)for key,val in flower_list.items())#将字典中的索引和类别进行一个调换
    json_str = json.dumps(cla_dict,indent=4)#将字典转化为json格式的字符串，对于前面的进行四个空格的后退，增强文件的可读性
    with open('class_indices.json','w') as json_file:
        json_file.write(json_str)#创建一个json文件，然后将json_str进行接入
    batch_size = 32
    nw = min([os.cpu_count(),batch_size if batch_size > 1 else 0,8])#选择一个合适的batch_size
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=nw)#创建训练集的data_loader
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])#
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    model_name = 'vgg16'
    net = vgg(model_name=model_name,num_classes=5,init_weights=True)#设置可供训练的网络
    net.to(device)#将其放入gpu中进行数据的处理
    loss_function=nn.CrossEntropyLoss()#创建损失函数
    optimizer = optim.Adam(net.parameters(),lr = 1e-4)#构造一个普通的优化器，并且设置一个学习率
    epochs = 30#设置训练的次数
    best_acc = 0.0#设置一个最初的正确率
    save_path = './{}Net.pth'.format(model_name)#保存网络的权重
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)#tqdm是为了创造一个进度条，可以更好的看到进度
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()#将现存的梯度归零
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))#定义了具体的损失函数
            loss.backward()#反向传播
            optimizer.step()#优化器对权重进行相应的优化

            # print statistics
            running_loss += loss.item()#更新损失的数值

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():#控制模型不进行梯度的更新计算
            val_bar = tqdm(validate_loader, file=sys.stdout)#可视化验证数据集加载的进度
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]#取出最大值所对应的索引作为预测值的结果
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()#将所有预测正确的数目都进行相加，求和

        val_accurate = acc / val_num#计算最后的正确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)#将网络对应的权重进行保存

    print('Finished Training')


if __name__ == '__main__':
    main()