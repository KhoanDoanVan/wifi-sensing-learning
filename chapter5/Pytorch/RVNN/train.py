import torch
from torch import nn
from chapter5.Pytorch.RVNN.train_dataset import TrainDataset
from torch.optim.lr_scheduler import StepLR
from chapter5.Pytorch.RVNN.ResNet import ResNet18
from test import test

def init_weights(model):
    if type(model) in [nn.Linear, nn.Conv3d, nn.Conv2d, nn.Conv1d]:
        nn.init.normal_(model.weight, std=0.03)

def train(device):

    model = ResNet18()

    model.apply(init_weights)

    model.to(device)
    batch_size = 64
    epoch = 200

    myloss = nn.CrossEntropyLoss().to(device)

    init_lr = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr = init_lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
   
    train_dataset = TrainDataset()

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    ) 

    for i in range(epoch):

        model.train()  
        total_loss = 0
        correct = 0

        print("-----{} epoch for training-----".format(i + 1))

        for data in train_dataloader:
            dfs, label = data
            dfs = dfs.to(device)
            label = label.to(device)
            x = model(dfs)
            loss = myloss(x, label)
            loss = loss.requires_grad_()
            total_loss += loss * x.size(0)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (x.argmax(axis=1) == label).sum().item()
            
        print("Loss:{}, acc:{}".format(total_loss/train_dataset.__len__(), correct/train_dataset.__len__()))

        scheduler.step()

        if (i+1)%5==0:  # for every 5 epochs to save model and test
            save_path = "/srv/csj/tutorial_submit/model_file/" + "ResNet" + "_model_{}.pth".format(i+1)
            torch.save(model, save_path)
            acc, avg_loss = test(device, save_path)
            print("-----{} epoch for testing-----".format(i + 1))
            print("Loss:{}, acc:{}".format(acc, avg_loss))


train(device = torch.device("cuda:0"))