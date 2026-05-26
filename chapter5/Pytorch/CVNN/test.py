import torch
from torch import nn
from chapter5.Pytorch.CVNN.test_dataset import TestDataset


def test(device, model_path):
    test_dataset = TestDataset()
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False
    ) 

    model = torch.load(model_path)
    model.to(device)
    model.eval()

    myloss = nn.CrossEntropyLoss().to(device)

    correct = 0
    total_loss = 0
    for data in test_dataloader:
        csi, label = data
        csi, label = csi.to(device), label.to(device)
        x = model(csi)
        loss = myloss(x, label)
        total_loss += loss.detach().cpu().numpy()
        _, predicted = torch.max(x.data, 1)
        correct += (predicted == label).item()

    accuracy = correct/test_dataset.__len__()
    avg_loss = total_loss/test_dataset.__len__()

    return accuracy, avg_loss