from trainer import fit
from datasets import TripletMNIST
from torchvision import datasets, transforms
from networks import EmbeddingNet, TripletNet
import torch
from torch import nn


def main():
    perc_transforms=transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # normalize,
    ])
    train_dataset = datasets.MNIST("/home2/lgfm95/mnist/", train=True, transform=perc_transforms,
                                                               target_transform=None, download=False)
    train_dataset = TripletMNIST(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8, shuffle=False,
        num_workers=4, pin_memory=True)
    val_dataset = datasets.MNIST("/home2/lgfm95/mnist/", train=False, transform=perc_transforms,
                                                               target_transform=None, download=False)
    val_dataset = TripletMNIST(train_dataset)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=8, shuffle=False,
        num_workers=4, pin_memory=True)

    embedding_net = EmbeddingNet()
    model = TripletNet(embedding_net).cuda()

    criterion = nn.TripletMarginLoss(margin=1.0).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 30, eta_min=0.001)

    fit(train_loader, val_loader, model, criterion, optimizer, lr_scheduler, 30, True, 2)


if __name__ == '__main__':
    main()
