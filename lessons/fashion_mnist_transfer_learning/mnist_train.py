import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_ds, test_ds


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


def train(model, device, loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print(f"epoch {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}] loss: {loss.item():.4f}")


def test(model, device, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
    acc = correct / len(loader.dataset)
    print(f"test accuracy: {acc:.4f}")
    return acc


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("using device:", device)

    train_ds, test_ds = load_mnist()

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_start = time.time()
    for epoch in range(1, 11):
        epoch_start = time.time()
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        print(f"epoch {epoch} took {time.time() - epoch_start:.2f}s")
    print(f"total training time: {time.time() - total_start:.2f}s")

    torch.save(model.state_dict(), "mnist_mlp.pt")
    print("model saved to mnist_mlp.pt")


if __name__ == "__main__":
    main()
