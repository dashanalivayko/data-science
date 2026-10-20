"""Fashion-MNIST с фильтрацией до 5 классов + обучение MLP (PyTorch).

Оставляем: T-shirt/top (0), Trouser (1), Pullover (2), Dress (3), Sneaker (7).
Метки затем переиндексируются в диапазон 0..4, чтобы их можно было
использовать напрямую в CrossEntropyLoss/softmax на 5 выходов.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

FASHION_MNIST_MEAN, FASHION_MNIST_STD = 0.2860, 0.3530

# Классы, которые нужно оставить (оригинальные индексы Fashion-MNIST)
KEEP_CLASSES = [0, 1, 2, 3, 7]  # T-shirt/top, Trouser, Pullover, Dress, Sneaker
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Sneaker"]

# Старый индекс -> новый индекс (0..4)
LABEL_MAP = {old: new for new, old in enumerate(KEEP_CLASSES)}


def filter_dataset(ds):
    keep_set = set(KEEP_CLASSES)
    indices = [i for i, label in enumerate(ds.targets.tolist()) if label in keep_set]
    subset = Subset(ds, indices)

    # переиндексация меток 0..4 через remap targets
    ds.targets = torch.tensor([
        LABEL_MAP[label] if label in keep_set else -1
        for label in ds.targets.tolist()
    ])
    return subset


def load_fashion_mnist_5classes():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
    ])
    train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    train_subset = filter_dataset(train_ds)
    test_subset = filter_dataset(test_ds)
    return train_subset, test_subset


class MLP(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
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
        if batch_idx % 100 == 0:
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


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("using device:", device)

    train_subset, test_subset = load_fashion_mnist_5classes()
    print(f"train: {len(train_subset)} примеров, test: {len(test_subset)} примеров")
    print("классы:", dict(enumerate(CLASS_NAMES)))

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=256)

    model = MLP(num_classes=len(KEEP_CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_start = time.time()
    for epoch in range(1, 11):
        epoch_start = time.time()
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        print(f"epoch {epoch} took {time.time() - epoch_start:.2f}s")
    print(f"total training time: {time.time() - total_start:.2f}s")

    torch.save(model.state_dict(), "fashion_mnist_5classes_mlp.pt")
    print("model saved to fashion_mnist_5classes_mlp.pt")
