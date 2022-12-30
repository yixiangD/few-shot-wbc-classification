from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_data_loader(data_path, train_transform, test_transform, batch_size):
    train_dataset = ImageFolder(root=f"{data_path}/train", transform=train_transform)
    test_dataset = ImageFolder(root=f"{data_path}/test", transform=test_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, test_loader
