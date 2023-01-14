from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from torchvision.datasets import ImageFolder


def get_data_loader(data_path, train_transform, test_transform, batch_size, data_imb):
    train_dataset = ImageFolder(root=f"{data_path}/train", transform=train_transform)
    test_dataset = ImageFolder(root=f"{data_path}/test", transform=test_transform)
    if data_imb == "resample":
        train_sampler = ImbalancedDatasetSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=1,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    return train_loader, test_loader
