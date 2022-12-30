import argparse
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from few_shot_wbc.datasets import get_data_loader
from few_shot_wbc.model import SimpleCNN
from few_shot_wbc.transforms import test_transform, train_transform
from few_shot_wbc.utils import save_model, save_plots, test, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="start training")
    parser.add_argument(
        "--path",
        default="data/2class_5fold",
        help="path contains train/ and test/ folders",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="number of epochs to train our network for",
    )
    parser.add_argument(
        "--batch_size", type=int, default=28, help="number of batch size"
    )
    parser.add_argument("--lr", type=int, default=1e-3, help="learning rate")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Data path {args.path} not found, exiting...")
        exit()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computation device: {device}\n")
    model = SimpleCNN().to(device)
    print(model)
    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.")
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # loss function
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = get_data_loader(
        args.path, train_transform, test_transform, args.batch_size
    )
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    # start the training
    if args.train:
        for epoch in range(args.epochs):
            print(f"[INFO]: Epoch {epoch + 1} of {args.epochs}")
            train_epoch_loss, train_epoch_acc = train(
                model, train_loader, optimizer, criterion, device
            )
            test_epoch_loss, test_epoch_acc = test(model, test_loader, criterion, device)
            train_loss.append(train_epoch_loss)
            test_loss.append(test_epoch_loss)
            train_acc.append(train_epoch_acc)
            test_acc.append(test_epoch_acc)
            print(
                f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}"
            )
            print(f"Test loss: {test_epoch_loss:.3f}, test acc: {test_epoch_acc:.3f}")
            print("-" * 50)
            time.sleep(5)
        # save the trained model weights
        save_model(args.epochs, model, optimizer, criterion)
        # save the loss and accuracy plots
        save_plots(train_acc, test_acc, train_loss, test_loss)
        print("TRAINING COMPLETE")


if __name__ == "__main__":
    main()
