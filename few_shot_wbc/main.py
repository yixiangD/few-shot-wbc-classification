import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from few_shot_wbc.datasets import get_data_loader
from few_shot_wbc.model import SimpleCNN, TorchVisionModel
from few_shot_wbc.transforms import test_transform, train_transform
from few_shot_wbc.utils import (  # vis_result,; vis_train,
    save_model,
    save_output,
    test,
    train,
    vis_acc_loss,
    vis_roc,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="start training")
    parser.add_argument("--vis", action="store_true", help="start training")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="load pretrained weights for torchvision models",
    )
    parser.add_argument(
        "--path",
        default="data/2class_5fold",
        type=str,
        help="path contains train/ and test/ folders",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simple",
        choices=[
            "simple",
            "alexnet",
            "vgg",
            "resnet",
            "densenet",
            "mobilenet",
            "resnext",
            "efficientnet",
        ],
        help="name of cnn model",
    )
    parser.add_argument(
        "--out_path", type=str, default="outputs", help="output folder path"
    )
    parser.add_argument(
        "--data_imb",
        type=str,
        default=None,
        choices=["reweight", "resample", "mixup"],
        help="data augmentation to deal with the imbalance of classes/labels",
    )
    parser.add_argument(
        "--imb_param", type=float, default=None, help="parameters for data augmentation"
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

    if args.data_imb in ["reweight", "mixup"]:
        if not args.imb_param:
            print(
                f"Model parameter for {args.data_imb} data augmentation needed but missing, exiting..."
            )
            exit()

    if not os.path.exists(args.path):
        print(f"Data path {args.path} not found, exiting...")
        exit()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computation device: {device}\n")
    nclass = 2
    # start the training
    if args.train:
        if args.model == "simple":
            model = SimpleCNN(nclass).to(device)
        else:
            model = TorchVisionModel(nclass, args.model, args.pretrained).to(device)
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
        new_weight = [1.0, 1.0]
        if args.data_imb == "reweight":
            new_weight = [1, args.imb_param]
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(new_weight).to(device))
        train_loader, test_loader = get_data_loader(
            args.path, train_transform, test_transform, args.batch_size, args.data_imb
        )
        train_loss, test_loss = [], []
        train_acc, test_acc = [], []
        for epoch in range(args.epochs):
            print(f"[INFO]: Epoch {epoch + 1} of {args.epochs}")
            train_epoch_loss, train_epoch_acc, train_epoch_prob = train(
                model, train_loader, optimizer, criterion, device, args
            )
            test_epoch_loss, test_epoch_acc, test_epoch_prob = test(
                model, test_loader, criterion, device
            )
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
        train_prob = np.vstack([x.cpu().numpy() for x in train_epoch_prob])
        test_prob = np.vstack([x.cpu().numpy() for x in test_epoch_prob])
        # save the trained model weights
        print(f"Saving model and loss history to {args.out_path}")
        np.save(os.path.join(args.out_path, "loss"), [train_loss, test_loss])
        np.save(os.path.join(args.out_path, "acc"), [train_acc, test_acc])
        save_model(args.out_path, args.epochs, model, optimizer, criterion)
        save_output(args.out_path, train_prob, test_prob)
        print("TRAINING COMPLETE")
        # save the loss and accuracy plots
    if args.vis:
        # checkpoint = torch.load(f"{out_path}/model.pth")
        clr = [
            "#808000",
            "#ff4500",
            "#c71585",
            "#00ff00",
            "#00ffff",
            "#0000ff",
            "#1e90ff",
        ]
        # vis_acc_loss(args.out_path, clr)
        # TODO : may need to overlay figures
        vis_roc(args.out_path, clr)


if __name__ == "__main__":
    main()
