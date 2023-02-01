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
from few_shot_wbc.utils import (
    save_model,
    save_output,
    test,
    train,
    vis_result,
    vis_train,
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
        for root, dir, file in os.walk(args.out_path):
            root, dir, file = root, dir, file
            break
        # clr = ["b", "g", "r", "k", "y"]
        clr = [
            "#808000",
            "#ff4500",
            "#c71585",
            "#00ff00",
            "#00ffff",
            "#0000ff",
            "#1e90ff",
        ]
        dir_norm = [x for x in dir if "pretrain" not in x]
        dir_pretrain = [x for x in dir if "pretrain" in x]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))

        for f, c in zip(dir_pretrain, clr):
            accs = np.load(os.path.join(args.out_path, f, "acc.npy"))
            losses = np.load(os.path.join(args.out_path, f, "loss.npy"))
            train_acc, test_acc = accs[0], accs[1]
            train_loss, test_loss = losses[0], losses[1]
            ax1.plot(np.arange(len(train_acc)), train_acc, linestyle="-", color=c)
            ax1.plot(np.arange(len(test_acc)), test_acc, linestyle="--", color=c)
            ax2.plot(
                np.arange(len(train_loss)),
                train_loss,
                linestyle="-",
                label=f"{f} train",
                color=c,
            )
            ax2.plot(
                np.arange(len(test_loss)),
                test_loss,
                linestyle="--",
                label=f"{f} test",
                color=c,
            )
            ax1.set_ylabel("Accuracy")
            ax2.set_ylabel("Loss (log scale)")
            ax1.set_xlabel("Epoch")
            ax2.set_xlabel("Epoch")
            ax2.set_yscale("log")
        fig.legend(loc="upper center", ncol=4)
        fig.tight_layout()
        fig.savefig(f"{args.out_path}/accuracy_pretrain.pdf")

        exit()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
        for f, c in zip(dir_norm, clr):
            accs = np.load(os.path.join(args.out_path, f, "acc.npy"))
            losses = np.load(os.path.join(args.out_path, f, "loss.npy"))
            train_acc, test_acc = accs[0], accs[1]
            train_loss, test_loss = losses[0], losses[1]
            ax1.plot(np.arange(len(train_acc)), train_acc, linestyle="-", color=c)
            ax1.plot(np.arange(len(test_acc)), test_acc, linestyle="--", color=c)
            ax2.plot(
                np.arange(len(train_loss)),
                train_loss,
                linestyle="-",
                label=f"{f} train",
                color=c,
            )
            ax2.plot(
                np.arange(len(test_loss)),
                test_loss,
                linestyle="--",
                label=f"{f} test",
                color=c,
            )
            ax1.set_ylabel("Accuracy")
            ax2.set_ylabel("Loss (log scale)")
            ax1.set_xlabel("Epoch")
            ax2.set_xlabel("Epoch")
            ax2.set_yscale("log")
        fig.legend(loc="upper center", ncol=4)
        fig.tight_layout()
        fig.savefig(f"{args.out_path}/accuracy_norm.pdf")
        # TODO : may need to overlay figures
        df = pd.read_csv(f"{args.out_path}/train_probs.csv")
        prefix = "train"
        vis_result(df, args.out_path, prefix)
        df = pd.read_csv(f"{args.out_path}/test_probs.csv")
        prefix = "test"
        vis_result(df, args.out_path, prefix)


if __name__ == "__main__":
    main()
