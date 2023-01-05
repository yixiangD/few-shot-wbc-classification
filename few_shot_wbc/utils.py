import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

matplotlib.style.use("ggplot")


def train(model, trainloader, optimizer, criterion, device):
    model.train()
    # print("Training")
    train_loss = 0.0
    train_correct = 0
    count = 0
    probs = []
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        count += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        probs.append(
            torch.cat((outputs.data, torch.reshape(labels, (labels.size(0), 1))), 1)
        )
        train_correct += (preds == labels).sum().item()
        # backward propagation: compute gradient for differiatable params
        loss.backward()
        # update parameters with new grad
        optimizer.step()
    # loss and accuracy for the complete epoch
    epoch_loss = train_loss / count
    epoch_acc = 100.0 * (train_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc, probs


def test(model, testloader, criterion, device):
    model.eval()
    # print("Testing")
    test_loss = 0.0
    test_correct = 0
    count = 0
    probs = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            count += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            probs.append(
                torch.cat((outputs.data, torch.reshape(labels, (labels.size(0), 1))), 1)
            )
            test_correct += (preds == labels).sum().item()
    epoch_loss = test_loss / count
    epoch_acc = 100.0 * (test_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc, probs


def save_model(out_path, epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
        },
        f"{out_path}/model.pth",
    )


def save_plots(out_path, train_acc, test_acc, train_loss, test_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color="green", linestyle="-", label="train accuracy")
    plt.plot(test_acc, color="blue", linestyle="-", label="test accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{out_path}/accuracy.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", linestyle="-", label="train loss")
    plt.plot(test_loss, color="red", linestyle="-", label="test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{out_path}/loss.png")


def save_output(out_path, train_prob, test_prob):
    """
    Function to save tabular data output; probs and labels
    """
    # accuracy plots
    cols = [str(x) for x in range(np.shape(train_prob)[1] - 1)] + ["y_test"]
    df = pd.DataFrame(train_prob, columns=cols)
    df.to_csv(f"{out_path}/train_probs.csv", index=False)
    df = pd.DataFrame(test_prob, columns=cols)
    df.to_csv(f"{out_path}/test_probs.csv", index=False)
