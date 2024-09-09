import os.path
import argparse
from models import AdvancedCNN
from datasets import AnimalDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, mobilenet_v2, \
    MobileNet_V2_Weights
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from tqdm.autonotebook import tqdm
import shutil
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Train CNN model")
    parser.add_argument("--data-path", "-d", type=str, default="data/animals", help="Path to data")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Common size of image")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--lr", "-l", type=float, default=0.001, help="Optimizer's learning rate")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="Optimizer's momentum")
    parser.add_argument("--checkpoint-dir", "-c", type=str, default="trained_models", help="Place to save checkpoints")
    parser.add_argument("--log-dir", "-g", type=str, default="tensorboard", help="Place to save logging infor")
    parser.add_argument("--early_stopping_duration", "-s", type=int, default=5,
                        help="How many epochs should we wait until stopping the training when there is no improvement")
    args = parser.parse_args()
    return args


def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="winter")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size))
    ])
    train_dataset = AnimalDataset(root=args.data_path, is_train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )
    val_dataset = AnimalDataset(root=args.data_path, is_train=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )
    # model = AdvancedCNN(num_classes=len(train_dataset.categories))
    # model = resnet18(weights=ResNet18_Weights)
    # model.fc = nn.Linear(in_features=512, out_features=len(train_dataset.categories), bias=True)
    model = mobilenet_v2(weights=MobileNet_V2_Weights)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=len(train_dataset.categories), bias=True)
    model.to(device)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if os.path.isdir(args.checkpoint_dir):
        shutil.rmtree(args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(args.log_dir)

    best_accuracy = -1
    best_epoch = 0
    num_iters_per_epoch = len(train_dataloader)
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        train_loss = []
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            # Forward
            predictions = model(images)

            loss_value = criterion(predictions, labels)

            # Backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, args.epochs, loss_value.item()))
            train_loss.append(loss_value.item())
            writer.add_scalar("Train/Loss", np.mean(train_loss), global_step=epoch * num_iters_per_epoch + iter)

        model.eval()
        all_losses = []
        all_labels = []
        all_predictions = []
        # with torch.inference_mode():    # From pytorch 1.9
        progress_bar = tqdm(val_dataloader, colour="yellow")
        with torch.no_grad():
            for iter, (images, labels) in enumerate(progress_bar):
                images = images.to(device)
                labels = labels.to(device)
                # Forward
                outputs = model(images)  # shape: [B, N]  (32, 10)
                loss_value = criterion(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())
                all_losses.append(loss_value.item())
            accuracy = accuracy_score(all_labels, all_predictions)
            conf_matrix = confusion_matrix(all_labels, all_predictions)
            plot_confusion_matrix(writer, conf_matrix, train_dataset.categories, epoch)
            avg_loss = np.mean(all_losses)

            print("Epoch {}/{}. Loss {:0.4f}. Acc {:0.4f}".format(epoch + 1, args.epochs, avg_loss, accuracy))
            writer.add_scalar("Val/Loss", avg_loss, global_step=epoch)
            writer.add_scalar("Val/Accuracy", accuracy, global_step=epoch)
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_epoch": best_epoch,
                "best_accuracy": best_accuracy
            }
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best.pt"))

            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last.pt"))

        if epoch - best_epoch > args.early_stopping_duration:
            print("Stop the training process at epoch {}".format(epoch + 1))
            exit(0)


if __name__ == '__main__':
    args = get_args()
    train(args)
