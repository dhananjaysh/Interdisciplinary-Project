
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
import torch.optim as optim
from torchvision.models import resnet18
from pathlib import Path
import os

from df.models.class_model import (
    DeepClassifier,
)
from df.models.cnn import CNN
from df.metrics import Accuracy
from df.trainer import ImgClassificationTrainer
from df.datasets.dataset import Subset
from df.models.vit import VIT



def train(args):

    ###  this function trains a specific model 

    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_data = Dataset(
        fdir=args.data_dir, subset=Subset.TRAINING, transform=train_transform
    )
    val_data = Dataset(
        fdir=args.data_dir, subset=Subset.VALIDATION, transform=val_transform
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepClassifier(resnet18())
    model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=0.001, amsgrad=True, weight_decay=1e-4
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    model_save_dir = Path("saved_models/resnet")
    model_save_dir.mkdir(exist_ok=True)

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    trainer = ImgClassificationTrainer(
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        args.num_epochs,
        model_save_dir,
        batch_size=128,  
        val_frequency=val_frequency,
    )
    trainer.train()


if __name__ == "__main__":
    
    args = argparse.ArgumentParser(description="Training")
    args.add_argument("--data_dir", type=str, required=True)
    args.add_argument("--num_epochs", type=int, default=30)
    args.add_argument(
        "-d", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0

    train(args) 
