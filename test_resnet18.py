
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
import os

from torchvision.models import resnet18  
from df.models.class_model import DeepClassifier
from df.metrics import Accuracy
from df.datasets.dataset import Subset
from df.models.cnn import CNN
from df.models.vit import VIT

def test(args):

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_data = Dataset(fdir="dataset",
    subset=Subset.TEST,
    transform=transform)
    test_data_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=128,
    shuffle=False,
    num_workers=4
)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_test_data = len(test_data)

    model = DeepClassifier(resnet18())
    model.load(args.path_to_trained_model)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    test_metric = Accuracy(classes=test_data.classes)

    ### Below implement testing loop and print final loss
    ### and metrics to terminal after testing is finished

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)
            test_metric.update(outputs, labels)

    avg_loss = total_loss / num_test_data
    print(f"\nTest Loss: {avg_loss:.6f}")
    print(test_metric)


if __name__ == "__main__":
    
    args = argparse.ArgumentParser(description="Training")
    args.add_argument(
        "-d", "--gpu_id", default="5", type=str, help="index of which GPU to use"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0
    args.path_to_trained_model = "saved_models/resnet/best_model_epoch_30.pt"

    test(args)
