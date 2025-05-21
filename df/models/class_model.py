import torch
import torch.nn as nn
from pathlib import Path


class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def save(self, save_dir: Path, suffix=None):
        """
        Saves the model, adds suffix to filename if given
        """
        filename = save_dir

        if suffix is not None:
            filename += suffix

        torch.save(self.state_dict(), filename)

    def load(self, path):
        """
        Loads model from path
        Does not work with transfer model
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(path, map_location=device))

