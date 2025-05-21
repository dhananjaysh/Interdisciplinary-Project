import torch
import torchvision
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import numpy as np


from df.datasets.images import Dataset
from df.datasets.dataset import Subset


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imsave("test_1.png", np.transpose(npimg, (1, 2, 0)))


if __name__ == "__main__":
    classes = (
        # images dataset classes
    )

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    train_data = Dataset(
        fdir="path_to_the_dataset", subset=Subset.TRAINING, transform=transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_data, batch_size=8, shuffle=False, num_workers=2
    )

    # get some random training images
    dataiter = iter(train_data_loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(8)))
