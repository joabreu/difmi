from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import random


def setup_reproducable(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def augment(image_size, image, label, count):
    augmentations = [
        torchvision.transforms.RandomRotation(degrees=0), # Identity
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomRotation(degrees=15),
    ]
    transformations = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
    ])
    aug_idx = torch.randint(0, len(augmentations), size=(count, ))
    return [(transformations(augmentations[aug_idx[i]](image)), label) for i in range(count)]


def get_data_from_path(image_size, augment_count, batch_size, path):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(image_size * 1.50)), # To account for Flip/Rotation/Etc.
    ])

    dataset = torchvision.datasets.ImageFolder(path, transform=transforms)
    aug_dataset = []
    for (image, label) in dataset:
        aug_dataset.extend(augment(image_size, image, label, count=augment_count))

    return DataLoader(
        aug_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()