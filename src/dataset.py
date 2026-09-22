import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from augmentations import ContrastiveTransform


DATA_DIR = "./data"

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


class ContrastiveCIFAR10(Dataset):
    """
    CIFAR-10 dataset for contrastive learning.

    Each sample produces two independently augmented
    views of the same original image.
    """

    def __init__(self, train=True):
        self.dataset = datasets.CIFAR10(
            root=DATA_DIR,
            train=train,
            download=True
        )

        self.transform = ContrastiveTransform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image, label = self.dataset[index]

        view_1, view_2 = self.transform(image)

        return view_1, view_2, label


def get_datasets():

    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


def get_dataloaders(batch_size=128, num_workers=2):

    train_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


def get_contrastive_dataloader(
    batch_size=128,
    num_workers=2
):

    dataset = ContrastiveCIFAR10(train=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    return loader


if __name__ == "__main__":

    print("=" * 50)
    print("CIFAR-10 DATASET")
    print("=" * 50)

    train_dataset, test_dataset = get_datasets()

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples:     {len(test_dataset)}")

    image, label = train_dataset[0]

    print(f"Image shape:      {image.shape}")
    print(f"Image dtype:      {image.dtype}")
    print(f"Label:            {label}")
    print(f"Class:            {CIFAR10_CLASSES[label]}")

    train_loader, test_loader = get_dataloaders(
        batch_size=128,
        num_workers=2
    )

    images, labels = next(iter(train_loader))

    print()
    print("=" * 50)
    print("FIRST STANDARD TRAINING BATCH")
    print("=" * 50)

    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")

    contrastive_loader = get_contrastive_dataloader(
        batch_size=128,
        num_workers=2
    )

    view_1, view_2, labels = next(iter(contrastive_loader))

    print()
    print("=" * 50)
    print("FIRST CONTRASTIVE BATCH")
    print("=" * 50)

    print(f"View 1 shape:      {view_1.shape}")
    print(f"View 2 shape:      {view_2.shape}")
    print(f"Labels shape:      {labels.shape}")
    print(f"View 1 dtype:      {view_1.dtype}")
    print(f"View 2 dtype:      {view_2.dtype}")