import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Directory where CIFAR-10 will be stored
DATA_DIR = "./data"


# CIFAR-10 class names
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


def get_datasets():
    """
    Download and load the CIFAR-10 dataset.

    Returns:
        train_dataset: CIFAR-10 training dataset
        test_dataset: CIFAR-10 test dataset
    """

    # Convert PIL images to PyTorch tensors.
    # Output shape: [C, H, W] = [3, 32, 32]
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
    """
    Create PyTorch DataLoaders for the training and test sets.

    Args:
        batch_size: Number of images in each batch.
        num_workers: Number of subprocesses used for loading data.

    Returns:
        train_loader
        test_loader
    """

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


if __name__ == "__main__":

    # Load datasets
    train_dataset, test_dataset = get_datasets()

    print("=" * 50)
    print("CIFAR-10 DATASET")
    print("=" * 50)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples:     {len(test_dataset)}")

    # Inspect one image
    image, label = train_dataset[0]

    print(f"Image shape:      {image.shape}")
    print(f"Image dtype:      {image.dtype}")
    print(f"Label:            {label}")
    print(f"Class:            {CIFAR10_CLASSES[label]}")

    # Create DataLoaders
    train_loader, test_loader = get_dataloaders(
        batch_size=128,
        num_workers=2
    )

    # Get one batch
    images, labels = next(iter(train_loader))

    print()
    print("=" * 50)
    print("FIRST TRAINING BATCH")
    print("=" * 50)

    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")