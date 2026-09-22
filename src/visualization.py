import matplotlib.pyplot as plt
from torchvision import datasets, transforms


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


def show_samples(num_samples=16):

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=transforms.ToTensor()
    )

    plt.figure(figsize=(8, 8))

    for i in range(num_samples):

        image, label = dataset[i]

        # Convert [C, H, W] → [H, W, C]
        image = image.permute(1, 2, 0)

        plt.subplot(4, 4, i + 1)
        plt.imshow(image)
        plt.title(CIFAR10_CLASSES[label])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_samples()