import torch
from torchvision import transforms


class ContrastiveTransform:
    """
    Generate two independently augmented views
    from the same input image.
    """

    def __init__(self, image_size=32):

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=image_size,
                scale=(0.6, 1.0)
            ),

            transforms.RandomHorizontalFlip(),

            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1
                    )
                ],
                p=0.8
            ),

            transforms.RandomGrayscale(p=0.2),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])

    def __call__(self, image):
        """
        Generate two independent augmented views.
        """

        view_1 = self.transform(image)
        view_2 = self.transform(image)

        return view_1, view_2


if __name__ == "__main__":

    from torchvision import datasets

    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True
    )

    transform = ContrastiveTransform()

    image, label = dataset[0]

    view_1, view_2 = transform(image)

    print("=" * 50)
    print("CONTRASTIVE AUGMENTATION TEST")
    print("=" * 50)

    print(f"Original image size: {image.size}")
    print(f"View 1 shape:        {view_1.shape}")
    print(f"View 2 shape:        {view_2.shape}")
    print(f"View 1 dtype:        {view_1.dtype}")
    print(f"View 2 dtype:        {view_2.dtype}")
    print(f"Label:               {label}")