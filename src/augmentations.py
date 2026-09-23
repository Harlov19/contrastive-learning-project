from torchvision import transforms


class ContrastiveTransform:
    """
    Generate two independently augmented views
    from the same input image.

    Supports two augmentation strengths:
        - weak
        - strong

    The strong configuration preserves the original
    augmentation pipeline used by the main experiment.
    """

    def __init__(self, image_size=32, strength="strong"):

        if strength == "weak":

            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)
                )
            ])

        elif strength == "strong":

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

                transforms.RandomGrayscale(
                    p=0.2
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)
                )
            ])

        else:
            raise ValueError(
                f"Unknown augmentation strength: {strength}. "
                f"Expected 'weak' or 'strong'."
            )

        self.strength = strength

    def __call__(self, image):
        """
        Generate two independent augmented views
        from the same original image.
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

    image, label = dataset[0]

    print("=" * 60)
    print("CONTRASTIVE AUGMENTATION TEST")
    print("=" * 60)

    for strength in ["weak", "strong"]:

        transform = ContrastiveTransform(
            strength=strength
        )

        view_1, view_2 = transform(image)

        print()
        print(f"Augmentation strength: {strength}")
        print(f"Original image size:   {image.size}")
        print(f"View 1 shape:          {view_1.shape}")
        print(f"View 2 shape:          {view_2.shape}")
        print(f"View 1 dtype:          {view_1.dtype}")
        print(f"View 2 dtype:          {view_2.dtype}")
        print(f"Label:                 {label}")

    print()
    print("=" * 60)
    print("AUGMENTATION TEST COMPLETE")
    print("=" * 60)
