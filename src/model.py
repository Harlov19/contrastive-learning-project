import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    CNN encoder for CIFAR-10.

    Maps a 32x32 RGB image to a fixed-dimensional
    representation vector.
    """

    def __init__(self, representation_dim=128):

        super().__init__()

        self.features = nn.Sequential(

            # 32x32 -> 32x32
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(
            512,
            representation_dim
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        h = self.fc(x)

        return h


class ProjectionHead(nn.Module):
    """
    MLP projection head used by the contrastive objective.
    """

    def __init__(
        self,
        input_dim=128,
        hidden_dim=128,
        output_dim=64
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),

            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):

        z = self.net(x)

        # Normalize embeddings before contrastive learning.
        z = F.normalize(z, dim=1)

        return z


class ContrastiveModel(nn.Module):
    """
    Complete contrastive learning model.

    Image -> Encoder -> Representation h
                           |
                           ↓
                    Projection Head
                           |
                           ↓
                       Embedding z
    """

    def __init__(
        self,
        representation_dim=128,
        projection_hidden_dim=128,
        projection_dim=64
    ):

        super().__init__()

        self.encoder = Encoder(
            representation_dim=representation_dim
        )

        self.projector = ProjectionHead(
            input_dim=representation_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim
        )

    def forward(self, x):

        h = self.encoder(x)

        z = self.projector(h)

        return h, z


if __name__ == "__main__":

    print("=" * 50)
    print("CONTRASTIVE MODEL TEST")
    print("=" * 50)

    model = ContrastiveModel()

    print(model)

    # Simulate a batch of CIFAR-10 images.
    x = torch.randn(8, 3, 32, 32)

    h, z = model(x)

    print()
    print(f"Input shape:          {x.shape}")
    print(f"Representation h:     {h.shape}")
    print(f"Projection z:         {z.shape}")

    print()
    print(f"h norm (sample 0):    {h[0].norm().item():.4f}")
    print(f"z norm (sample 0):    {z[0].norm().item():.4f}")

    print()
    print("=" * 50)
    print("MODEL TEST COMPLETE")
    print("=" * 50)