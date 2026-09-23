import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Encoder
# ============================================================

class Encoder(nn.Module):
    """
    CNN encoder for CIFAR-10.

    Maps a 32x32 RGB image to a fixed-dimensional
    representation vector h.
    """

    def __init__(self, representation_dim=128):
        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # 32x32 -> 32x32
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 32x32 -> 16x16
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 16x16 -> 8x8
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 8x8 -> 4x4
            # ------------------------------------------------

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

        x = torch.flatten(
            x,
            start_dim=1
        )

        h = self.fc(x)

        return h


# ============================================================
# Projection Head
# ============================================================

class ProjectionHead(nn.Module):
    """
    MLP projection head used by the contrastive objective.

    Maps the encoder representation h to a lower-dimensional
    normalized embedding z.
    """

    def __init__(
        self,
        input_dim=128,
        hidden_dim=128,
        output_dim=64
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                output_dim
            )
        )

    def forward(self, x):

        z = self.net(x)

        # Normalize embeddings before contrastive learning.
        z = F.normalize(
            z,
            dim=1
        )

        return z


# ============================================================
# Contrastive Model
# ============================================================

class ContrastiveModel(nn.Module):
    """
    Complete contrastive learning model.

    With projection head:

        Image
          |
          v
        Encoder
          |
          v
        Representation h
          |
          v
     Projection Head
          |
          v
     Normalized z
          |
          v
    Contrastive Loss


    Without projection head:

        Image
          |
          v
        Encoder
          |
          v
     Representation h
          |
          v
    Normalize h
          |
          v
    Contrastive Loss


    The encoder representation h is always returned and should
    be used for downstream representation evaluation.
    """

    def __init__(
        self,
        representation_dim=128,
        projection_hidden_dim=128,
        projection_dim=64,
        use_projection=True
    ):
        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = Encoder(
            representation_dim=representation_dim
        )

        # ----------------------------------------------------
        # Projection-head configuration
        # ----------------------------------------------------

        self.use_projection = use_projection

        if self.use_projection:

            self.projector = ProjectionHead(
                input_dim=representation_dim,
                hidden_dim=projection_hidden_dim,
                output_dim=projection_dim
            )

        else:

            self.projector = None

    def forward(self, x):

        # ----------------------------------------------------
        # Encoder representation
        # ----------------------------------------------------

        h = self.encoder(x)

        # ----------------------------------------------------
        # Projection head
        # ----------------------------------------------------

        if self.use_projection:

            z = self.projector(h)

        else:

            # Without a projection head, use the encoder
            # representation directly for contrastive learning.
            z = F.normalize(
                h,
                dim=1
            )

        return h, z


# ============================================================
# Model Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CONTRASTIVE MODEL TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Test with projection head
    # --------------------------------------------------------

    print()
    print("1. Model WITH projection head")
    print("-" * 60)

    model_with_projection = ContrastiveModel(
        use_projection=True
    )

    print(model_with_projection)

    # Simulate a batch of CIFAR-10 images.
    x = torch.randn(
        8,
        3,
        32,
        32
    )

    h_with, z_with = model_with_projection(x)

    print()

    print(
        f"Input shape:          {x.shape}"
    )

    print(
        f"Representation h:     {h_with.shape}"
    )

    print(
        f"Projection z:         {z_with.shape}"
    )

    print(
        f"h norm (sample 0):    "
        f"{h_with[0].norm().item():.4f}"
    )

    print(
        f"z norm (sample 0):    "
        f"{z_with[0].norm().item():.4f}"
    )

    # --------------------------------------------------------
    # Test without projection head
    # --------------------------------------------------------

    print()
    print("2. Model WITHOUT projection head")
    print("-" * 60)

    model_without_projection = ContrastiveModel(
        use_projection=False
    )

    print(model_without_projection)

    h_without, z_without = model_without_projection(x)

    print()

    print(
        f"Input shape:          {x.shape}"
    )

    print(
        f"Representation h:     {h_without.shape}"
    )

    print(
        f"Contrastive z:         {z_without.shape}"
    )

    print(
        f"h norm (sample 0):     "
        f"{h_without[0].norm().item():.4f}"
    )

    print(
        f"z norm (sample 0):     "
        f"{z_without[0].norm().item():.4f}"
    )

    print()

    print("=" * 60)
    print("MODEL TEST COMPLETE")
    print("=" * 60)

