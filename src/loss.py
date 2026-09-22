import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss
    (NT-Xent), commonly used in contrastive learning.

    Given two augmented views:

        z1 = embeddings from view 1
        z2 = embeddings from view 2

    Each sample in z1 has exactly one positive sample
    in z2, while all other samples act as negatives.
    """

    def __init__(self, temperature=0.5):

        super().__init__()

        self.temperature = temperature

    def forward(self, z1, z2):

        batch_size = z1.size(0)

        # Normalize embeddings.
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Combine both views.
        z = torch.cat([z1, z2], dim=0)

        # Pairwise cosine similarity.
        similarity = torch.matmul(z, z.T)

        # Scale by temperature.
        similarity = similarity / self.temperature

        # Remove self-similarity from the matrix.
        mask = torch.eye(
            2 * batch_size,
            device=z.device,
            dtype=torch.bool
        )

        similarity = similarity.masked_fill(mask, float("-inf"))

        # Positive pair indices.
        #
        # z1[i] -> z2[i]
        # z2[i] -> z1[i]
        targets = torch.arange(
            batch_size,
            device=z.device
        )

        targets = torch.cat([
            targets + batch_size,
            targets
        ])

        # Cross entropy treats the positive pair as
        # the correct class among all other samples.
        loss = F.cross_entropy(
            similarity,
            targets
        )

        return loss

if __name__ == "__main__":

    print("=" * 50)
    print("NT-XENT LOSS BEHAVIOR TEST")
    print("=" * 50)

    batch_size = 8
    embedding_dim = 64

    loss_fn = NTXentLoss(
        temperature=0.5
    )

    # --------------------------------------------------
    # Test 1: Random positive pairs
    # --------------------------------------------------

    z1_random = torch.randn(
        batch_size,
        embedding_dim
    )

    z2_random = torch.randn(
        batch_size,
        embedding_dim
    )

    random_loss = loss_fn(
        z1_random,
        z2_random
    )

    print()
    print("Test 1: Random positive pairs")
    print(f"Loss: {random_loss.item():.4f}")

    # --------------------------------------------------
    # Test 2: Similar positive pairs
    # --------------------------------------------------

    z1_similar = torch.randn(
        batch_size,
        embedding_dim
    )

    # Add small noise to create a similar representation.
    z2_similar = z1_similar + 0.05 * torch.randn(
        batch_size,
        embedding_dim
    )

    similar_loss = loss_fn(
        z1_similar,
        z2_similar
    )

    print()
    print("Test 2: Similar positive pairs")
    print(f"Loss: {similar_loss.item():.4f}")

    # --------------------------------------------------
    # Compare
    # --------------------------------------------------

    print()
    print("=" * 50)
    print("COMPARISON")
    print("=" * 50)

    print(f"Random-pair loss:  {random_loss.item():.4f}")
    print(f"Similar-pair loss: {similar_loss.item():.4f}")

    if similar_loss < random_loss:
        print()
        print("PASS: Similar positive pairs produce lower loss.")
    else:
        print()
        print("WARNING: Loss behavior is unexpected.")

    print()
    print("=" * 50)
    print("LOSS BEHAVIOR TEST COMPLETE")
    print("=" * 50)