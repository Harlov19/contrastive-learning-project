import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ContrastiveModel


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "./data"
CHECKPOINT_PATH = "./checkpoints/best_model.pt"
RESULTS_DIR = "./results"

BATCH_SIZE = 256
NUM_WORKERS = 2

# Number of test images used for visualization.
# 2,000 gives a clear visualization without being expensive.
NUM_SAMPLES = 2000
SEED = 42

REPRESENTATION_DIM = 128

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

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
    "truck",
]


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            f"Using GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    return device


# ============================================================
# DATA
# ============================================================

def get_test_loader():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD
        )
    ])

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return loader


# ============================================================
# MODEL
# ============================================================

def load_trained_model(device):

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    config = checkpoint.get("config", {})

    model = ContrastiveModel(
        representation_dim=config.get(
            "representation_dim",
            128
        ),
        projection_hidden_dim=config.get(
            "projection_hidden_dim",
            128
        ),
        projection_dim=config.get(
            "projection_dim",
            64
        )
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print(
        f"Loaded trained checkpoint "
        f"from epoch "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    return model


def create_random_model(device):

    model = ContrastiveModel(
        representation_dim=REPRESENTATION_DIM,
        projection_hidden_dim=128,
        projection_dim=64
    )

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    device,
    num_samples
):

    embeddings = []
    labels = []

    collected = 0

    for images, batch_labels in loader:

        images = images.to(device)

        # Use encoder representation h.
        h = model.encoder(images)

        remaining = num_samples - collected

        if h.size(0) > remaining:

            h = h[:remaining]
            batch_labels = batch_labels[:remaining]

        embeddings.append(
            h.cpu()
        )

        labels.append(
            batch_labels
        )

        collected += h.size(0)

        if collected >= num_samples:
            break

    embeddings = torch.cat(
        embeddings,
        dim=0
    )

    labels = torch.cat(
        labels,
        dim=0
    )

    return embeddings, labels


# ============================================================
# PCA
# ============================================================

def pca_2d(embeddings):

    """
    Compute a 2D PCA projection using SVD.

    PCA steps:

        X
        ↓
        Center features
        ↓
        SVD
        ↓
        First two principal components
        ↓
        2D representation
    """

    x = embeddings.numpy()

    # Center each feature.
    x = x - x.mean(axis=0, keepdims=True)

    # Singular Value Decomposition.
    u, s, vt = np.linalg.svd(
        x,
        full_matrices=False
    )

    # First two principal components.
    components = vt[:2]

    # Project samples into 2D.
    projected = x @ components.T

    explained_variance_ratio = (
        s ** 2
    ) / np.sum(s ** 2)

    return (
        projected,
        explained_variance_ratio[:2]
    )


# ============================================================
# PLOT
# ============================================================

def plot_pca(
    projected,
    labels,
    explained_variance,
    title,
    output_path
):

    plt.figure(
        figsize=(10, 8)
    )

    for class_id, class_name in enumerate(
        CIFAR10_CLASSES
    ):

        mask = labels == class_id

        plt.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=10,
            alpha=0.5,
            label=class_name
        )

    variance_1 = explained_variance[0] * 100
    variance_2 = explained_variance[1] * 100

    plt.xlabel(
        f"Principal Component 1 "
        f"({variance_1:.2f}% variance)"
    )

    plt.ylabel(
        f"Principal Component 2 "
        f"({variance_2:.2f}% variance)"
    )

    plt.title(title)

    plt.legend(
        markerscale=2,
        fontsize=8,
        ncol=2
    )

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PCA REPRESENTATION VISUALIZATION")
    print("=" * 70)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = get_device()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print()
    print("Loading CIFAR-10 test set...")

    loader = get_test_loader()

    print(
        f"Using {NUM_SAMPLES} test images."
    )

    # --------------------------------------------------------
    # Trained model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINED ENCODER")
    print("=" * 70)

    trained_model = load_trained_model(
        device
    )

    trained_embeddings, labels = (
        extract_embeddings(
            model=trained_model,
            loader=loader,
            device=device,
            num_samples=NUM_SAMPLES
        )
    )

    print(
        f"Embedding shape: "
        f"{trained_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Random model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RANDOM ENCODER")
    print("=" * 70)

    random_model = create_random_model(
        device
    )

    random_embeddings, random_labels = (
        extract_embeddings(
            model=random_model,
            loader=loader,
            device=device,
            num_samples=NUM_SAMPLES
        )
    )

    print(
        f"Embedding shape: "
        f"{random_embeddings.shape}"
    )

    # --------------------------------------------------------
    # PCA: trained
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPUTING TRAINED PCA")
    print("=" * 70)

    trained_2d, trained_variance = (
        pca_2d(
            trained_embeddings
        )
    )

    print(
        f"PC1 explained variance: "
        f"{trained_variance[0] * 100:.2f}%"
    )

    print(
        f"PC2 explained variance: "
        f"{trained_variance[1] * 100:.2f}%"
    )

    trained_plot_path = os.path.join(
        RESULTS_DIR,
        "pca_trained.png"
    )

    plot_pca(
        projected=trained_2d,
        labels=labels.numpy(),
        explained_variance=trained_variance,
        title=(
            "PCA of Learned Contrastive "
            "Representations"
        ),
        output_path=trained_plot_path
    )

    print(
        f"Saved: {trained_plot_path}"
    )

    # --------------------------------------------------------
    # PCA: random
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPUTING RANDOM PCA")
    print("=" * 70)

    random_2d, random_variance = (
        pca_2d(
            random_embeddings
        )
    )

    print(
        f"PC1 explained variance: "
        f"{random_variance[0] * 100:.2f}%"
    )

    print(
        f"PC2 explained variance: "
        f"{random_variance[1] * 100:.2f}%"
    )

    random_plot_path = os.path.join(
        RESULTS_DIR,
        "pca_random.png"
    )

    plot_pca(
        projected=random_2d,
        labels=random_labels.numpy(),
        explained_variance=random_variance,
        title=(
            "PCA of Random Encoder "
            "Representations"
        ),
        output_path=random_plot_path
    )

    print(
        f"Saved: {random_plot_path}"
    )

    # --------------------------------------------------------
    # Save numerical PCA information
    # --------------------------------------------------------

    pca_results_path = os.path.join(
        RESULTS_DIR,
        "pca_results.txt"
    )

    with open(
        pca_results_path,
        "w"
    ) as f:

        f.write(
            "PCA Representation Visualization\n"
        )

        f.write(
            "================================\n\n"
        )

        f.write(
            f"Number of samples: "
            f"{NUM_SAMPLES}\n"
        )

        f.write(
            f"Embedding dimension: "
            f"{trained_embeddings.shape[1]}\n\n"
        )

        f.write(
            "TRAINED ENCODER\n"
        )

        f.write(
            f"PC1 explained variance: "
            f"{trained_variance[0] * 100:.4f}%\n"
        )

        f.write(
            f"PC2 explained variance: "
            f"{trained_variance[1] * 100:.4f}%\n"
        )

        f.write(
            f"PC1 + PC2 explained variance: "
            f"{trained_variance.sum() * 100:.4f}%\n\n"
        )

        f.write(
            "RANDOM ENCODER\n"
        )

        f.write(
            f"PC1 explained variance: "
            f"{random_variance[0] * 100:.4f}%\n"
        )

        f.write(
            f"PC2 explained variance: "
            f"{random_variance[1] * 100:.4f}%\n"
        )

        f.write(
            f"PC1 + PC2 explained variance: "
            f"{random_variance.sum() * 100:.4f}%\n"
        )

    print()
    print(
        f"Saved: {pca_results_path}"
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PCA VISUALIZATION COMPLETE")
    print("=" * 70)

    print()
    print("Generated files:")

    print(
        f"  {trained_plot_path}"
    )

    print(
        f"  {random_plot_path}"
    )

    print(
        f"  {pca_results_path}"
    )


if __name__ == "__main__":
    main()