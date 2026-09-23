import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ContrastiveModel


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "./data"
CHECKPOINT_PATH = "./checkpoints/best_model.pt"

BATCH_SIZE = 256
NUM_WORKERS = 2

# Reproducibility
SEED = 42

# k-NN evaluation
K = 5
QUERY_BATCH_SIZE = 128

# Representation geometry
GEOMETRY_SAMPLES = 2000
GEOMETRY_SEED = 42

RESULTS_DIR = "./results"


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
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    """
    Set random seeds for reproducible evaluation.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

        print(
            "CUDA not available. Using CPU."
        )

    return device


# ============================================================
# DATA
# ============================================================

def get_evaluation_datasets():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD
        )
    ])

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


def get_dataloaders():

    train_dataset, test_dataset = (
        get_evaluation_datasets()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, test_loader


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(device):

    if not os.path.exists(CHECKPOINT_PATH):

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    config = checkpoint.get(
        "config",
        {}
    )

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
        f"Loaded checkpoint from epoch "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Checkpoint loss: "
        f"{checkpoint.get('loss', 'unknown')}"
    )

    return model


def create_random_model(device):

    model = ContrastiveModel(
        representation_dim=128,
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
    device
):

    embeddings = []
    labels = []

    for batch_idx, (
        images,
        batch_labels
    ) in enumerate(loader):

        images = images.to(device)

        # IMPORTANT:
        # Use encoder representation h,
        # not projection embedding z.
        h = model.encoder(images)

        embeddings.append(
            h.cpu()
        )

        labels.append(
            batch_labels
        )

        if (batch_idx + 1) % 20 == 0:

            print(
                f"Embedding batch "
                f"[{batch_idx + 1}/"
                f"{len(loader)}]"
            )

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
# k-NN
# ============================================================

@torch.no_grad()
def knn_predict(
    train_embeddings,
    train_labels,
    test_embeddings,
    k=5,
    query_batch_size=128
):

    # Cosine similarity
    train_embeddings = F.normalize(
        train_embeddings,
        dim=1
    )

    test_embeddings = F.normalize(
        test_embeddings,
        dim=1
    )

    predictions = []

    num_test = test_embeddings.size(0)

    for start in range(
        0,
        num_test,
        query_batch_size
    ):

        end = min(
            start + query_batch_size,
            num_test
        )

        queries = test_embeddings[
            start:end
        ]

        similarities = torch.mm(
            queries,
            train_embeddings.T
        )

        _, nearest_indices = torch.topk(
            similarities,
            k=k,
            dim=1
        )

        nearest_labels = train_labels[
            nearest_indices
        ]

        batch_predictions = []

        for labels in nearest_labels:

            counts = torch.bincount(
                labels,
                minlength=10
            )

            prediction = torch.argmax(
                counts
            )

            batch_predictions.append(
                prediction
            )

        batch_predictions = torch.stack(
            batch_predictions
        )

        predictions.append(
            batch_predictions
        )

        print(
            f"k-NN queries "
            f"[{end}/{num_test}]"
        )

    predictions = torch.cat(
        predictions
    )

    return predictions


def evaluate_knn(
    train_embeddings,
    train_labels,
    test_embeddings,
    test_labels,
    k=5
):

    predictions = knn_predict(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        test_embeddings=test_embeddings,
        k=k,
        query_batch_size=QUERY_BATCH_SIZE
    )

    accuracy = (
        predictions == test_labels
    ).float().mean().item()

    return accuracy, predictions


def calculate_class_accuracy(
    predictions,
    labels
):

    results = {}

    for class_id, class_name in enumerate(
        CIFAR10_CLASSES
    ):

        mask = labels == class_id

        class_accuracy = (
            predictions[mask] == labels[mask]
        ).float().mean().item()

        results[class_name] = class_accuracy

    return results


# ============================================================
# REPRESENTATION GEOMETRY
# ============================================================

def select_geometry_samples(
    embeddings,
    labels,
    num_samples=2000,
    seed=42
):

    if num_samples > embeddings.size(0):

        num_samples = embeddings.size(0)

    generator = torch.Generator()

    generator.manual_seed(seed)

    indices = torch.randperm(
        embeddings.size(0),
        generator=generator
    )[:num_samples]

    return (
        embeddings[indices],
        labels[indices]
    )


@torch.no_grad()
def calculate_representation_geometry(
    embeddings,
    labels
):

    """
    Measure how the embedding space is organized.

    Metrics:

    1. Same-class cosine similarity
       Average similarity between samples
       belonging to the same semantic class.

    2. Different-class cosine similarity
       Average similarity between samples
       belonging to different classes.

    3. Similarity gap
       Same-class similarity minus
       different-class similarity.

    4. Average pairwise cosine similarity
       Overall similarity of the embedding space.

    5. Average per-dimension variance
       Useful for detecting representation collapse.
    """

    # Normalize embeddings so dot product
    # becomes cosine similarity.
    normalized = F.normalize(
        embeddings,
        dim=1
    )

    # Full pairwise cosine similarity matrix.
    similarity_matrix = torch.mm(
        normalized,
        normalized.T
    )

    n = embeddings.size(0)

    # Only use upper triangle.
    # This removes:
    # - self-similarity
    # - duplicate pair (i,j) / (j,i)
    upper_indices = torch.triu_indices(
        n,
        n,
        offset=1
    )

    pair_similarities = similarity_matrix[
        upper_indices[0],
        upper_indices[1]
    ]

    pair_labels_1 = labels[
        upper_indices[0]
    ]

    pair_labels_2 = labels[
        upper_indices[1]
    ]

    same_class_mask = (
        pair_labels_1 == pair_labels_2
    )

    different_class_mask = (
        pair_labels_1 != pair_labels_2
    )

    same_class_similarities = (
        pair_similarities[
            same_class_mask
        ]
    )

    different_class_similarities = (
        pair_similarities[
            different_class_mask
        ]
    )

    same_class_mean = (
        same_class_similarities.mean().item()
    )

    same_class_std = (
        same_class_similarities.std().item()
    )

    different_class_mean = (
        different_class_similarities.mean().item()
    )

    different_class_std = (
        different_class_similarities.std().item()
    )

    similarity_gap = (
        same_class_mean -
        different_class_mean
    )

    average_pairwise_similarity = (
        pair_similarities.mean().item()
    )

    # Average variance across embedding dimensions.
    per_dimension_variance = (
        embeddings.var(
            dim=0,
            unbiased=False
        )
    )

    average_embedding_variance = (
        per_dimension_variance.mean().item()
    )

    return {
        "same_class_mean": same_class_mean,
        "same_class_std": same_class_std,
        "different_class_mean":
            different_class_mean,
        "different_class_std":
            different_class_std,
        "similarity_gap": similarity_gap,
        "average_pairwise_similarity":
            average_pairwise_similarity,
        "average_embedding_variance":
            average_embedding_variance,
        "num_samples": n,
        "num_same_class_pairs":
            int(
                same_class_mask.sum().item()
            ),
        "num_different_class_pairs":
            int(
                different_class_mask.sum().item()
            )
    }


def print_geometry_results(
    name,
    results
):

    print()
    print("=" * 70)
    print(
        f"{name.upper()} REPRESENTATION GEOMETRY"
    )
    print("=" * 70)

    print(
        f"Samples: "
        f"{results['num_samples']}"
    )

    print(
        f"Same-class cosine similarity: "
        f"{results['same_class_mean']:.4f} "
        f"± {results['same_class_std']:.4f}"
    )

    print(
        f"Different-class cosine similarity: "
        f"{results['different_class_mean']:.4f} "
        f"± {results['different_class_std']:.4f}"
    )

    print(
        f"Similarity gap: "
        f"{results['similarity_gap']:.4f}"
    )

    print(
        f"Average pairwise similarity: "
        f"{results['average_pairwise_similarity']:.4f}"
    )

    print(
        f"Average embedding variance: "
        f"{results['average_embedding_variance']:.6f}"
    )

    print(
        f"Same-class pairs: "
        f"{results['num_same_class_pairs']:,}"
    )

    print(
        f"Different-class pairs: "
        f"{results['num_different_class_pairs']:,}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    trained_accuracy,
    random_accuracy,
    class_accuracy,
    embedding_dim,
    trained_geometry,
    random_geometry
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results_path = os.path.join(
        RESULTS_DIR,
        "representation_evaluation.txt"
    )

    absolute_improvement = (
        trained_accuracy -
        random_accuracy
    )

    relative_improvement = (
        absolute_improvement /
        random_accuracy
    ) * 100

    geometry_gap_improvement = (
        trained_geometry["similarity_gap"] -
        random_geometry["similarity_gap"]
    )

    with open(
        results_path,
        "w"
    ) as f:

        f.write(
            "Contrastive Learning "
            "Representation Evaluation\n"
        )

        f.write(
            "====================================\n\n"
        )

        f.write(
            f"Checkpoint: "
            f"{CHECKPOINT_PATH}\n"
        )

        f.write(
            f"Seed: {SEED}\n"
        )

        f.write(
            f"k: {K}\n"
        )

        f.write(
            f"Embedding dimension: "
            f"{embedding_dim}\n"
        )

        f.write(
            f"Geometry samples: "
            f"{GEOMETRY_SAMPLES}\n\n"
        )

        # ----------------------------------------------------
        # k-NN
        # ----------------------------------------------------

        f.write(
            "1. k-NN CLASSIFICATION\n"
        )

        f.write(
            "-----------------------\n"
        )

        f.write(
            f"Random encoder: "
            f"{random_accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Trained encoder: "
            f"{trained_accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Absolute improvement: "
            f"{absolute_improvement * 100:.2f} "
            f"percentage points\n"
        )

        f.write(
            f"Relative improvement: "
            f"{relative_improvement:.2f}%\n\n"
        )

        # ----------------------------------------------------
        # Class accuracy
        # ----------------------------------------------------

        f.write(
            "Class-wise accuracy\n"
        )

        f.write(
            "-------------------\n"
        )

        for class_name, value in (
            class_accuracy.items()
        ):

            f.write(
                f"{class_name}: "
                f"{value * 100:.2f}%\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        f.write(
            "2. REPRESENTATION GEOMETRY\n"
        )

        f.write(
            "---------------------------\n\n"
        )

        f.write(
            "RANDOM ENCODER\n"
        )

        f.write(
            f"Same-class cosine similarity: "
            f"{random_geometry['same_class_mean']:.6f}\n"
        )

        f.write(
            f"Different-class cosine similarity: "
            f"{random_geometry['different_class_mean']:.6f}\n"
        )

        f.write(
            f"Similarity gap: "
            f"{random_geometry['similarity_gap']:.6f}\n"
        )

        f.write(
            f"Average pairwise similarity: "
            f"{random_geometry['average_pairwise_similarity']:.6f}\n"
        )

        f.write(
            f"Average embedding variance: "
            f"{random_geometry['average_embedding_variance']:.8f}\n\n"
        )

        f.write(
            "TRAINED ENCODER\n"
        )

        f.write(
            f"Same-class cosine similarity: "
            f"{trained_geometry['same_class_mean']:.6f}\n"
        )

        f.write(
            f"Different-class cosine similarity: "
            f"{trained_geometry['different_class_mean']:.6f}\n"
        )

        f.write(
            f"Similarity gap: "
            f"{trained_geometry['similarity_gap']:.6f}\n"
        )

        f.write(
            f"Average pairwise similarity: "
            f"{trained_geometry['average_pairwise_similarity']:.6f}\n"
        )

        f.write(
            f"Average embedding variance: "
            f"{trained_geometry['average_embedding_variance']:.8f}\n\n"
        )

        f.write(
            "GEOMETRY IMPROVEMENT\n"
        )

        f.write(
            f"Similarity-gap improvement: "
            f"{geometry_gap_improvement:.6f}\n"
        )

    return results_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "CONTRASTIVE REPRESENTATION EVALUATION"
    )

    print("=" * 70)

    # Set the seed before model construction.
    # This makes the random baseline deterministic.
    set_seed(SEED)

    print(
        f"Random seed: {SEED}"
    )

    device = get_device()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print()
    print("Loading datasets...")

    train_loader, test_loader = (
        get_dataloaders()
    )

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Test samples:     "
        f"{len(test_loader.dataset)}"
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print()
    print("Loading trained model...")

    model = load_model(device)

    # --------------------------------------------------------
    # Trained embeddings
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "EXTRACTING TRAINED TRAINING EMBEDDINGS"
    )
    print("=" * 70)

    train_embeddings, train_labels = (
        extract_embeddings(
            model=model,
            loader=train_loader,
            device=device
        )
    )

    print()
    print(
        f"Training embedding shape: "
        f"{train_embeddings.shape}"
    )

    print()
    print("=" * 70)
    print(
        "EXTRACTING TRAINED TEST EMBEDDINGS"
    )
    print("=" * 70)

    test_embeddings, test_labels = (
        extract_embeddings(
            model=model,
            loader=test_loader,
            device=device
        )
    )

    print()
    print(
        f"Test embedding shape: "
        f"{test_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Trained k-NN
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TRAINED ENCODER k-NN EVALUATION"
    )
    print("=" * 70)

    trained_accuracy, predictions = (
        evaluate_knn(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            k=K
        )
    )

    print()
    print(
        f"Trained encoder "
        f"k-NN (k={K}) accuracy: "
        f"{trained_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Class-wise accuracy
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASS-WISE ACCURACY")
    print("=" * 70)

    class_accuracy = (
        calculate_class_accuracy(
            predictions,
            test_labels
        )
    )

    for class_name, value in (
        class_accuracy.items()
    ):

        print(
            f"{class_name:12s}: "
            f"{value * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Random baseline
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RANDOM ENCODER BASELINE")
    print("=" * 70)

    print()
    print(
        "Creating untrained model..."
    )

    # Because set_seed(SEED) was called before this
    # model was constructed, its initialization is
    # reproducible.
    random_model = create_random_model(
        device
    )

    print()
    print(
        "Extracting random training embeddings..."
    )

    random_train_embeddings, _ = (
        extract_embeddings(
            model=random_model,
            loader=train_loader,
            device=device
        )
    )

    print()
    print(
        f"Random training embedding shape: "
        f"{random_train_embeddings.shape}"
    )

    print()
    print(
        "Extracting random test embeddings..."
    )

    random_test_embeddings, _ = (
        extract_embeddings(
            model=random_model,
            loader=test_loader,
            device=device
        )
    )

    print()
    print(
        f"Random test embedding shape: "
        f"{random_test_embeddings.shape}"
    )

    print()
    print(
        "Running k-NN on random embeddings..."
    )

    random_accuracy, _ = evaluate_knn(
        train_embeddings=random_train_embeddings,
        train_labels=train_labels,
        test_embeddings=random_test_embeddings,
        test_labels=test_labels,
        k=K
    )

    print()
    print(
        f"Random encoder "
        f"k-NN (k={K}) accuracy: "
        f"{random_accuracy * 100:.2f}%"
    )

    absolute_improvement = (
        trained_accuracy -
        random_accuracy
    )

    relative_improvement = (
        absolute_improvement /
        random_accuracy
    ) * 100

    print()
    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    print()
    print(
        f"Random encoder:       "
        f"{random_accuracy * 100:.2f}%"
    )

    print(
        f"Trained encoder:      "
        f"{trained_accuracy * 100:.2f}%"
    )

    print(
        f"Absolute improvement: "
        f"{absolute_improvement * 100:.2f} "
        f"percentage points"
    )

    print(
        f"Relative improvement: "
        f"{relative_improvement:.2f}%"
    )

    # --------------------------------------------------------
    # Representation geometry
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SELECTING GEOMETRY EVALUATION SAMPLES"
    )
    print("=" * 70)

    (
        trained_geometry_embeddings,
        geometry_labels
    ) = select_geometry_samples(
        embeddings=test_embeddings,
        labels=test_labels,
        num_samples=GEOMETRY_SAMPLES,
        seed=GEOMETRY_SEED
    )

    (
        random_geometry_embeddings,
        random_geometry_labels
    ) = select_geometry_samples(
        embeddings=random_test_embeddings,
        labels=test_labels,
        num_samples=GEOMETRY_SAMPLES,
        seed=GEOMETRY_SEED
    )

    print()
    print(
        "Calculating trained representation geometry..."
    )

    trained_geometry = (
        calculate_representation_geometry(
            embeddings=trained_geometry_embeddings,
            labels=geometry_labels
        )
    )

    print_geometry_results(
        "trained",
        trained_geometry
    )

    print()
    print(
        "Calculating random representation geometry..."
    )

    random_geometry = (
        calculate_representation_geometry(
            embeddings=random_geometry_embeddings,
            labels=random_geometry_labels
        )
    )

    print_geometry_results(
        "random",
        random_geometry
    )

    # --------------------------------------------------------
    # Geometry comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "REPRESENTATION GEOMETRY COMPARISON"
    )
    print("=" * 70)

    print()

    print(
        f"{'Metric':40s}"
        f"{'Random':>15s}"
        f"{'Trained':>15s}"
    )

    print("-" * 70)

    print(
        f"{'Same-class cosine similarity':40s}"
        f"{random_geometry['same_class_mean']:>15.4f}"
        f"{trained_geometry['same_class_mean']:>15.4f}"
    )

    print(
        f"{'Different-class cosine similarity':40s}"
        f"{random_geometry['different_class_mean']:>15.4f}"
        f"{trained_geometry['different_class_mean']:>15.4f}"
    )

    print(
        f"{'Similarity gap':40s}"
        f"{random_geometry['similarity_gap']:>15.4f}"
        f"{trained_geometry['similarity_gap']:>15.4f}"
    )

    print(
        f"{'Average pairwise similarity':40s}"
        f"{random_geometry['average_pairwise_similarity']:>15.4f}"
        f"{trained_geometry['average_pairwise_similarity']:>15.4f}"
    )

    print(
        f"{'Average embedding variance':40s}"
        f"{random_geometry['average_embedding_variance']:>15.6f}"
        f"{trained_geometry['average_embedding_variance']:>15.6f}"
    )

    geometry_gap_improvement = (
        trained_geometry["similarity_gap"]
        - random_geometry["similarity_gap"]
    )

    print()

    print(
        f"Similarity-gap improvement: "
        f"{geometry_gap_improvement:.4f}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_path = save_results(
        trained_accuracy=trained_accuracy,
        random_accuracy=random_accuracy,
        class_accuracy=class_accuracy,
        embedding_dim=train_embeddings.shape[1],
        trained_geometry=trained_geometry,
        random_geometry=random_geometry
    )

    print()
    print(
        f"Results saved to: "
        f"{results_path}"
    )

    print()
    print("=" * 70)
    print(
        "EVALUATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()