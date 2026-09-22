import os

import torch
import torch.nn.functional as F
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

# Retrieval evaluation
K_VALUES = [1, 5]

# Number of queries used for quantitative evaluation
NUM_QUERIES = 1000

# Number of queries shown in visualization
NUM_VISUALIZATION_QUERIES = 5

# Number of retrieved images displayed per query
NUM_RETRIEVALS_TO_DISPLAY = 5

SEED = 42

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

        print(
            "CUDA not available. Using CPU."
        )

    return device


# ============================================================
# DATA
# ============================================================

def get_test_dataset():

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

    return dataset


def get_test_loader(dataset):

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )


# ============================================================
# MODELS
# ============================================================

def load_trained_model(device):

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

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
        checkpoint[
            "model_state_dict"
        ]
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

        # Use the encoder representation h,
        # not the projection z.
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
# RETRIEVAL
# ============================================================

@torch.no_grad()
def retrieve_neighbors(
    query_embeddings,
    database_embeddings,
    database_labels,
    k=5
):

    query_embeddings = F.normalize(
        query_embeddings,
        dim=1
    )

    database_embeddings = F.normalize(
        database_embeddings,
        dim=1
    )

    similarities = torch.mm(
        query_embeddings,
        database_embeddings.T
    )

    values, indices = torch.topk(
        similarities,
        k=k,
        dim=1
    )

    retrieved_labels = (
        database_labels[indices]
    )

    return (
        values,
        indices,
        retrieved_labels
    )


def precision_at_k(
    retrieved_labels,
    query_labels,
    k
):

    top_k_labels = retrieved_labels[
        :, :k
    ]

    correct = (
        top_k_labels ==
        query_labels.unsqueeze(1)
    )

    precision = (
        correct.float()
        .mean(dim=1)
    )

    return precision.mean().item()


def retrieval_accuracy_at_k(
    retrieved_labels,
    query_labels,
    k
):

    top_k_labels = retrieved_labels[
        :, :k
    ]

    correct = (
        top_k_labels ==
        query_labels.unsqueeze(1)
    )

    # At least one correct item.
    accuracy = (
        correct.any(dim=1)
        .float()
        .mean()
        .item()
    )

    return accuracy


# ============================================================
# QUERY SELECTION
# ============================================================

def select_queries(
    embeddings,
    labels,
    num_queries,
    seed=42
):

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    indices = torch.randperm(
        embeddings.size(0),
        generator=generator
    )[:num_queries]

    return (
        embeddings[indices],
        labels[indices],
        indices
    )


# ============================================================
# VISUALIZATION
# ============================================================

def denormalize(image):

    mean = torch.tensor(
        CIFAR10_MEAN
    ).view(3, 1, 1)

    std = torch.tensor(
        CIFAR10_STD
    ).view(3, 1, 1)

    image = image * std + mean

    return image.clamp(
        0,
        1
    )


def create_retrieval_visualization(
    dataset,
    model,
    device,
    query_indices,
    output_path,
    title
):

    # --------------------------------------------------------
    # Collect all database embeddings
    # --------------------------------------------------------

    loader = get_test_loader(
        dataset
    )

    database_embeddings, database_labels = (
        extract_embeddings(
            model=model,
            loader=loader,
            device=device
        )
    )

    # --------------------------------------------------------
    # Get query images
    # --------------------------------------------------------

    query_images = []
    query_labels = []

    for index in query_indices:

        image, label = dataset[
            int(index)
        ]

        query_images.append(
            image
        )

        query_labels.append(
            label
        )

    query_embeddings = (
        database_embeddings[
            query_indices
        ]
    )

    query_labels_tensor = torch.tensor(
        query_labels
    )

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    _, neighbor_indices, retrieved_labels = (
        retrieve_neighbors(
            query_embeddings=query_embeddings,
            database_embeddings=database_embeddings,
            database_labels=database_labels,
            k=NUM_RETRIEVALS_TO_DISPLAY + 1
        )
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    rows = len(query_indices)
    cols = NUM_RETRIEVALS_TO_DISPLAY + 1

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(15, 3 * rows)
    )

    if rows == 1:

        axes = axes.reshape(
            1,
            -1
        )

    for row in range(rows):

        # ----------------------------------------------------
        # Query image
        # ----------------------------------------------------

        query_image = denormalize(
            query_images[row]
        )

        axes[row, 0].imshow(
            query_image.permute(
                1,
                2,
                0
            )
        )

        axes[row, 0].set_title(
            "QUERY\n"
            + CIFAR10_CLASSES[
                query_labels[row]
            ],
            fontweight="bold"
        )

        axes[row, 0].axis("off")

        # ----------------------------------------------------
        # Retrieved images
        # ----------------------------------------------------

        displayed = 0

        for retrieval_position in range(
            NUM_RETRIEVALS_TO_DISPLAY + 1
        ):

            database_index = int(
                neighbor_indices[
                    row,
                    retrieval_position
                ]
            )

            # Skip query image itself.
            if database_index == int(
                query_indices[row]
            ):

                continue

            image, label = dataset[
                database_index
            ]

            image = denormalize(
                image
            )

            column = displayed + 1

            is_correct = (
                label ==
                query_labels[row]
            )

            axes[row, column].imshow(
                image.permute(
                    1,
                    2,
                    0
                )
            )

            axes[row, column].set_title(
                f"Top-{displayed + 1}\n"
                f"{CIFAR10_CLASSES[label]}"
            )

            axes[row, column].axis(
                "off"
            )

            displayed += 1

            if displayed >= (
                NUM_RETRIEVALS_TO_DISPLAY
            ):

                break

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    trained_results,
    random_results
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    path = os.path.join(
        RESULTS_DIR,
        "retrieval_results.txt"
    )

    with open(
        path,
        "w"
    ) as f:

        f.write(
            "Contrastive Learning "
            "Retrieval Evaluation\n"
        )

        f.write(
            "====================================\n\n"
        )

        f.write(
            f"Number of queries: "
            f"{NUM_QUERIES}\n"
        )

        f.write(
            "Similarity: cosine similarity\n"
        )

        f.write(
            "Representation: encoder output h\n\n"
        )

        f.write(
            "PRECISION@K\n"
        )

        f.write(
            "-----------\n"
        )

        for k in K_VALUES:

            f.write(
                f"Precision@{k}\n"
            )

            f.write(
                f"  Random encoder: "
                f"{random_results[k]['precision'] * 100:.2f}%\n"
            )

            f.write(
                f"  Trained encoder: "
                f"{trained_results[k]['precision'] * 100:.2f}%\n"
            )

            improvement = (
                trained_results[k]["precision"]
                -
                random_results[k]["precision"]
            )

            f.write(
                f"  Improvement: "
                f"{improvement * 100:.2f} "
                f"percentage points\n\n"
            )

        f.write(
            "RETRIEVAL ACCURACY@K\n"
        )

        f.write(
            "---------------------\n"
        )

        for k in K_VALUES:

            f.write(
                f"Accuracy@{k}\n"
            )

            f.write(
                f"  Random encoder: "
                f"{random_results[k]['accuracy'] * 100:.2f}%\n"
            )

            f.write(
                f"  Trained encoder: "
                f"{trained_results[k]['accuracy'] * 100:.2f}%\n\n"
            )

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "NEAREST-NEIGHBOR RETRIEVAL EVALUATION"
    )
    print("=" * 70)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    torch.manual_seed(SEED)

    device = get_device()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("Loading CIFAR-10 test dataset...")

    dataset = get_test_dataset()

    print(
        f"Test samples: "
        f"{len(dataset)}"
    )

    loader = get_test_loader(
        dataset
    )

    # --------------------------------------------------------
    # Trained embeddings
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TRAINED ENCODER EMBEDDINGS"
    )
    print("=" * 70)

    trained_model = load_trained_model(
        device
    )

    trained_embeddings, labels = (
        extract_embeddings(
            model=trained_model,
            loader=loader,
            device=device
        )
    )

    print(
        f"Embedding shape: "
        f"{trained_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Random embeddings
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RANDOM ENCODER EMBEDDINGS"
    )
    print("=" * 70)

    random_model = create_random_model(
        device
    )

    random_embeddings, random_labels = (
        extract_embeddings(
            model=random_model,
            loader=loader,
            device=device
        )
    )

    print(
        f"Embedding shape: "
        f"{random_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Select identical queries
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SELECTING RETRIEVAL QUERIES"
    )
    print("=" * 70)

    (
        trained_queries,
        query_labels,
        query_indices
    ) = select_queries(
        embeddings=trained_embeddings,
        labels=labels,
        num_queries=NUM_QUERIES,
        seed=SEED
    )

    # Use identical indices for random model.
    random_queries = random_embeddings[
        query_indices
    ]

    print(
        f"Queries: "
        f"{len(query_indices)}"
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    trained_results = {}
    random_results = {}

    for k in K_VALUES:

        print()
        print(
            f"Running trained retrieval "
            f"for k={k}..."
        )

        _, _, trained_retrieved_labels = (
            retrieve_neighbors(
                query_embeddings=trained_queries,
                database_embeddings=trained_embeddings,
                database_labels=labels,
                k=k + 1
            )
        )

        # Remove the query itself.
        # The first nearest neighbor is normally
        # the query itself.
        trained_retrieved_labels = (
            trained_retrieved_labels[:, 1:]
        )

        trained_precision = (
            precision_at_k(
                retrieved_labels=trained_retrieved_labels,
                query_labels=query_labels,
                k=k
            )
        )

        trained_accuracy = (
            retrieval_accuracy_at_k(
                retrieved_labels=trained_retrieved_labels,
                query_labels=query_labels,
                k=k
            )
        )

        trained_results[k] = {
            "precision": trained_precision,
            "accuracy": trained_accuracy
        }

        print(
            f"Trained Precision@{k}: "
            f"{trained_precision * 100:.2f}%"
        )

        print(
            f"Trained Accuracy@{k}: "
            f"{trained_accuracy * 100:.2f}%"
        )

        print()
        print(
            f"Running random retrieval "
            f"for k={k}..."
        )

        _, _, random_retrieved_labels = (
            retrieve_neighbors(
                query_embeddings=random_queries,
                database_embeddings=random_embeddings,
                database_labels=random_labels,
                k=k + 1
            )
        )

        random_retrieved_labels = (
            random_retrieved_labels[:, 1:]
        )

        random_precision = (
            precision_at_k(
                retrieved_labels=random_retrieved_labels,
                query_labels=query_labels,
                k=k
            )
        )

        random_accuracy = (
            retrieval_accuracy_at_k(
                retrieved_labels=random_retrieved_labels,
                query_labels=query_labels,
                k=k
            )
        )

        random_results[k] = {
            "precision": random_precision,
            "accuracy": random_accuracy
        }

        print(
            f"Random Precision@{k}: "
            f"{random_precision * 100:.2f}%"
        )

        print(
            f"Random Accuracy@{k}: "
            f"{random_accuracy * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RETRIEVAL COMPARISON"
    )
    print("=" * 70)

    print()

    print(
        f"{'Metric':25s}"
        f"{'Random':>15s}"
        f"{'Trained':>15s}"
    )

    print("-" * 55)

    for k in K_VALUES:

        print(
            f"{'Precision@' + str(k):25s}"
            f"{random_results[k]['precision'] * 100:>14.2f}%"
            f"{trained_results[k]['precision'] * 100:>14.2f}%"
        )

        print(
            f"{'Accuracy@' + str(k):25s}"
            f"{random_results[k]['accuracy'] * 100:>14.2f}%"
            f"{trained_results[k]['accuracy'] * 100:>14.2f}%"
        )

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------

    results_path = save_results(
        trained_results=trained_results,
        random_results=random_results
    )

    print()
    print(
        f"Results saved to: "
        f"{results_path}"
    )

    # --------------------------------------------------------
    # Visualization queries
    # --------------------------------------------------------

    visualization_indices = (
        query_indices[
            :NUM_VISUALIZATION_QUERIES
        ]
    )

    print()
    print("=" * 70)
    print(
        "CREATING TRAINED RETRIEVAL VISUALIZATION"
    )
    print("=" * 70)

    trained_visualization_path = os.path.join(
        RESULTS_DIR,
        "retrieval_trained.png"
    )

    create_retrieval_visualization(
        dataset=dataset,
        model=trained_model,
        device=device,
        query_indices=visualization_indices,
        output_path=trained_visualization_path,
        title=(
            "Nearest-Neighbor Retrieval "
            "Using Learned Representations"
        )
    )

    print(
        f"Saved: "
        f"{trained_visualization_path}"
    )

    print()
    print("=" * 70)
    print(
        "CREATING RANDOM RETRIEVAL VISUALIZATION"
    )
    print("=" * 70)

    random_visualization_path = os.path.join(
        RESULTS_DIR,
        "retrieval_random.png"
    )

    create_retrieval_visualization(
        dataset=dataset,
        model=random_model,
        device=device,
        query_indices=visualization_indices,
        output_path=random_visualization_path,
        title=(
            "Nearest-Neighbor Retrieval "
            "Using Random Representations"
        )
    )

    print(
        f"Saved: "
        f"{random_visualization_path}"
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RETRIEVAL EVALUATION COMPLETE"
    )
    print("=" * 70)

    print()
    print("Generated files:")

    print(
        f"  {results_path}"
    )

    print(
        f"  {trained_visualization_path}"
    )

    print(
        f"  {random_visualization_path}"
    )


if __name__ == "__main__":
    main()